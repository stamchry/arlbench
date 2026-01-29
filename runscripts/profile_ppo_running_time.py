"""Script to measure env_step_time, inference_time, and gradient_backprop_time for PPO."""

from __future__ import annotations
import sys
import time
import logging
import traceback

import hydra
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf, DictConfig
from arlbench.autorl import AutoRLEnv
from hydra_plugins.hyper_smac.hyper_smac import read_additional_configs

# Register resolvers
OmegaConf.register_new_resolver("read_additional_configs", read_additional_configs, replace=True)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)
OmegaConf.register_new_resolver("len", lambda x: len(x), replace=True)

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if cfg.jax_enable_x64:
        logger.info("Enabling x64 support for JAX.")
        jax.config.update("jax_enable_x64", True)

    logger.info("Initializing AutoRLEnv for PPO Profiling...")

    # -------------------------------------------------------------------------
    # 1. Merge NAS Config Overrides
    # -------------------------------------------------------------------------
    if "nas_config" in cfg and cfg.nas_config:
        logger.info(f"Applying NAS Config overrides: {cfg.nas_config}")
        OmegaConf.set_struct(cfg.autorl, False)
        if "nas_config" not in cfg.autorl or cfg.autorl.nas_config is None:
            cfg.autorl.nas_config = {}
        cfg.autorl.nas_config = OmegaConf.merge(cfg.autorl.nas_config, cfg.nas_config)

    # -------------------------------------------------------------------------
    # 2. Initialize and Reset Environment
    # -------------------------------------------------------------------------
    env = AutoRLEnv(cfg.autorl)
    env.reset()

    # -------------------------------------------------------------------------
    # 3. Initialization Step (Warmup)
    # -------------------------------------------------------------------------
    logger.info("Running initialization step...")
    try:
        action = env.config_space.get_default_configuration()
        env.step(
            action, 
            n_total_timesteps=128, 
            n_eval_steps=1,
            n_eval_episodes=1
        )
    except Exception as e:
        logger.error(f"Initialization step failed. Benchmark cannot proceed. Error: {e}")
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # 4. Extract Internals for Benchmarking
    # -------------------------------------------------------------------------
    algo = env._algorithm
    algo_state = env._algorithm_state
    
    if algo_state is None:
        logger.error("Algorithm state is None even after env.step(). Aborting.")
        return

    runner_state = algo_state.runner_state
    if not hasattr(runner_state, "train_state"):
        logger.error("Runner state does not contain 'train_state'. Is this PPO?")
        return
    
    train_state = runner_state.train_state

    # -------------------------------------------------------------------------
    # 5. Prepare Evaluation Environment
    # -------------------------------------------------------------------------
    rng = jax.random.PRNGKey(cfg.seed)
    rng, reset_rng = jax.random.split(rng)
    
    # Check for eval_env (PPO standard)
    if not hasattr(algo, "eval_env") or algo.eval_env is None:
        logger.warning("eval_env missing, falling back to training env.")
        target_env = algo.env
    else:
        target_env = algo.eval_env
        
    # The GymnaxEnv wrapper in this codebase returns (env_state, obs)
    env_state, obs = target_env.reset(reset_rng)

    # Robust logging for observation shape
    if hasattr(obs, 'shape'):
        logger.info(f"Benchmark Observation Shape: {obs.shape}")
    else:
        logger.info(f"Benchmark Observation Type: {type(obs)} (Likely struct or tuple)")

    # -------------------------------------------------------------------------
    # 6. Helper Measurement Logic
    # -------------------------------------------------------------------------
    def measure_fn(name, fn, *args, n_warmup=10, n_iter=100, **kwargs):
        logger.info(f"Measuring {name}...")
        try:
            # Warmup
            for _ in range(n_warmup):
                out = fn(*args, **kwargs)
                jax.block_until_ready(out)
            
            # Measurement
            start = time.time()
            for _ in range(n_iter):
                out = fn(*args, **kwargs)
                jax.block_until_ready(out)
            end = time.time()
            
            avg_time = (end - start) / n_iter
            logger.info(f"  > Avg time for {name}: {avg_time:.8f} s")
            return avg_time
        except Exception as e:
            logger.error(f"Failed to measure {name}: {e}")
            traceback.print_exc()
            return 0.0

    # -------------------------------------------------------------------------
    # Metric 1: Inference Time
    # -------------------------------------------------------------------------
    
    inference_time = measure_fn(
        "Inference (predict)",
        jax.jit(algo.predict, static_argnames=("deterministic",)),
        runner_state, obs, rng, deterministic=True
    )
    if inference_time == 0.0: inference_time = 1e-8

    # -------------------------------------------------------------------------
    # Metric 2: Environment Step Time
    # -------------------------------------------------------------------------
    # Generate action for step
    rng, act_rng = jax.random.split(rng)
    action = algo.predict(runner_state, obs, act_rng, deterministic=True)
    
    # Abstract definition: def step(self, env_state: Any, action: Any, rng: PRNGKey)
    def env_step_with_fresh_inputs(rng_key):
        rng1, rng2 = jax.random.split(rng_key)
        state, obs = target_env.reset(rng1)
        action = algo.predict(runner_state, obs, rng2, deterministic=True)
        return target_env.step(state, action, rng2)

    env_step_time = measure_fn(
        "Environment Step",
        jax.jit(env_step_with_fresh_inputs),
        rng
    )

    # -------------------------------------------------------------------------
    # Metric 3: Gradient Backprop Time
    # -------------------------------------------------------------------------
    
    # 1. Determine Input Shape
    if hasattr(obs, 'shape'):
        obs_shape = obs.shape
        dtype = obs.dtype
    else:
        leaves = jax.tree_util.tree_leaves(obs)
        obs_shape = leaves[0].shape
        dtype = leaves[0].dtype

    batch_size = cfg.hp_config.get("minibatch_size", 32)
    feature_shape = obs_shape[1:] if len(obs_shape) > 1 else obs_shape 
    
    # 2. Use RANDOM data (Avoids Zero-optimization shortcuts)
    rng, data_rng = jax.random.split(rng)
    dummy_batch = jax.random.normal(data_rng, (batch_size, *feature_shape), dtype=dtype)

    # 3. Handle RNNs (Hidden States)
    init_hstate = None
    if hasattr(runner_state, 'env_state') and hasattr(runner_state.env_state, 'hidden'):
         init_hstate = runner_state.env_state.hidden

    def ppo_update_step(t_state, batch, h_state=None):
        def loss_fn(params):
            if h_state is not None:
                # Assuming RNN returns (hidden, (dist, value)) or similar
                _, (dist, value) = algo.network.apply(params, h_state, batch)
            else:
                dist, value = algo.network.apply(params, batch)
            
            # Make computation realistic: Calculate Log Prob
            dummy_action = dist.mode()
            log_prob = dist.log_prob(dummy_action)
            
            # Standard PPO-like scalar reduction
            loss = -jnp.mean(log_prob) + 0.5 * jnp.mean(jnp.square(value))
            return loss

        grads = jax.grad(loss_fn)(t_state.params)
        return t_state.apply_gradients(grads=grads)

    # Curry the function to match measure_fn signature
    if init_hstate is not None:
        h_batch = jax.tree_map(lambda x: jnp.repeat(x, batch_size, axis=0), init_hstate)
        update_fn = lambda ts, b: ppo_update_step(ts, b, h_batch)
    else:
        update_fn = lambda ts, b: ppo_update_step(ts, b, None)

    backprop_time = measure_fn(
        "Gradient Backprop (PPO)",
        jax.jit(update_fn),
        train_state, dummy_batch
    )

    # -------------------------------------------------------------------------
    # Report Results
    # -------------------------------------------------------------------------
    logger.info("="*60)
    logger.info(f"TIMING RESULTS for PPO on {cfg.autorl.get('env_name', 'unknown')}")
    if "nas_config" in cfg:
        logger.info(f"  Configured Architecture: {cfg.autorl.get('nas_config', {})}")
    logger.info("="*60)
    logger.info(f"env_step_time          : {env_step_time:.8f} s")
    logger.info(f"inference_time         : {inference_time:.8f} s")
    logger.info(f"gradient_backprop_time : {backprop_time:.8f} s")
    logger.info("="*60)

if __name__ == "__main__":
    sys.exit(main())