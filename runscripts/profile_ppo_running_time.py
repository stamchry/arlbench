"""Script to measure env_step_time, inference_time, and gradient_backprop_time for PPO."""

from __future__ import annotations
import sys
import time
import logging
import traceback

import hydra
import numpy as np
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
    # 5. Prepare TRAINING Environment (The Batch)
    # -------------------------------------------------------------------------
    rng = jax.random.PRNGKey(cfg.seed)
    
    # We want to measure the exact environment used for training (batch size = n_envs)
    target_env = algo.env 
    
    # Use the actual state from the runner (already initialized with n_envs)
    env_state = runner_state.env_state
    
    # Retrieve the observation batch (shape: [n_envs, obs_dim])
    # Note: Depending on implementation, obs might be in env_state or separate
    if hasattr(runner_state, 'env_state') and hasattr(runner_state.env_state, 'obs'):
         obs = runner_state.env_state.obs
    else:
         # Fallback: Reset the batch env
         rng, reset_rng = jax.random.split(rng)
         # algo.env.reset typically returns a batch of states/obs
         env_state, obs = target_env.reset(reset_rng)

    logger.info(f"Corrected Benchmark Obs Shape: {obs.shape}") # Should be (1024, ...) for Ant

    # Robust logging for observation shape
    if hasattr(obs, 'shape'):
        logger.info(f"Benchmark Observation Shape: {obs.shape}")
    else:
        logger.info(f"Benchmark Observation Type: {type(obs)} (Likely struct or tuple)")

    # -------------------------------------------------------------------------
    # 6. Helper Measurement Logic
    # -------------------------------------------------------------------------
    def measure_fn(name, fn, *args, n_warmup=100, n_iter=100000, **kwargs):
        logger.info(f"Measuring {name}...")
        try:
            # Warmup
            for _ in range(n_warmup):
                out = fn(*args, **kwargs)
                jax.block_until_ready(out)
            
            # Measurement - collect individual timings
            times = []
            for _ in range(n_iter):
                start = time.time()
                out = fn(*args, **kwargs)
                jax.block_until_ready(out)
                end = time.time()
                times.append(end - start)
            
            times = np.array(times)
            avg_time = np.mean(times)
            std_dev = np.std(times)
            cv = std_dev / avg_time if avg_time > 0 else 0.0  # Coefficient of Variation
            
            # Compute percentiles to show distribution
            p5, p25, p50, p75, p95 = np.percentile(times, [5, 25, 50, 75, 95])
            
            logger.info(f"  > Avg time for {name}: {avg_time:.8f} s (±{std_dev:.8f} s, CV={cv:.4f})")
            logger.info(f"     Percentiles [5%, 25%, 50%, 75%, 95%]: [{p5:.8f}, {p25:.8f}, {p50:.8f}, {p75:.8f}, {p95:.8f}] s")
            
            return avg_time, std_dev, cv, (p5, p25, p50, p75, p95)
        except Exception as e:
            logger.error(f"Failed to measure {name}: {e}")
            traceback.print_exc()
            return 0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # Metric 1: Inference Time
    # -------------------------------------------------------------------------
    
    inference_time, inference_std, inference_cv, inference_percentiles = measure_fn(
        "Inference (predict)",
        jax.jit(algo.predict, static_argnames=("deterministic",)),
        runner_state, obs, rng, deterministic=True
    )
    if inference_time == 0.0: inference_time = 1e-8

    # -------------------------------------------------------------------------
    # Metric 2: Environment Step Time (Batch)
    # -------------------------------------------------------------------------
    # Generate action batch
    rng, act_rng = jax.random.split(rng)
    action = algo.predict(runner_state, obs, act_rng, deterministic=True)
    
    def simple_step_fn(key, state, act):
        # Just step, don't reset!
        return target_env.step(state, act, key)

    env_step_time, env_step_std, env_step_cv, env_step_percentiles = measure_fn(
        "Environment Step (Batch)",
        jax.jit(simple_step_fn),
        rng, env_state, action
    )


    
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
    
    # If dtype is not a float (e.g. uint8), force it to float32 for random.normal
    gen_dtype = dtype
    if not np.issubdtype(dtype, np.floating) and not np.issubdtype(dtype, np.complexfloating):
        gen_dtype = jnp.float32

    dummy_batch = jax.random.normal(data_rng, (batch_size, *feature_shape), dtype=gen_dtype)

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

    backprop_time, backprop_std, backprop_cv, backprop_percentiles = measure_fn(
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
    logger.info(f"env_step_time          : {env_step_time:.8f} s (±{env_step_std:.8f} s, CV={env_step_cv:.4f})")
    logger.info(f"inference_time         : {inference_time:.8f} s (±{inference_std:.8f} s, CV={inference_cv:.4f})")
    logger.info(f"gradient_backprop_time : {backprop_time:.8f} s (±{backprop_std:.8f} s, CV={backprop_cv:.4f})")
    logger.info("="*60)

if __name__ == "__main__":
    sys.exit(main())