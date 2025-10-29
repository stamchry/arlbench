"""Console script for arlbench."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import logging
import sys
import traceback
from typing import TYPE_CHECKING

import hydra
import jax
from arlbench.arlbench import run_arlbench

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="examples/configs", config_name="base")
def execute(cfg: DictConfig):
    """Helper function for nice logging and error handling."""
    logging.basicConfig(
        filename="job.log", format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if cfg.jax_enable_x64:
        logger.info("Enabling x64 support for JAX.")
        jax.config.update("jax_enable_x64", True)
    try:
        return run(cfg, logger)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def run(cfg: DictConfig, logger: logging.Logger):
    """Console script for arlbench."""
    objectives = run_arlbench(cfg, logger=logger)
    logger.info(f"Returned objectives: {objectives}")

    with open("./performance.csv", "w+") as f:
        f.write(str(objectives))
    with open("./done.txt", "w+") as f:
        f.write("yes")

    if "reward_mean" in objectives:
        objectives["performance"] = objectives.pop("reward_mean")
    if "runtime" in objectives:
        objectives["cost"] = objectives.pop("runtime")
        
    return objectives


if __name__ == "__main__":
    sys.exit(execute())  # pragma: no cover
