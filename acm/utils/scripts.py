"""Useful functions usually called in scripts."""

import argparse
import gc
import logging
import sys
from collections.abc import Callable
from multiprocessing import cpu_count
from pathlib import Path
from subprocess import check_output
from typing import Any

import numpy as np
import yaml

try:
    from jax import clear_caches  # pyright: ignore[reportMissingImports]
except ImportError:
    clear_caches = lambda: None  # noqa: E731

logger = logging.getLogger(__name__)


def detect_gpu() -> bool:
    """Detect if a GPU is available on the system."""
    try:
        check_output("nvidia-smi")  # noqa: S607
        logger.debug("GPU found!")
    except Exception:  # noqa: BLE001
        logger.debug("No GPU found")
        return False
    else:
        return True


def get_nthreads(nthread_per_cpu: int = 1) -> int:
    """Determine the number of threads available on the system."""
    if nthread_per_cpu < 1:
        raise ValueError("nthread_per_cpu must be bigger than 1")
    ncpu = cpu_count()
    logger.debug(f"Found {ncpu} CPUs")
    return ncpu * nthread_per_cpu


def load_parser_default(parser: argparse.ArgumentParser) -> dict:
    """
    Load default parameters to a parser, from a config file provided in a parser arguments.

    Expects at least a 'config' argument in the parser.
    """
    args, _ = parser.parse_known_args()
    if not hasattr(args, "config"):
        raise ValueError("parser_default_config requires a 'config' argument to work.")
    config_args = {}
    if args.config:  # None or False
        config = Path(args.config)
        with config.open("r") as f:
            _loaded = yaml.safe_load(f)
            config_args.update(**_loaded)
    logger.debug(f"Loaded parser with config: {config_args}")
    return config_args


def apply_parser_default(parser: argparse.ArgumentParser, config_args: dict) -> None:
    """Apply parser default values from a dictionary."""
    # Reset `required` attribute when provided from config file
    for action in parser._actions:
        if action.dest in config_args:
            action.required = False
    parser.set_defaults(**config_args)


def dump_config(parser: argparse.ArgumentParser) -> None:
    """
    Dump the current parser argument to stdout if args.dump_config is True.

    Expects at least a 'dump_config' argument in the parser.
    """
    args = parser.parse_args()
    if not hasattr(args, "dump_config"):
        raise ValueError(
            "parser_default_config requires a 'dump_config' argument to work."
        )
    if args.dump_config:  # None or False
        parser.print_help(sys.stdout)
        print("\nCurrent configuration:")  # noqa: T201
        print("----------------------")  # noqa: T201
        tmp_args = vars(args).copy()
        del tmp_args["config"]
        del tmp_args["dump_config"]
        for arg in tmp_args:
            print(f"{arg}: {getattr(args, arg)}")  # noqa: T201
        sys.exit(-1)


def retry(times: int, operation: Callable, *args, **kwargs) -> Any | None:  # noqa: ANN401
    """Run a function n times then fails with logged error."""
    name = getattr(operation, "__name__", "operation")
    if times < 1:
        raise ValueError(f"'times' must be >= 1, got {times}.")
    for i in range(times):
        try:
            logger.debug(f"Calling {name}, attempt {i + 1} of {times}")
            return operation(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 FIXME: catch jax exception type here
            logger.warning(f"Calling {name} failed with error: {e}")
            logger.info("Clearing cache and retrying...")
            clear_caches() # Either jax or a no-op lambda
            gc.collect()
    # Only runs when run reaches n
    logger.error(f"Calling {name} definitely failed after {times} times.")
    return None


class NumpyLoader(yaml.SafeLoader):
    """A YAML loader to allow numpy functions to be registered."""


def _np_arange(loader: Any, node: Any) -> np.ndarray:  # noqa: ANN401
    args = loader.construct_sequence(node)
    return np.arange(*args)


def _np_linspace(loader: Any, node: Any) -> np.ndarray:  # noqa: ANN401
    args = loader.construct_sequence(node)
    return np.linspace(*args)


NumpyLoader.add_constructor("!np.arange", _np_arange)
NumpyLoader.add_constructor("!np.linspace", _np_linspace)
