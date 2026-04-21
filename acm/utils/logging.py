"""Manage logging.

taken from https://github.com/cosmodesi/desilike/blob/main/desilike/utils.py

How to use it, two cases : module and script

For module, at the top of the module, add
'''
import logging
logger = logging.getLogger(__name__)


logger.info("This is an info message from the module.")
...
'''

For script, use this structure
'''
from acm.utils.logging import get_logger_for_script, setup_logging

# script logger  init
setup_logging()
logger = get_logger_for_script(__file__)
...

def example_function():
    logger.info("This is an info message from example_function.")

...

if __name__ == '__main__':

    logger.info("Start processing")
    <your processing code>
    logger.info("End processing")
'''

"""
import datetime
import logging
import sys
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import TextIO

NAME_LIB_GIT = __name__.split(".")[0]

logger = logging.getLogger(__name__)


def setup_logging(
    level: int | str = logging.INFO,
    stream: TextIO = sys.stdout,
    filename: str | None = None,
    filemode: str = "w",
    **kwargs,
) -> None:
    """
    Set up logging.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.
    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.
    filename : string, default=None
        If not ``None`` stream to file name.
    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.
    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level, str):
        level = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            self._style._fmt = (
                "[%09.2f] " % (time.time() - t0)
                + " %(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
            )
            super().format(record)
            msg = logging.Formatter.format(self, record)
            if record.message != "":
                parts = msg.split(record.message)
                msg = msg.replace("\n", "\n" + parts[0])
            return msg

    fmt = MyFormatter(datefmt="%m-%d %H:%M ")
    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)
    sys.excepthook = exception_handler
    logger.debug(
        f"Start logger at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"  # noqa: DTZ005
    )


def exception_handler(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None ) -> None:
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = "Exception"
    log = logging.getLogger(_logger_name)
    line = "=" * 100
    # log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical(
        "\n"
        + line
        + "\n"
        + "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        + line
    )
    if exc_type is KeyboardInterrupt:
        log.critical("Interrupted by the user.")
    else:
        log.critical("An error occurred.")


@contextmanager
def suppress_logging(enabled: bool = True, highest_level: int | str = logging.CRITICAL) -> Generator:
    """Context manager to temporarily suppress logging messages."""
    root = logging.getLogger()
    origin_level = root.getEffectiveLevel()
    if enabled:
        root.setLevel(highest_level)  # Keep only messages at or above the highest_level
    yield
    if enabled:
        root.setLevel(origin_level)


def get_logger_for_script(pfile: str | Path) -> logging.Logger:
    """Return a logger with root logger is defined by the path of the file.

    Problem for script the value of __name__ is "__main__",
    so we cannot use it to define the logger name.
    Instead, we can use the path of the file to define the logger name,
    so that we can have different loggers for different scripts.

    Note: this function should be called before setup_logging()

    Parameters
    ----------
    pfile: path of the file, so always call with __file__ value

    """
    str_logger = _get_logger_path(pfile)
    logger.debug(f"\nFull name script: {pfile}\n")
    return logging.getLogger(str_logger)


def _get_logger_path(pfile: str | Path, pkg_name: str = NAME_LIB_GIT) -> str:
    """Convert path string to logger name string.

    For example, if pfile is "/home/user/acm/utils/logging.py",
    and NAME_LIB_GIT is "acm", then the logger name will be "acm.utils.logging"

    If pfile isn't in the package, return just name script to avoid long string
    In all cases, the full name is written in the logger.

    Parameters
    ----------
    pfile: path of the file,

    return
    ------
    string like NAME_PKG_GIT.xx.yy.zz of script that call this function or just name script
    """
    pfile = Path(pfile)  # Ensure pfile is a Path object
    parts = pfile.parts
    if pkg_name in parts:
        idx = parts.index(pkg_name)
        module_parts = parts[idx:]  # keep only the parts after the package name
        logger_name = ".".join(module_parts)
    else:
        logger_name = parts[-1]  # just the script name
    return logger_name
