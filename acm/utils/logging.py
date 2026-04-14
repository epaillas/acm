# taken from https://github.com/cosmodesi/desilike/blob/main/desilike/utils.py
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager


def setup_logging(
    level=logging.INFO, stream=sys.stdout, filename=None, filemode="w", **kwargs
):
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
        def format(self, record):
            self._style._fmt = (
                "[%09.2f] " % (time.time() - t0)
                + " %(asctime)s %(name)s %(levelname)s:(lineno)d %(message)s"
            )
            return super(MyFormatter, self).format(record)

    fmt = MyFormatter(datefmt="%m-%d %H:%M ")
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)
    sys.excepthook = exception_handler


def exception_handler(exc_type, exc_value, exc_traceback):
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
        log.critical("An error occured.")


def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


@contextmanager
def supress_logging(enabled=True, highest_level=logging.CRITICAL):
    """Context manager to temporarily suppress logging messages."""
    root = logging.getLogger()
    origin_level = root.getEffectiveLevel()
    if enabled:
        root.setLevel(highest_level)  # Keep only messages at or above the highest_level
    yield
    if enabled:
        root.setLevel(origin_level)
