"""
taken from https://github.com/cosmodesi/desilike/blob/main/desilike/utils.py

This file contains logging related functions, such as setup_logging and get_logger_for_script.

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

logger = get_logger_for_script(__file__)
...

def example_function():
    logger.info("This is an info message from example_function.")

...

if __name__ == '__main__':
    setup_logging()

    logger.info("Start of script")
    <your processing code>
    logger.info("End of script")
'''

"""

import logging
import os
import os.path as osp
import sys
import time
import traceback
from contextlib import contextmanager

NAME_PKG_GIT = "acm"


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
                + " %(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
            )
            super(MyFormatter, self).format(record)
            msg = logging.Formatter.format(self, record)
            if record.message != "":
                parts = msg.split(record.message)
                msg = msg.replace("\n", "\n" + parts[0])
            return msg

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
        log.critical("An error occurred.")


def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


@contextmanager
def suppress_logging(enabled=True, highest_level=logging.CRITICAL):
    """Context manager to temporarily suppress logging messages."""
    root = logging.getLogger()
    origin_level = root.getEffectiveLevel()
    if enabled:
        root.setLevel(highest_level)  # Keep only messages at or above the highest_level
    yield
    if enabled:
        root.setLevel(origin_level)


def get_logger_for_script(pfile):
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
    # print(pfile)
    str_logger = _get_logger_path(pfile)
    # print("logger name: %s" % str_logger)
    return logging.getLogger(str_logger)


def _get_logger_path(pfile):
    """Convert path string to logger name string

    For example, if pfile is "/home/user/acm/utils/logging.py",
    and NAME_PKG_GIT is "acm", then the logger name will be "acm.utils.logging"

    Parameters
    ----------
    pfile: path of the file,

    return
    ------
    string like NAME_PKG_GIT.xx.yy.zz of script that call this function
    """
    l_sep = osp.sep
    r_str = l_sep + NAME_PKG_GIT + l_sep
    p_grand = pfile.find(r_str)
    if p_grand > 0:
        # -3 for size of ".py"
        g_str = pfile[p_grand + 1 : -3].replace(l_sep, ".")
    else:
        # out package git
        # -3 for size of ".py"
        print("out package git")
        if pfile[0] == l_sep:
            g_str = pfile[1:-3].replace(l_sep, ".")
        else:
            g_str = pfile[0:-3].replace(l_sep, ".")
    return g_str
