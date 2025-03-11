# Utilities

## Logging

A logger can be setup using the `acm.utils.setup_logging` function, following the convention of the `cosmodesi` packages. By default, the logger is setup to log to the console, with the level set to `INFO` by default. The logger can be configured to log to a file by specifying the `filename` argument.

```python
from acm.utils import setup_logging
from logging import getLogger

# Setup the logger
setup_logging() 

# Get the logger
logger = getLogger(__name__)

# Log a message
logger.info("This is an info message")
```


## API

```{eval-rst}
.. automodule:: acm.utils
    :members:
    :undoc-members:
    :show-inheritance:
```