from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from acm.utils.decorators import require_nersc


@require_nersc(enabled=True)
def lookup_registry_path(
    filename: str,
    *keys: str,
    loader: Callable = yaml.safe_load,
) -> Any: # noqa: ANN401
    """
    Read a file and parses the arguments as nested keys to return the corresponding value.

    Parameters
    ----------
    filename : str
        The path to the file to read. Relative paths will be resolved as relative to the current working
        directory; if the file is not found there, they are resolved relative to the directory of the
        ``acm.utils.paths`` package.
    *keys : str
        Nested keys to traverse the file structure.

    Returns
    -------
    Any
        The resolved value corresponding to the nested keys provided in *keys.

    Raises
    ------
    KeyError
        If the provided keys do not lead to a valid value in the structure.
    """
    here = Path(__file__).parent
    fn = Path(filename)  # Assert absolute or relative to cwd

    if not fn.exists():
        fn = here / filename  # Relative to this script's directory

    with fn.open("r") as file:
        data = loader(file)

        for key in keys:
            if not isinstance(data, dict) or key not in data:
                raise KeyError(f"Invalid key path: {' -> '.join(keys)}")
            data = data.get(key)

    return data


def list_registry_files(
    ext: tuple[str, ...] = (".yaml", ".yml"),
    recursive: bool = False,
) -> list[str]:
    """
    List all available registry files shipped with this package.

    Parameters
    ----------
    ext : tuple[str, ...]
        A tuple of file extensions to look for. Default is (".yaml", ".yml").
    recursive : bool
        Whether to search for files recursively in subdirectories. Default is False.

    Returns
    -------
    list[str]
        A list of filenames available in the specified directory.
    """
    base_dir = Path(__file__).parent
    patterns = [f"**/*{e}" if recursive else f"*{e}" for e in ext]
    files = []

    for pattern in patterns:
        files.extend([f.name for f in base_dir.glob(pattern)])

    return files
