import argparse
import importlib
from ast import literal_eval


def add_observables_to_parser(parser: argparse.ArgumentParser):
    """
    Adds observable parser arguments to a given argparse parser.
    To call before parsing the main arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to which the observable arguments will be added.

    Returns
    -------
    argparse.ArgumentParser
        The parser with added observables argument and epilog.
    """
    parser.add_argument(
        "--observables",
        nargs=argparse.REMAINDER,
        help="Additional arguments for observable parser to provide at the end (see below)",
    )

    parser.epilog = """
Observable parser arguments (can be repeated for multiple observables):
    -n, --stat_name,                Observable stat_name value (required)
    -p, --paths                     Paths to the data files for the observable as key-value pairs, e.g. -p path1 /path/to/file1 path2 /path/to/file2
    -sf, --select_filters           Selection filters as key-value pairs, e.g. -sf cosmo_idx 0 -sf hod_idx 96
    -lf, --slice_filters            Slice filters as key-value pairs, e.g. -lf s "[10,20]"
    -si, --select_indices           List of indices to select from the data arrays
    -so, --select_indices_on        List of data variables to apply the indices selection on.
    -d, --flat_output_dims          Number of leading dimensions to flatten in the output (default: 0)
    -np, --numpy_output             If set, output will be converted to numpy arrays
    -s, --squeeze_output            If set, output will be squeezed to remove single-dimensional entries
    See `acm.observables.Observable` for more details.
    """
    return parser


def parse_observable(argv: list):
    """
    Parses a single observable's arguments from a list of command line arguments.

    Parameters
    ----------
    argv : list
        List of command line arguments for a single observable.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for the observable.
    """
    parser = argparse.ArgumentParser(description="Observable parser", add_help=False)
    parser.add_argument("-n", "--stat_name", type=str, required=True)
    parser.add_argument(
        "-p",
        "--paths",
        nargs="*",
        help="Paths to the data files for the observable as key-value pairs, e.g. -p path1 /path/to/file1 path2 /path/to/file2",
    )
    parser.add_argument(
        "-sf",
        "--select_filters",
        nargs="*",
        help="Selection filters as key-value pairs, e.g. -sf cosmo_idx 0 hod_idx 96",
    )
    parser.add_argument(
        "-lf",
        "--slice_filters",
        nargs="*",
        help='Slice filters as key-value pairs, e.g. -lf s "[10,20]"',
    )
    parser.add_argument(
        "-si",
        "--select_indices",
        type=int,
        nargs="+",
        help="List of indices to select from the data arrays",
    )
    parser.add_argument(
        "-so",
        "--select_indices_on",
        type=str,
        nargs="+",
        help="List of data variables to apply the indices selection on.",
    )
    parser.add_argument(
        "-d",
        "--flat_output_dims",
        type=int,
        help="Number of leading dimensions to flatten in the output",
    )
    parser.add_argument(
        "-np",
        "--numpy_output",
        action="store_true",
        help="If set, output will be converted to numpy arrays",
    )
    parser.add_argument(
        "-s",
        "--squeeze_output",
        action="store_true",
        help="If set, output will be squeezed to remove single-dimensional entries",
    )
    args = parser.parse_args(argv)
    # Edge case: if select_filters or slice_filters are provided as one element, try to eval them as dicts (otherwise, they should be pairs of key, value)
    if args.paths and len(args.paths) == 1:
        args.paths = literal_eval(args.paths[0])
    if args.select_filters and len(args.select_filters) == 1:
        args.select_filters = literal_eval(args.select_filters[0])
    if args.slice_filters and len(args.slice_filters) == 1:
        args.slice_filters = literal_eval(args.slice_filters[0])
    return args


def split_observables(argv: list):
    """
    Splits argv in chunks starting with --stat_name, or -n and returns a list of lists.
    Formats the elements of the list to be read by `parse_observables_arg`.

    Parameters
    ----------
    argv : list or dict
        List of command line arguments.
    """
    chunks = []
    current_chunk = []
    for arg in argv:
        if isinstance(arg, dict):
            for k, v in arg.items():
                current_chunk.append(f"--{k}")
                current_chunk.append(
                    str(v)
                )  # Has to be a string to be parsed by argparse
            chunks.append(current_chunk)
            current_chunk = []
            continue
        if arg.startswith("--stat_name") or arg.startswith("-n"):
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
        current_chunk.append(arg)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def parse_observables_arg(args: argparse.Namespace):
    """
    Parses the observables argument into a list of dictionaries. To call after parsing the main arguments.

    Parameters
    ----------
    arg : argparse.Namespace
        Parsed arguments from the main parser, containing the 'observables' attribute (from add_observables_to_parser).

    Returns
    -------
    argparse.Namespace
        The input args with the 'observables' attribute parsed into a list of argparse Namespaces.
    """
    if not args.observables:
        return args

    # Parse each observable with its specific arguments
    observable_args = []
    chunks = split_observables(args.observables)
    for chunk in chunks:
        obs_args = parse_observable(chunk)
        observable_args.append(obs_args)
    args.observables = observable_args
    return args


def to_pairs(lst: list | dict):
    """
    Converts a flat list or a dict into a list of key-value pairs.

    Parameters
    ----------
    lst : list
        Flat list of alternating keys and values, e.g. ['key1', 'value1', 'key2', 'value2'] or a dict.

    Returns
    -------
    list of tuples
        List of (key, value) pairs, e.g. [('key1', 'value1'), ('key2', 'value2')].
    """
    if isinstance(lst, dict):
        return list(lst.items())
    if len(lst) % 2 != 0:
        raise ValueError(
            "List must contain an even number of elements to form key-value pairs."
        )
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]


def args_to_observables(
    args: list[argparse.Namespace],
    module: str | None = None,
    mapping: dict | None = None,
    pop_args: list | None = None,
    **kwargs,
) -> list:
    """
    Converts a list of argparse Namespaces into a list of observable instances.

    Parameters
    ----------
    args : list of argparse.Namespace
        List of parsed observable arguments.
    module : str
        The name of the module containing the observable classes.
    mapping : dict
        A dictionary mapping observable names in args to their corresponding classes in `module`.
        If the observable name is not in the mapping, it is assumed to be the class name.
        Only used if `module` is provided.
    pop_args : list
        List of argument names to pop from each args Namespace before instantiation.
    **kwargs : dict
        Additional keyword arguments to pass to each observable class upon instantiation.

    Returns
    -------
    list
        An list of instantiated observable classes that can be passed to `acm.observables.CombinedObservable`.
    """
    if module:
        _module = importlib.import_module(module)
    else:
        _module = importlib.import_module("acm.observables")
    mapping = mapping if mapping else {}
    pop_args = pop_args if pop_args else []

    observable_list = []
    for obs_args in args:
        observable_args = {
            "stat_name": obs_args.stat_name,
            "paths": {
                key: literal_eval(value) for key, value in to_pairs(obs_args.paths)
            }
            if obs_args.paths
            else None,
            "select_filters": {
                key: literal_eval(value)
                for key, value in to_pairs(obs_args.select_filters)
            }
            if obs_args.select_filters
            else None,
            "slice_filters": {
                key: literal_eval(value)
                for key, value in to_pairs(obs_args.slice_filters)
            }
            if obs_args.slice_filters
            else None,
            "select_indices": obs_args.select_indices,
            "select_indices_on": obs_args.select_indices_on,
            "flat_output_dims": obs_args.flat_output_dims,
            "numpy_output": obs_args.numpy_output,
            "squeeze_output": obs_args.squeeze_output,
        }
        observable_args = {
            k: v for k, v in observable_args.items() if v is not None
        }  # Remove default values
        for arg in pop_args:
            observable_args.pop(arg, None)

        if _module.__name__ != "acm.observables":
            class_name = mapping.get(obs_args.stat_name, obs_args.stat_name)
            observable_class = getattr(_module, class_name)
        else:
            observable_class = getattr(_module, "Observable")

        observable_instance = observable_class(**observable_args, **kwargs)
        observable_list.append(observable_instance)

    return observable_list
