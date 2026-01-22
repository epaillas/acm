import yaml
from pathlib import Path
from acm.utils.decorators import require_nersc

@require_nersc(enabled=True)
def resolve_yaml_path(filename: str, *keys: str):
    """
    Reads a .yaml file and parses the arguments as nested keys to return the corresponding value.

    Parameters
    ----------
    filename : str
        The path to the .yaml file to read. Relative paths will be resolved as relative to the script's location, 
        with a fallback to the acm.utils.paths directory.
    *keys : str
        Nested keys to traverse the YAML structure.
    
    Returns
    -------
    Any
        The resolved value corresponding to the nested keys provided in *keys.
    
    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.
    ValueError
        If the provided keys do not lead to a valid value in the YAML structure.
    """
    here = Path(__file__).parent
    fn = Path(filename) # Assert absolute or relative to cwd
    
    if not fn.exists():
        fn = here / filename # Relative to this script
        
    if not fn.exists():
        raise FileNotFoundError(f"YAML file not found: {filename}")

    with open(fn, 'r') as file:
        data = yaml.safe_load(file)
        
        for key in keys:
            if not isinstance(data, dict) or key not in data:
                raise KeyError(f"Invalid YAML key path: {' -> '.join(keys)}")
            data = data.get(key)

    return data