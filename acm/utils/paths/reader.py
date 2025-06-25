import yaml
from pathlib import Path

def get_Abacus_dirs(tracer:str, simtype:str = None) -> dict:
    """
    Get the Abacus Dark Matter Catalogs paths for a given tracer from the Abacus.yaml file.

    Parameters
    ----------
    tracer : str
        The tracer for which to get the paths (e.g., 'BGS', 'LRG').
    simtype : str, optional
        The type of simulation (e.g., 'box', 'lightcone'). 
        If provided, it will only return paths for that simulation type.

    Returns
    -------
    dict
        A dictionary containing the paths for the specified tracer.

    Raises
    ------
    ValueError
        If the tracer is not found in the Abacus.yaml file or if no paths are available for the specified simulation type.
    """
    here = Path(__file__).parent

    with open(here / 'Abacus.yaml', 'r') as file:
        abacus_dirs = yaml.safe_load(file)
        tracer_dict = abacus_dirs.get(tracer, {})
        if not tracer_dict:
            raise ValueError(f"No paths found for tracer '{tracer}' in Abacus.yaml")
        if simtype:
            tracer_dict = tracer_dict.get(simtype, {})
            if not tracer_dict:
                raise ValueError(f"No paths found for tracer '{tracer}' with simulation type '{simtype}' in Abacus.yaml")
    return tracer_dict
    
def get_data_dirs(project: str) -> dict:
    """
    Get the data directories for a given project from the data.yaml file.

    Parameters
    ----------
    project : str
        The project name for which to get the paths (e.g., 'emc', 'bgs').

    Returns
    -------
    dict
        A dictionary containing the paths for the specified project.

    Raises
    ------
    ValueError
        If the project is not found in the data.yaml file or if no paths are available for the specified project.
    """
    here = Path(__file__).parent
    
    with open(here / 'data.yaml', 'r') as file:
        data_dirs = yaml.safe_load(file)
        project_dirs = data_dirs.get(project, {})
        if not project_dirs:
            raise ValueError(f"No paths found for project '{project}' in data_dirs.yaml")
    return project_dirs