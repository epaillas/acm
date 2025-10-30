import pandas as pd

def load_abacus_cosmologies(
    filename: str, 
    cosmologies: list[int], 
    parameters: list[str],
    mapping: dict[str, str],
    ) -> None:
    """
    Loads the AbacusSummit cosmology parameters from the AbacusSummit cosmologies csv file and selects
    the `cosmologies` indexes. Also selects the parameters to keep. Renames the parameters according to mapping.

    Parameters
    ----------
    filename : str
        Filename (csv) with the AbacusSummit cosmology parameters.
    cosmologies : list[int]
        List of cosmologies indexes to select.
    parameters : list[str]
        List of parameters to keep.
    mapping : dict[str, str]
        Dictionary with the mapping from the original parameter names to the desired names.
    """
    cosmo_params = pd.read_csv(
        filename,
        usecols = ['root'] + parameters,
    )
    cosmo_params = cosmo_params[cosmo_params['root'].isin([f'abacus_cosm{c:03d}' for c in cosmologies])]
    cosmo_params.drop(columns=['root'], inplace=True)
    cosmo_params.set_index(pd.Index([f'c{c:03d}' for c in cosmologies]), inplace=True)
    cosmo_params.rename(columns=mapping, inplace=True)
    return cosmo_params.to_dict(orient='index')