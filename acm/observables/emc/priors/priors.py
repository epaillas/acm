from sunbird.inference.priors.priors import *

def get_priors(cosmo: bool = True, 
               hod: bool = True, 
               ):
    """
    Get the priors, ranges and labels for the parameters.

    Parameters
    ----------
    cosmo : bool, optional
        Whether to include the cosmological parameters. Defaults to True.
    hod : bool, optional
        Whether to include the HOD parameters. Defaults to True.

    Returns
    -------
    priors : dict
        Dictionary containing the priors for the parameters.
    ranges : dict
        Dictionary containing the prior ranges for the parameters.
    labels : dict
        Dictionary containing the labels for the parameters.
    """
    stats_module = 'scipy.stats'
    priors, ranges, labels = {}, {}, {}
    if cosmo:
        priors.update(AbacusSummit(stats_module).priors)
        ranges.update(AbacusSummit(stats_module).ranges)
        labels.update(AbacusSummit(stats_module).labels)
    if hod:
        priors.update(Yuan23(stats_module).priors)
        ranges.update(Yuan23(stats_module).ranges)
        labels.update(Yuan23(stats_module).labels)
    return priors, ranges, labels