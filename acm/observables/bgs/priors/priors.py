from sunbird.inference.priors.priors import *

#%% Priors 
class Bouchard25(Yuan23): # Following naming convention : name + year TODO: Change to a name related to the project ?
    def __init__(self, stats_module='scipy.stats'):
        dirname = os.path.dirname(__file__)
        self.priors = load_prior(os.path.join(dirname, 'bouchard25.yaml'),
                          stats_module=stats_module)
        self.ranges = load_ranges(os.path.join(dirname, 'bouchard25.yaml'))
        self.labels = load_labels(os.path.join(dirname, 'bouchard25.yaml'))
    

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
        priors.update(Bouchard25(stats_module).priors)
        ranges.update(Bouchard25(stats_module).ranges)
        labels.update(Bouchard25(stats_module).labels)
    return priors, ranges, labels


def truth_within_priors(truth: dict, 
                        priors_ranges: dict = None,
                        verbose: bool = False) -> bool:
    """
    Check if the truth is within the priors.

    Parameters
    ----------
    truth : dict
        Dictionary containing the true values of the parameters.
    priors_ranges : dict, optional
        Dictionary containing the priors for the parameters. If None, the AbacusSummit and Yuan23 priors will be used. Defaults to None.
    verbose : bool, optional
        Whether to print the parameters that are not within the priors. Defaults to False.

    Returns
    -------
    bool
        Whether the truth is within the priors.
    """
    
    if priors_ranges is None:
        _, priors_ranges, _ = get_priors()
    
    priors_ranges = {key: priors_ranges[key] for key in truth.keys()} # Order the priors in the same order as the LHC and keep only the relevant ones
    
    is_within = True
    for key, value in truth.items():
        if value < priors_ranges[key][0] or value > priors_ranges[key][1]:
            is_within = False
            if verbose:
                print(f'Parameter {key} ({value:.2f}) is not within the priors: {priors_ranges[key]}')
            
    return is_within