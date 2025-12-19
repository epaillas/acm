from pathlib import Path
from sunbird.inference.samples import Chain
from acm.observables import CombinedObservable
from acm.utils.modules import get_class_from_module

# TODO: move this to sunbird.inference.priors
def get_fixed_params(cosmo_model: str, hod_model: str, priors: dict) -> list:
    """
    Return a list of fixed parameter names based on the cosmological and HOD models.
    This function checks which parameters are free in the specified models.
    
    Parameters
    ----------
    cosmo_model : str
        The cosmological model string containing keywords (e.g., 'base', 'w0', 'wa', etc.).
    hod_model : str
        The HOD model string containing keywords (e.g., 'base', 'AB', 'CB', etc.).
    priors : dict
        A dictionary of all parameter priors.
    """
    free = []
    # cosmology
    if 'base' in cosmo_model:
        free += ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
    if 'w0' in cosmo_model:
        free += ['w0_fld']
    if 'wa' in cosmo_model:
        free += ['wa_fld']
    if 'Nur' in cosmo_model:
        free += ['N_ur']
    if 'nrun' in cosmo_model:
        free += ['nrun']
    if 'fixed-ns' in cosmo_model:
        free.remove('n_s')
    # HOD
    if 'base' in hod_model:
        free += ['logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa']
    if 'AB' in hod_model:
        free += ['B_cen', 'B_sat']
    if "CB" in hod_model:
        free += ["A_cen", "A_sat"]
    if 'VB' in hod_model:
        free += ['alpha_c', 'alpha_s']
    if '_s' in hod_model or '-s' in hod_model:
        free += ['s']
    fixed = [par for par in priors.keys() if par not in free]
    return fixed


def get_observable(
    observable_names: list[str]|str, 
    module: str = 'acm.observables.bgs', 
    select_filters_map: dict[dict]|None = None,
    slice_filters_map: dict[dict]|None = None,
    **kwargs,
) -> CombinedObservable:
    """
    Get the observable class by name.
    
    Parameters
    ----------
    observable_names : list[str] | str
        Name(s) of the observable class(es) to retrieve.
    module : str
        The base module path where the observable classes are located.
    select_filters_map : dict[dict] | None
        A mapping from observable names to their select filters.
    slice_filters_map : dict[dict] | None
        A mapping from observable names to their slice filters.
    **kwargs
        Additional keyword arguments to pass to the observable class constructors.
        If 'select_filters' or 'slice_filters' are provided in kwargs, they will be
        updated with the corresponding filters from the maps.
    """
    if isinstance(observable_names, str):
        observable_names = [observable_names]
        
    # Prevent crash in get call later
    if select_filters_map is None:
        select_filters_map = {}
    if slice_filters_map is None:
        slice_filters_map = {}
        
    observables = []
    for observable_name in observable_names:
        cls = get_class_from_module(module, observable_name)
        
        select_filters = kwargs.pop('select_filters', {})
        slice_filters = kwargs.pop('slice_filters', {})

        select_filters.update(select_filters_map.get(observable_name, {}))
        slice_filters.update(slice_filters_map.get(observable_name, {}))
        kwargs['select_filters'] = select_filters
        kwargs['slice_filters'] = slice_filters
        
        obs = cls(**kwargs)
        observables.append(obs)
    
    return CombinedObservable(observables)

def save_and_plot(
    sampler, 
    observable, 
    identifier: str = None,
    save_dir: str = None
) -> None:
    """
    Save sampler results and generate plots.

    Parameters
    ----------
    sampler : PocoMCSampler
        The sampler object containing the sampling results.
    observable : CombinedObservable
        The observable object containing data and methods to get covariance and model predictions.
    identifier : str, optional
        An optional identifier for the run. Defaults to None.
    save_dir : str, optional
        The directory to save the results. If None, no saving is performed. Defaults to None.
    """
    if save_dir is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    handle = observable.get_save_handle(save_dir=str(save_dir))
    if identifier is not None:
        handle += f'_{identifier}'

    sampler.save_chain(save_fn = f'{handle}.npy', metadata={'markers': sampler.markers, 'zeff': 0.5})
    sampler.save_table(save_fn=f'{handle}_stats.txt')
    
    chain = Chain.load(f'{handle}.npy')
    chain.plot_triangle(save_fn=f'{handle}_triangle.png', thin=128, markers=sampler.markers, title_limit=1)
    chain.plot_trace(save_fn=f'{handle}_trace.png', thin=128)
    observable.plot_observable(model_params=chain.bestfit, save_fn=f'{handle}_bestfit.pdf')