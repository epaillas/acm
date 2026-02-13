import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from sunbird.inference.samples import Chain
from acm.observables import CombinedObservable
from acm.utils.modules import get_class_from_module

from secondgen_bgs import get_secondgen_data

#%% Inference script utils
def get_observable(
    observable_names: list[str]|str, 
    module: str = 'acm.observables.bgs', 
    select_filters_map: dict[dict]|None = None,
    slice_filters_map: dict[dict]|None = None,
    kwargs_map: dict[dict]|None = None,
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
    kwargs_map : dict[dict] | None
        A mapping from observable names to their specific keyword arguments.
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
    if kwargs_map is None:
        kwargs_map = {}
        
    observables = []
    _select_filters = kwargs.pop('select_filters', {})
    _slice_filters = kwargs.pop('slice_filters', {})
    for observable_name in observable_names:
        cls = get_class_from_module(module, observable_name)
        
        # Prevent modifying the original kwargs
        select_filters = _select_filters.copy()
        slice_filters = _slice_filters.copy()
        select_filters.update(select_filters_map.get(observable_name, {}))
        slice_filters.update(slice_filters_map.get(observable_name, {}))
        kwargs['select_filters'] = select_filters
        kwargs['slice_filters'] = slice_filters
        
        kwargs.update(kwargs_map.get(observable_name, {})) # Update with specific kwargs for this observable if provided
        
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
    
#%% Control plots utils
def get_chain(
    type_fit: str, 
    cosmo_model: str, 
    hod_model: str, 
    stat_name: str, 
    chain_dir: str,
    identifier: str = None,
) -> Chain:
    """Load a chain from disk trough the Chain class."""
    fn = stat_name
    if identifier is not None:
        fn += f'_{identifier}'
    chain_dir = Path(chain_dir)
    fn = chain_dir / type_fit / f'cosmo-{cosmo_model}_hod-{hod_model}' / f'{fn}.npy'
    if not fn.exists():
        return None
    chain = Chain.load(fn)
    chain.data['label'] = stat_name
    return chain

def get_chains(
    type_fit: str, 
    cosmo_model: str, 
    hod_model: str, 
    chain_dir: str,
    stats: list[str] = ['tpcf', 'ds_xiqg+ds_xiqq', 'tpcf+ds_xiqg+ds_xiqq'],
    identifier: str = None,
) -> list[Chain]:
    """Get several chains corresponding to different statistics."""
    chains = []
    for stat in stats:
        chain = get_chain(type_fit, cosmo_model, hod_model, stat, identifier=identifier, chain_dir=chain_dir)
        if chain is not None:
            chains.append(chain)
    return chains

def print_std_improvements(ref: Chain, comp: Chain, params: list[str]) -> dict:
    """
    Print the standard deviation improvements from a reference chain to a comparison chain.
    
    Parameters
    ----------
    ref: Chain
        The reference chain.
    comp: Chain
        The comparison chain.
    params: list[str]
        The list of parameter names to consider.
    
    Returns
    -------
    improvements: dict
        A dictionary with parameter names as keys and a list of tuples (mean, std, improvement) as values.
    """
    
    improvements = {}
    
    chains = [ref, comp]
    chains_mean = [chain.samples.mean(axis=0) for chain in chains]
    chains_std = [chain.samples.std(axis=0) for chain in chains]
    names = chains[0].names
    
    for name in params:
        if name not in names:
            continue
        
        improvements.setdefault(name, [])
        for i, chain in enumerate(chains):
            mean = chains_mean[i][names.index(name)]
            std = chains_std[i][names.index(name)]
            
            if i == 0:
                ref_std = std
            else:
                improvement = (ref_std - std) / ref_std * 100
                improvements[name].append((mean, std, improvement))
                print(f'    {name}: {mean:.5f} ± {std:.5f} ({improvement:.2f}% std improvement)')
    
    return improvements

def add_model_errors_to_ax(
    observable,
    chain: Chain,
    ax: plt.Axes,
    **kwargs,
) -> None: 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_err = observable.get_emulator_error()
        mock_err = np.sqrt(np.diag(observable.get_covariance_matrix()))
        pred_err = np.sqrt(model_err**2 + mock_err**2)
        
    if observable.stat_name == 'tpcf':
        s = observable.s.values
        ells = kwargs.get('ells', [0, 2])
        for i, ell in enumerate(ells):
            y = observable.get_model_prediction(chain.bestfit).unstack().sel(ells=ell).values
            err = pred_err.unstack().sel(ells=ell).values
            ax[0].fill_between(s, (y - err)*s**2, (y + err)*s**2, alpha=0.2, color=f'C{i}', zorder=-1)
    elif observable.stat_name in ['ds_xiqq', 'ds_xiqg']:
        s = observable.s.values
        ell = kwargs.get('ell', 0)
        quantiles = kwargs.get('quantiles', [0, 1, 3, 4])
        for i, q in enumerate(quantiles):
            y = observable.get_model_prediction(chain.bestfit).sel(ells=ell, quantiles=q).values
            err = pred_err.sel(ells=ell, quantiles=q).values
            ax[0].fill_between(s, (y - err)*s**2, (y + err)*s**2, alpha=0.2, color=f'C{i}', zorder=-1)

def plot_model_vs_truth(
    chain: Chain,
    data_obs,
    model_obs = None,
    add_model_errors: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes, tuple]:
    """
    Uses the plot_observable method from the BGS observables to plot 
    the observable data vs the model prediction at the bestfit point of the chain.

    Parameters
    ----------
    chain : Chain
        The chain from which to get the bestfit parameters.
    data_obs : acm.observables.Observable
        The Observable instance to load the data from.
    model_obs: acm.observables.Observable
        The Observable instance to load the model from. If set to None, will be
        a copy of data_obs. Defaults to None.
    add_model_errors : bool, optional
        Whether to add model errors to the plot, by default True.

    Returns
    -------
    fig : plt.Figure
        The figure object containing the plot.
    ax : plt.Axes
        The axes object containing the plot.
    tuple :
        A tuple containing the observable instance and the predicted errors.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = data_obs.plot_observable(chain.bestfit, show_legend=True, **kwargs)
        
        if model_obs is None:
            model_obs = deepcopy(data_obs)
            
        if add_model_errors:
            add_model_errors_to_ax(model_obs, chain, ax, **kwargs)
            
    return fig, ax