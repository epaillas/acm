"""
A script to run visual tools for measurements of the BGS project.
Opens an interactive matplotlib window to explore measurements outputs.

Usage:

"""
from pathlib import Path

import lsstypes
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from pycorr import TwoPointEstimator
from IPython.display import display, clear_output

from outliers import nested_set

#%% HOD plots functions
def csv_to_structured_array(filename: str | Path, **kwargs) -> np.ndarray:
    """
    Load a CSV file into a structured array.

    Parameters
    ----------
    filename : str | Path
        Path to the CSV file.
    **kwargs
        Additional keyword arguments to pass to `np.genfromtxt`.

    Returns
    -------
    np.ndarray
        A structured array containing the data from the CSV file.
    """
    delimiter = kwargs.pop('delimiter', ',') # Ensure default delimiter is comma
    names = kwargs.pop('names', True) # Use header names by default
    data = np.genfromtxt(filename, delimiter=delimiter, names=names, **kwargs)
    # data = {name: data[name] for name in data.dtype.names}
    return data

def load_hod_params(hod_params_dir: str | Path, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """
    Load HOD parameters from CSV files in a specified directory into a dictionary of structured arrays.

    Parameters
    ----------
    hod_params_dir : str | Path
        Path to the directory containing HOD parameter CSV files.
    keys : list[str] | None, optional
        List of keys to use for the dictionary.
        Must be at least as long as the number of files and correspond to the sorted order of the files.
        If None, the filenames (without extension) are used as keys. Defaults to None.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary where keys are specified or derived from filenames,
        and values are structured arrays of HOD parameters.

    Raises
    ------
    ValueError
        If the length of keys is less than the number of HOD parameter files.
    """
    hod_params_dir = Path(hod_params_dir)
    hod_params_files = sorted(hod_params_dir.glob('*.csv'))
    
    if keys is not None and len(keys) < len(hod_params_files):
        raise ValueError("Length of keys must at least match number of HOD parameter files")
    
    hod_params = {}
    for i, fn in enumerate(hod_params_files):
        key = keys[i] if keys is not None else fn.stem
        hod_params[key] = csv_to_structured_array(fn)
    return hod_params

def get_hod_folders(dir: Path, cosmologies: list[int], phases: list[int], seeds: list[int], sim_type: str, hod_idx: int = None) -> dict:
    """
    Get HOD folders for given cosmologies, phases, and seeds from the directory of the measurements. 

    Parameters
    ----------
    dir : Path
        Directory of the measurements.
    cosmologies : list[int]
        List of cosmologies to get the HOD folders for.
    phases : list[int]
        List of phases to get the HOD folders for.
    seeds : list[int]
        List of seeds to get the HOD folders for.
    sim_type : str
        Abacus simulation type (e.g. 'base').
    hod_idx : int, optional
        HOD realization index to filter the folders. If None, gets all HOD folders. Defaults to None.

    Returns
    -------
    dict
        A dictionary where keys are tuples of (cosmology, phase, seed) and values are lists of HOD folder paths.
    """
    hod_fns = {}
    for cosmology in cosmologies:
        for phase in phases:
            for seed in seeds:
                f = Path(dir) / sim_type / f'c{cosmology:03d}_ph{phase:03d}' / f'seed{seed}'
                if hod_idx is not None:
                    fn_list = [f / f'hod{hod_idx:03d}']
                else:
                    fn_list = sorted(f.glob('hod*'))
                nested_set(hod_fns, (cosmology, phase, seed), fn_list)
    return hod_fns

#%% Density functions
def get_densities(dir: Path|str, cosmologies: list, phases: list, seeds: list, sim_type: str, hod_idx: int = None) -> list:
    """
    Get densities from measurement files for given cosmologies, phases, and seeds from the directory of the measurements.
    
    Parameters
    ----------
    dir : Path | str
        Directory of the measurements.
    cosmologies : list[int]
        List of cosmologies to get the densities for.
    phases : list[int]
        List of phases to get the densities for.
    seeds : list[int]
        List of seeds to get the densities for.
    sim_type : str
        Abacus simulation type (e.g. 'base').
    hod_idx : int, optional
        HOD realization index to get the densities for. If None, gets all HOD realizations. Defaults to None.

    Returns
    -------
    list
        A list of density values from the measurement files.
    """
    densities = []
    for cosmology in cosmologies:
        for phase in phases:
            for seed in seeds:
                dir_ = Path(dir) / sim_type / f'c{cosmology:03d}_ph{phase:03d}' / f'seed{seed}'
                
                if hod_idx is not None:
                    hod_folders = [dir_ / f'hod{hod_idx:03d}']
                else:
                    hod_folders = sorted(dir_.glob('hod*')) # Get all the HOD realizations
                
                for hod in hod_folders:
                    fn_list = sorted(hod.glob('density.npy'))
                    for fn in fn_list:
                        d = np.load(fn, allow_pickle=True).item()
                        densities.append(d)
    return densities

#%% TPCF functions
def get_tpcf(dir: Path|str, cosmologies: list, phases: list, seeds: list, sim_type: str, hod_idx: int = None) -> list:
    """
    Get two-point correlation functions from measurement files for given cosmologies, phases, and seeds from the directory of the measurements.
    
    Parameters
    ----------
    dir : Path | str
        Directory of the measurements.
    cosmologies : list[int]
        List of cosmologies to get the two-point correlation functions for.
    phases : list[int]
        List of phases to get the two-point correlation functions for.
    seeds : list[int]
        List of seeds to get the two-point correlation functions for.
    sim_type : str
        Abacus simulation type (e.g. 'base').
    hod_idx : int, optional
        HOD realization index to get the two-point correlation functions for. If None, gets all

    Returns
    -------
    list
        A list of two-point correlation function objects from the measurement files.
    """
    tpcfs = []
    for cosmology in cosmologies:
        for phase in phases:
            for seed in seeds:
                dir_ = Path(dir) / sim_type / f'c{cosmology:03d}_ph{phase:03d}' / f'seed{seed}'
                
                if hod_idx is not None:
                    hod_folders = [dir_ / f'hod{hod_idx:03d}']
                else:
                    hod_folders = sorted(dir_.glob('hod*')) # Get all the HOD realizations

                for hod in hod_folders:
                    fn_list = sorted(hod.glob('tpcf_los_*.npy'))
                    cf_list = [TwoPointEstimator.load(fn).normalize() for fn in fn_list]
                    cf = sum(cf_list)
                    if len(cf_list) > 0:
                        tpcfs.append(cf)
    return tpcfs

def plot_tpcf(ax: plt.Axes, measurements: list, ells: list[int] = [0, 2, 4], add_handles: bool = True, **kwargs) -> None:
    """
    Plot two-point correlation functions on a given axis.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    measurements : list
        List of two-point correlation function objects to plot.
    ells : list[int], optional
        List of multipole moments to plot, by default [0, 2, 4].
    add_handles : bool, optional
        Whether to add legend handles for the multipole moments, by default True.
    **kwargs
        Additional keyword arguments to pass to the plot function.
        
    Returns
    -------
    plt.Axes
        The matplotlib axis with the plotted two-point correlation functions.
    """
    kwargs.setdefault('alpha', 0.1)
    _c = kwargs.pop('c', None) # Ensure default color cycling if not specified
    
    for cf in measurements:
        s, poles = cf(ells=ells, return_sep=True)
        for i, ell in enumerate(ells):
            c =  _c if _c is not None else f'C{i}'
            ax.plot(s, poles[i]*s**2, c=c, **kwargs)
    
    handles = []
    for i, ell in enumerate(ells):
        color = _c if _c is not None else f'C{i}'
        kwargs['alpha'] = 1.0 # Make legend handles fully opaque
        handles.append(plt.Line2D([0], [0], color=color, label=rf'$\ell={{{ell}}}$', **kwargs)) 
    if add_handles:
        ax.legend(handles=handles)
    
    return ax

#%% DensitySplit functions
def get_ds_cf(stat_name: str, dir: Path|str, cosmologies: list, phases: list, seeds: list, sim_type: str, quantiles: list = [0,1,2,3,4], hod_idx: int = None) -> list:
    """
    Get density split correlation functions from measurement files for given cosmologies, phases, and seeds from the directory of the measurements.
    
    Parameters
    ----------
    stat_name : str
        Name of the statistic to load (e.g. 'ds_cf').
    dir : Path | str
        Directory of the measurements.
    cosmologies : list[int]
        List of cosmologies to get the density split correlation functions for.
    phases : list[int]
        List of phases to get the density split correlation functions for.
    seeds : list[int]
        List of seeds to get the density split correlation functions for.
    sim_type : str
        Abacus simulation type (e.g. 'base').
    quantiles : list, optional
        List of quantiles to load, by default [0,1,2,3,4].
    hod_idx : int, optional
        HOD realization to load. If None, loads all HOD realizations. Defaults to None.
    
    Returns
    -------
    list
        A list of density split correlation function objects from the measurement files.
    """
    ds_cfs = []
    for cosmology in cosmologies:
        for phase in phases:
            for seed in seeds:
                dir_ = Path(dir) / sim_type / f'c{cosmology:03d}_ph{phase:03d}' / f'seed{seed}'
                
                if hod_idx is not None:
                    hod_folders = [dir_ / f'hod{hod_idx:03d}']
                else:
                    hod_folders = sorted(dir_.glob('hod*')) # Get all the HOD realizations

                for hod in hod_folders:
                    fn_list = sorted(hod.glob(f'{stat_name}_los_*.npy'))
                    cf_list = []
                    for q in quantiles:
                        cfs = [np.load(fn, allow_pickle=True)[q].normalize() for fn in fn_list]
                        cf = sum(cfs)
                        if len(cfs) > 0:
                            cf_list.append(cf)
                    if len(cf_list) > 0:
                        ds_cfs.append(cf_list)
    return ds_cfs

def plot_ds_cf(ax: plt.Axes, measurements: list, quantiles: list = [0, 1, 2, 3, 4] , ell: int = 0, add_handles: bool = True, **kwargs) -> None:
    """
    Plot density split correlation functions on a given axis.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    measurements : list
        List of density split correlation function objects to plot.
    quantiles : list, optional
        List of quantiles to plot. Defaults to [0, 1, 2, 3, 4] .
    ell : int, optional
        Multipole moment to plot. Defaults to 0.
    add_handles : bool, optional
        Whether to add legend handles for the quantiles. Defaults to True.
    **kwargs
        Additional keyword arguments to pass to the plot function.
    
    Returns
    -------
    plt.Axes
        The matplotlib axis with the plotted density split correlation functions.
    """
    kwargs.setdefault('alpha', 0.1)
    _c = kwargs.pop('c', None) # Ensure default color cycling if not specified
    
    for cf in measurements:
        for i, q in enumerate(quantiles):
            c =  _c if _c is not None else f'C{i}'
            s, poles = cf[q](ells=[ell], return_sep=True)
            ax.plot(s, poles[0]*s**2, c=c, **kwargs)
    
    handles = []
    for i, q in enumerate(quantiles):
        color = _c if _c is not None else f'C{i}'
        kwargs['alpha'] = 1.0 # Make legend handles fully opaque
        handles.append(plt.Line2D([0], [0], color=color, label=f'DS={q}', **kwargs)) 
    if add_handles:
        ax.legend(handles=handles, title='Quantiles', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax

#%% Power spectrum functions
def get_power_spectra(dir: Path|str, cosmologies: list, phases: list, seeds: list, sim_type: str, hod_idx: int = None) -> list:
    """
    Get power spectra from measurement files for given cosmologies, phases, and seeds from the directory of the measurements.
    
    Parameters
    ----------
    dir : Path | str
        Directory of the measurements.
    cosmologies : list[int]
        List of cosmologies to get the power spectra for.
    phases : list[int]
        List of phases to get the power spectra for.
    seeds : list[int]
        List of seeds to get the power spectra for.
    sim_type : str
        Abacus simulation type (e.g. 'base').
    hod_idx : int, optional
        HOD realization index to get the power spectra for. If None, gets all HOD realizations. Defaults to None.

    Returns
    -------
    list
        A list of power spectrum objects from the measurement files.
    """
    power_spectra = []
    for cosmology in cosmologies:
        for phase in phases:
            for seed in seeds:
                dir_ = Path(dir) / sim_type / f'c{cosmology:03d}_ph{phase:03d}' / f'seed{seed}'
                
                if hod_idx is not None:
                    hod_folders = [dir_ / f'hod{hod_idx:03d}']
                else:
                    hod_folders = sorted(dir_.glob('hod*')) # Get all the HOD realizations

                for hod in hod_folders:
                    fn_list = sorted(hod.glob('power_spectrum_*.h5'))
                    ps_list = [lsstypes.read(fn) for fn in fn_list]
                    ps = sum(ps_list)
                    if len(ps_list) > 0:
                        power_spectra.append(ps)
    return power_spectra

def plot_power_spectra(ax: plt.Axes, measurements: list, ells: list = [0, 2], add_handles: bool = True, **kwargs) -> None:
    """
    Plot power spectra on a given axis.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    measurements : list
        List of power spectrum objects to plot.
    add_handles : bool, optional
        Whether to add legend handles for the power spectra. Defaults to True.
    **kwargs
        Additional keyword arguments to pass to the plot function.
    
    Returns
    -------
    plt.Axes
        The matplotlib axis with the plotted power spectra.
    """
    kwargs.setdefault('alpha', 0.1)
    _c = kwargs.pop('c', None) # Ensure default color cycling if not specified
    
    for ps in measurements:
        poles = [ps.get(ell) for ell in ells]
        for i, ell in enumerate(ells):
            k = poles[i].coords('k')
            pk = poles[i].value()
            c =  _c if _c is not None else f'C{i}'
            ax.plot(k, pk*k**2, c=c, **kwargs)
    
    handles = []
    for i, ell in enumerate(ells):
        color = _c if _c is not None else f'C{i}'
        kwargs['alpha'] = 1.0 # Make legend handles fully opaque
        handles.append(plt.Line2D([0], [0], color=color, label=rf'$\ell={{{ell}}}$', **kwargs)) 
    if add_handles:
        ax.legend(handles=handles)
    
    return ax

#%% Fourier-space density split functions
def get_ds_power_spectra(stat_name: str, dir: Path|str, cosmologies: list, phases: list, seeds: list, sim_type: str, quantiles: list = [0,1,2,3,4], hod_idx: int = None) -> list:
    """
    Get density split power spectra from measurement files for given cosmologies, phases, and seeds from the directory of the measurements.
    
    Parameters
    ----------
    stat_name : str
        Name of the statistic to load (e.g. 'quantile_power').
    dir : Path | str
        Directory of the measurements.
    cosmologies : list[int]
        List of cosmologies to get the density split power spectra for.
    phases : list[int]
        List of phases to get the density split power spectra for.
    seeds : list[int]
        List of seeds to get the density split power spectra for.
    sim_type : str
        Abacus simulation type (e.g. 'base').
    quantiles : list, optional
        List of quantiles to load, by default [0,1,2,3,4].
    hod_idx : int, optional
        HOD realization to load. If None, loads all HOD realizations. Defaults to None.
    
    Returns
    -------
    list
        A list of density split power spectrum objects from the measurement files.
    """
    ds_power_spectra = []
    for cosmology in cosmologies:
        for phase in phases:
            for seed in seeds:
                dir_ = Path(dir) / sim_type / f'c{cosmology:03d}_ph{phase:03d}' / f'seed{seed}'
                
                if hod_idx is not None:
                    hod_folders = [dir_ / f'hod{hod_idx:03d}']
                else:
                    hod_folders = sorted(dir_.glob('hod*')) # Get all the HOD realizations

                for hod in hod_folders:
                    fn_list = sorted(hod.glob(f'{stat_name}_los_*.h5'))
                    ps_list = []
                    for q in quantiles:
                        pss = [lsstypes.read(fn).get(quantiles=q) for fn in fn_list]
                        ps = sum(pss)
                        if len(pss) > 0:
                            ps_list.append(ps)
                    if len(ps_list) > 0:
                        ds_power_spectra.append(ps_list)
    return ds_power_spectra

def plot_ds_power_spectra(ax: plt.Axes, measurements: list, quantiles: list = [0, 1, 2, 3, 4] , ell: int = 0, add_handles: bool = True, **kwargs) -> None:
    """
    Plot density split power spectra on a given axis.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on.
    measurements : list
        List of density split power spectrum objects to plot.
    quantiles : list, optional
        List of quantiles to plot. Defaults to [0, 1, 2, 3, 4] .
    ell : int, optional
        Multipole moment to plot. Defaults to 0.
    add_handles : bool, optional
        Whether to add legend handles for the quantiles and multipoles. Defaults to True.
    **kwargs
        Additional keyword arguments to pass to the plot function.
    
    Returns
    -------
    plt.Axes
        The matplotlib axis with the plotted density split power spectra.
    """
    kwargs.setdefault('alpha', 0.1)
    _c = kwargs.pop('c', None) # Ensure default color cycling if not specified
    
    for ps_list in measurements:
        for i, q in enumerate(quantiles):
            pss = ps_list[i]
            k = pss.get(ell).coords('k')
            pk = pss.get(ell).value()
            c =  _c if _c is not None else f'C{i}'
            ax.plot(k, pk*k**2, c=c, **kwargs)
    
    handles = []
    for i, q in enumerate(quantiles):
        color = _c if _c is not None else f'C{i}'
        kwargs['alpha'] = 1.0 # Make legend handles fully opaque
        handles.append(plt.Line2D([0], [0], color=color, label=f'DS={q}', **kwargs)) 
    if add_handles:
        ax.legend(handles=handles, title='Quantiles', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return ax

#%% Interactive plotting functions
def plot_interactive(
    stat_name: str,
    cosmologies: list[int],
    phases: list[int],
    seeds: list[int],
    sim_type: str,
    dir: str | Path,
    hod_idx: int = 0,
    **kwargs,
):

    load_kwargs = {}
    if stat_name == "tpcf":
        loader = get_tpcf
        plotter = plot_tpcf
    elif stat_name in ["quantile_correlation", "quantile_data_correlation"]:
        loader = get_ds_cf
        plotter = plot_ds_cf
        load_kwargs.update(
            stat_name = stat_name,
            quantiles = kwargs.pop('quantiles', [0,1,2,3,4]),
        )
    elif stat_name == "power_spectrum":
        loader = get_power_spectra
        plotter = plot_power_spectra
    else:
        raise ValueError(f"Unknown stat_name: {stat_name}")

    layout = widgets.Layout(width="150px")
    w_cosmo = widgets.IntText(value=cosmologies[0], description="Cosmo", layout=layout)
    w_phase = widgets.IntText(value=phases[0], description="Phase", layout=layout)
    w_seed = widgets.IntText(value=seeds[0], description="Seed", layout=layout)
    w_hod = widgets.IntText(value=hod_idx, description="HOD idx", layout=layout)

    controls = widgets.HBox([w_cosmo, w_phase, w_seed, w_hod])
    output = widgets.Output()

    def _update(*_):
        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            
            load_kwargs.update(
                dir=dir,
                sim_type=sim_type,
                cosmologies=[w_cosmo.value],
                phases=[w_phase.value],
                seeds=[w_seed.value],
                hod_idx=w_hod.value,
            )

            measurements = loader(**load_kwargs)
            # print(measurements)
            if len(measurements) == 0:
                ax.text(
                    0.5, 0.5, "No data found",
                    ha="center", va="center",
                    transform=ax.transAxes,
                )
            else:
                plotter(ax, measurements, **kwargs)

            # Set labels with generic names that do not depend on measurements
            ax.set_xlabel("Measurement Scale")
            ax.set_ylabel("Measurement Value")
            ax.set_title(
                f"{stat_name} | "
                f"c{w_cosmo.value:03d} "
                f"ph{w_phase.value:03d} "
                f"seed{w_seed.value} "
                f"hod{w_hod.value}"
            )
            
            plt.show()

    for w in (w_cosmo, w_phase, w_seed, w_hod):
        w.observe(_update, names="value")

    display(controls, output)
    _update()