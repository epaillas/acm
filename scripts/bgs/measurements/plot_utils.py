import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


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


def plot_parameters_histogram(parameters: list, names: list[str], mapping: dict = None, **kwargs) -> tuple:
    """
    Plot histograms of specified parameters from a list of parameter arrays.
    
    Parameters
    ----------
    parameters : list
        List of parameter dicts or structured arrays with dtype column names associated with parameters.
    names : list[str]
        List of parameter names to plot histograms for.
    mapping : dict, optional
        Dictionary mapping parameter names to labels for the x-axis, by default None. If None, `parameter` names are used as labels.
    **kwargs
        Additional keyword arguments to pass to `plt.hist`.
        Can also include:
        - figsize : tuple, optional
            Figure size, by default (8, 4 * len(names)).
        - labels : list[str], optional
            List of labels for each parameter array, by default None. Must match length of `parameters` if provided.
        - colors : list[str], optional
            List of colors for each parameter array, by default ['C0', 'C1', ...]. Must match length of `parameters` if provided.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects containing the histograms.
    """
    
    figsize = kwargs.pop('figsize', (4, 2 * len(names)))
    labels = kwargs.pop('labels', None)
    colors = kwargs.pop('colors', [f'C{i}' for i in range(len(parameters))])

    fig, ax = plt.subplots(len(names), 1, figsize=figsize)
    
    if labels is not None:
        assert len(labels) == len(parameters), "Length of labels must match length of parameters"
    for i, param in enumerate(names):
        for j, p in enumerate(parameters):
            if labels:
                ax[i].hist(p[param].flatten(), color=colors[j], label=labels[j], **kwargs)
            else:
                ax[i].hist(p[param].flatten(), color=colors[j], **kwargs)
        l = mapping.get(param, param) if mapping else param
        ax[i].set_xlabel(l)
    return fig, ax


def plot_parameters_triangle(parameters: list, names: list[str], mapping: dict = None, **kwargs) -> tuple:
    """
    Plot a triangle scatter plot of specified parameters names.

    Parameters
    ----------
    parameters : list
        List of parameters parameter dicts or structured arrays with dtype column names associated with names.
    names : list[str]
        List of parameter names to include in the triangle plot.
    mapping : dict, optional
        Dictionary mapping `parameter` names to labels for the axes, by default None. If None, `parameter` names are used as labels.
    **kwargs
        Additional keyword arguments to pass to `plt.scatter`.
        Can also include:
        - figsize : tuple, optional
            Figure size, by default (3 * len(names), 3 * len(names)).
        - labels : list[str], optional
            List of labels for each parameter array, by default None. Must match length of `parameters`
            if provided.
        - colors : list[str], optional
            List of colors for each parameter array, by default ['C0', 'C1', ...]. Must match length of `parameters` if provided.
        - bins : int, optional
            Number of bins for the diagonal histograms, by default 30.
        - histtype : str, optional
            Type of histogram for the diagonal plots, by default 'step'.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        The figure and axes objects containing the triangle plot.
    """
    figsize = kwargs.pop('figsize', (3 * len(names), 3 * len(names)))
    labels = kwargs.pop('labels', None)
    colors = kwargs.pop('colors', [f'C{i}' for i in range(len(parameters))])
    bins = kwargs.pop('bins', 30)
    histtype = kwargs.pop('histtype', 'step')
    alpha = kwargs.pop('alpha', 1/len(parameters))
    s = kwargs.pop('s', 1)  # size of scatter points
    
    fig, axes = plt.subplots(len(names), len(names), figsize=figsize)
    
    for i, x_param in enumerate(names):
        for j, y_param in enumerate(names):
            ax = axes[i, j]
            for k, p in enumerate(parameters):
                if i == j:
                    ax.hist(p[x_param], bins=bins, color=colors[k], histtype=histtype, alpha=alpha)
                elif i > j:
                    ax.scatter(p[y_param], p[x_param], color=colors[k], s=s, alpha=alpha, **kwargs)
                else:
                    ax.axis('off')
    # Set labels on bottom and left axis
    for i, param in enumerate(names):
        l = mapping.get(param, param) if mapping else param
        axes[-1, i].set_xlabel(l)
        axes[i, 0].set_ylabel(l)
    # Set legend only on the top-right plot
    handles = [plt.Line2D([0], [0], color=colors[k], lw=2) for k in range(len(parameters))]
    if labels is not None:
        fig.legend(handles, labels)
    return fig, axes