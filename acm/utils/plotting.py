from collections.abc import Callable
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np


# %% Styling decorators
def set_plot_style(func: Callable) -> Callable:
    """Set the plotting style to acm standard."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> object:
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        return func(*args, **kwargs)

    return wrapper


# %% Plotting functions
def plot_parameters_histogram(
    parameters: list,
    names: list[str],
    mapping: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot histograms of specified parameters from a list of parameter arrays.

    Parameters
    ----------
    parameters : list
        List of parameter dicts or structured arrays with dtype column names associated with parameters.
    names : list[str]
        List of parameter names to plot histograms for.
    mapping : dict, optional
        Dictionary mapping parameter names to labels for the x-axis, by default None. If None, parameter names are used as labels.
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
    figsize = kwargs.pop("figsize", (4, 2 * len(names)))
    labels = kwargs.pop("labels", None)
    colors = kwargs.pop("colors", [f"C{i}" for i in range(len(parameters))])

    fig, ax = plt.subplots(len(names), 1, figsize=figsize)
    ax = np.atleast_1d(ax)  # Ensure ax is always an array

    if labels is not None and len(labels) != len(parameters):
        raise ValueError("Length of labels must match length of parameters")
    for i, param in enumerate(names):
        for j, p in enumerate(parameters):
            if labels is not None:
                ax[i].hist(
                    p[param].flatten(), color=colors[j], label=labels[j], **kwargs
                )
            else:
                ax[i].hist(p[param].flatten(), color=colors[j], **kwargs)
        _l = mapping.get(param, param) if mapping else param
        ax[i].set_xlabel(_l)
    return fig, ax


def plot_parameters_triangle(
    parameters: list,
    names: list[str],
    mapping: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a triangle scatter plot of specified parameter names.

    Parameters
    ----------
    parameters : list
        List of parameter dicts or structured arrays with dtype column names associated with names.
    names : list[str]
        List of parameter names to include in the triangle plot.
    mapping : dict, optional
        Dictionary mapping parameter names to labels for the axes, by default None. If None, parameter names are used as labels.
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
    figsize = kwargs.pop("figsize", (3 * len(names), 3 * len(names)))
    labels = kwargs.pop("labels", None)
    colors = kwargs.pop("colors", [f"C{i}" for i in range(len(parameters))])
    bins = kwargs.pop("bins", 30)
    histtype = kwargs.pop("histtype", "step")
    alpha = kwargs.pop("alpha", 1.0 / len(parameters) if len(parameters) else 1.0)
    s = kwargs.pop("s", 1)  # size of scatter points

    fig, axes = plt.subplots(len(names), len(names), figsize=figsize)
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D array

    for i, x_param in enumerate(names):
        for j, y_param in enumerate(names):
            ax = axes[i, j]
            for k, p in enumerate(parameters):
                if i == j:
                    ax.hist(
                        p[x_param],
                        bins=bins,
                        color=colors[k],
                        histtype=histtype,
                        alpha=alpha,
                    )
                elif i > j:
                    ax.scatter(
                        p[y_param],
                        p[x_param],
                        color=colors[k],
                        s=s,
                        alpha=alpha,
                        **kwargs,
                    )
                else:
                    ax.axis("off")
    # Set labels on bottom and left axis
    for i, param in enumerate(names):
        _l = mapping.get(param, param) if mapping else param
        axes[-1, i].set_xlabel(_l)
        axes[i, 0].set_ylabel(_l)
    # Set legend only on the top-right plot
    handles = [
        plt.Line2D([0], [0], color=colors[k], lw=2) for k in range(len(parameters))
    ]
    if labels is not None:
        fig.legend(handles, labels)
    return fig, axes
