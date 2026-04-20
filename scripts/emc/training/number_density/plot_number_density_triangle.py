"""
Triangle plot of predicted galaxy number density in the HOD parameter space.

Loads a trained FCN emulator checkpoint, evaluates it on the full training
data, and produces a lower-triangular corner scatter plot where each point is
coloured by the predicted (or true) galaxy number density.

Usage
-----
    python plot_ngal_triangle.py --checkpoint /path/to/epoch=X-step=Y.ckpt
    python plot_ngal_triangle.py --checkpoint model.ckpt --use-truth --output ngal.pdf
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc  # registers 'managua' and other scientific colormaps
import numpy as np
import torch

from sunbird.emulators import FCN
from acm.utils.default import cosmo_list
from acm.utils.abacus import load_abacus_cosmologies

# ---------------------------------------------------------------------------
# Paths  (mirror train_ngal.py)
# ---------------------------------------------------------------------------
COSMO_FILE = Path('/pscratch/sd/e/epaillas/emc/AbacusSummit.csv')
HOD_PARAMS_FILE = Path('/pscratch/sd/n/ntbfin/emulator/hods/hod_params.npy')
NGAL_FILE = Path('/pscratch/sd/n/ntbfin/emulator/hods/n_gal.npy')

COSMO_PARAM_NAMES = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
COSMO_PARAMS_MAPPING = {'alpha_s': 'nrun'}

# Target observed number density (used to centre the diverging colour scale)
TARGET_DENSITY = 4.85e-4


# Number of leading cosmological parameters to skip when plotting
N_COSMO_PARAMS = len(COSMO_PARAM_NAMES)

LABELS = {
    'omega_b':   r'$\omega_{\rm b}$',
    'omega_cdm': r'$\omega_{\rm cdm}$',
    'sigma8_m':  r'$\sigma_8$',
    'n_s':       r'$n_s$',
    'nrun':      r'$\alpha_s$',
    'N_ur':      r'$N_{\rm ur}$',
    'w0_fld':    r'$w_0$',
    'wa_fld':    r'$w_a$',
    'logM_1':    r'$\log M_1$',
    'logM_cut':  r'$\log M_{\rm cut}$',
    'alpha':     r'$\alpha$',
    'alpha_s':   r'$\alpha_{\rm vel, s}$',
    'alpha_c':   r'$\alpha_{\rm vel, c}$',
    'sigma':     r'$\log \sigma$',
    'kappa':     r'$\kappa$',
    'A_cen':     r'$A_{\rm cen}$',
    'A_sat':     r'$A_{\rm sat}$',
    'B_cen':     r'$B_{\rm cen}$',
    'B_sat':     r'$B_{\rm sat}$',
    's':         r'$s$',
    'fsigma8':   r'$f \sigma_8$',
    'Omega_m':   r'$\Omega_{\rm m}$',
    'H0':        r'$H_0$',
}


def get_x_and_names(cosmo_indices=None):
    """Build the input feature matrix and the ordered list of parameter names.

    Parameters
    ----------
    cosmo_indices : list of int or None, optional
        Subset of cosmology indices to load. Defaults to the full ``cosmo_list``.

    Returns
    -------
    x : np.ndarray
        2-D array of shape ``(N_samples, N_cosmo_params + N_hod_params)``.
    param_names : list of str
        Ordered list of parameter names matching the columns of ``x``.
    """
    if cosmo_indices is None:
        cosmo_indices = cosmo_list

    cosmo_params = load_abacus_cosmologies(
        COSMO_FILE,
        cosmologies=cosmo_indices,
        parameters=COSMO_PARAM_NAMES,
        mapping=COSMO_PARAMS_MAPPING,
    )

    # Enforce parameter ordering after renaming
    x_cosmo_names = COSMO_PARAM_NAMES.copy()
    for key, value in COSMO_PARAMS_MAPPING.items():
        x_cosmo_names[x_cosmo_names.index(key)] = value

    hod_params = np.load(HOD_PARAMS_FILE, allow_pickle=True).item()

    x, hod_names = [], None
    for cosmo_idx in cosmo_indices:
        x_hod_dict = hod_params[f'c{cosmo_idx:03}']
        if hod_names is None:
            hod_names = list(x_hod_dict.keys())
        x_hod = np.array([x_hod_dict[p] for p in hod_names]).T  # (N_hod, N_hod_params)

        x_cosmo = cosmo_params[f'c{cosmo_idx:03}']
        x_cosmo = np.array([x_cosmo[p] for p in x_cosmo_names])
        x_cosmo = np.repeat(x_cosmo.reshape(1, -1), x_hod.shape[0], axis=0)

        x.append(np.concatenate([x_cosmo, x_hod], axis=1))

    param_names = x_cosmo_names + hod_names
    return np.concatenate(x), param_names


def get_y(cosmo_indices=None):
    """Load the true galaxy number density values.

    Parameters
    ----------
    cosmo_indices : list of int or None, optional
        Subset of cosmology indices to load. Defaults to the full ``cosmo_list``.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(N_samples,)`` containing the galaxy number density.
    """
    if cosmo_indices is None:
        cosmo_indices = cosmo_list
    data = np.load(NGAL_FILE, allow_pickle=True).item()
    y = np.concatenate([data[f'c{cosmo_idx:03}'] for cosmo_idx in cosmo_indices])
    return y.ravel()


def load_model(checkpoint_path: str) -> FCN:
    """Load a trained FCN emulator from a Lightning checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.ckpt`` file produced during training.

    Returns
    -------
    FCN
        Emulator in evaluation mode on CPU.
    """
    model = FCN.load_from_checkpoint(checkpoint_path, strict=True)
    model.eval().to('cpu')
    return model


def make_triangle_plot(
    x: np.ndarray,
    y: np.ndarray,
    param_names: list,
    skip_cosmo: bool = True,
    ndim: int | None = None,
    cmap_name: str = 'managua',
    point_size: float = 0.5,
    colorbar_label: str = r'$n_{\rm gal}$',
    title: str = 'Galaxy number density — HOD parameter space',
    target_density: float | None = TARGET_DENSITY,
) -> plt.Figure:
    """Create a lower-triangular corner scatter plot.

    Each panel shows a 2-D projection of the parameter space, with points
    coloured by the galaxy number density ``y``.

    When ``target_density`` is set the colour scale uses
    `~matplotlib.colors.TwoSlopeNorm` centred at that value so regions
    above and below are immediately distinguishable.  A dashed line is
    also drawn on the colourbar at the target value.

    Parameters
    ----------
    x : np.ndarray
        Input feature matrix of shape ``(N_samples, N_params)``.
    y : np.ndarray
        1-D array of colour values (number density), shape ``(N_samples,)``.
    param_names : list of str
        Ordered parameter names matching the columns of ``x``.
    skip_cosmo : bool, optional
        If ``True`` (default), skip the first ``N_COSMO_PARAMS`` columns and
        show only HOD parameters.
    ndim : int or None, optional
        Number of HOD parameters to include in the plot. Defaults to all
        available HOD parameters.
    cmap_name : str, optional
        Matplotlib colour map name (default ``'managua'``).
    point_size : float, optional
        Scatter point size (default ``0.5``).
    colorbar_label : str, optional
        Label for the colour bar.
    title : str, optional
        Figure suptitle.
    target_density : float or None, optional
        Observed galaxy number density used to centre the diverging colour
        scale. Set to ``None`` to use a linear norm instead.

    Returns
    -------
    matplotlib.figure.Figure
    """
    skip = N_COSMO_PARAMS if skip_cosmo else 0
    available = len(param_names) - skip
    if ndim is None:
        ndim = available
    ndim = min(ndim, available)

    param_indices = list(range(skip, skip + ndim))
    plot_names = [param_names[i] for i in param_indices]

    cmap = mpl.colormaps[cmap_name]
    if target_density is not None:
        vmin, vmax = y.min(), y.max()
        # Clamp vcenter so TwoSlopeNorm never fails if target is outside range
        vcenter = float(np.clip(target_density, vmin + 1e-30, vmax - 1e-30))
        norm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=y.min(), vmax=y.max())

    fig, axes = plt.subplots(ndim, ndim, figsize=(2 * ndim, 2 * ndim))

    for i, ipar in enumerate(param_indices):
        for j, jpar in enumerate(param_indices):
            ax = axes[i][j]

            if i < j:
                # Upper triangle — remove the axis
                fig.delaxes(ax)
                continue

            ax.scatter(
                x[:, jpar],
                x[:, ipar],
                c=cmap(norm(y)),
                s=point_size,
                rasterized=True,
            )

            # Axis labels only on the outer edges
            if i == ndim - 1:
                ax.set_xlabel(LABELS.get(plot_names[j], plot_names[j]), fontsize=15)
            else:
                ax.xaxis.set_visible(False)

            if j == 0:
                ax.set_ylabel(LABELS.get(plot_names[i], plot_names[i]), fontsize=15)
            else:
                ax.yaxis.set_visible(False)

            ax.tick_params(labelsize=12)

    # Colour bar spanning the full figure height
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    if isinstance(norm, mpl.colors.TwoSlopeNorm):
        # Explicit ticks: 5 evenly spaced on each side of vcenter,
        # plus vcenter itself — guarantees both halves are labelled
        # regardless of how compressed one half is.
        n_ticks = 5
        lower_ticks = np.linspace(norm.vmin, norm.vcenter, n_ticks, endpoint=False)
        upper_ticks = np.linspace(norm.vcenter, norm.vmax, n_ticks + 1)
        ticks = np.concatenate([lower_ticks, upper_ticks])
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{t:.1e}' for t in ticks])

    fig.suptitle(title, fontsize=25, y=0.9)
    return fig


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments:

        ``checkpoint`` : str
            Path to the trained FCN ``.ckpt`` file.
        ``output`` : str
            Output figure file (default: ``ngal_triangle.pdf``).
        ``use_truth`` : bool
            Colour points by ground-truth ``n_gal`` instead of model predictions.
        ``ndim`` : int or None
            Number of HOD dimensions to display (default: all).
        ``point_size`` : float
            Scatter marker size.
        ``cmap`` : str
            Matplotlib colour map name.
        ``include_cosmo`` : bool
            If set, include cosmological parameters in the triangle plot.
    """
    parser = argparse.ArgumentParser(
        description='Triangle plot of n_gal predictions in the HOD parameter space.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the trained FCN checkpoint (.ckpt).',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ngal_triangle.pdf',
        help='Output figure filename (default: ngal_triangle.pdf).',
    )
    parser.add_argument(
        '--use-truth',
        action='store_true',
        default=False,
        help='Colour points by ground-truth n_gal instead of model predictions.',
    )
    parser.add_argument(
        '--ndim',
        type=int,
        default=None,
        help='Number of HOD dimensions to include in the plot (default: all).',
    )
    parser.add_argument(
        '--point-size',
        type=float,
        default=0.5,
        help='Scatter marker size (default: 0.5).',
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default='managua_r',
        help='Matplotlib colour map name (default: managua).',
    )
    parser.add_argument(
        '--target-density',
        type=float,
        default=TARGET_DENSITY,
        metavar='N_GAL',
        help=(
            'Observed galaxy number density used to centre the diverging colour scale '
            f'(default: {TARGET_DENSITY:.2e}). Pass 0 to disable.'
        ),
    )
    parser.add_argument(
        '--include-cosmo',
        action='store_true',
        default=False,
        help='Include cosmological parameters in the triangle plot.',
    )
    parser.add_argument(
        '--cosmo-idx',
        type=int,
        default=None,
        metavar='IDX',
        help=(
            'Restrict the plot to a single AbacusSummit cosmology index '
            '(e.g. 0 for c000). Defaults to all cosmologies.'
        ),
    )
    return parser.parse_args()


def main():
    """Load model, run predictions, and save the triangle plot."""
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve which cosmologies to load
    # ------------------------------------------------------------------
    if args.cosmo_idx is not None:
        if args.cosmo_idx not in cosmo_list:
            raise ValueError(
                f'cosmo_idx {args.cosmo_idx} is not in cosmo_list. '
                f'Available indices: {cosmo_list}'
            )
        cosmo_indices = [args.cosmo_idx]
        print(f'Restricting to cosmology c{args.cosmo_idx:03}.')
    else:
        cosmo_indices = None  # use full cosmo_list inside helpers

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print('Loading data...')
    x, param_names = get_x_and_names(cosmo_indices=cosmo_indices)
    y_truth = get_y(cosmo_indices=cosmo_indices)
    print(f'  x shape: {x.shape}  |  y shape: {y_truth.shape}')
    print(f'  Parameters: {param_names}')

    # ------------------------------------------------------------------
    # Model predictions
    # ------------------------------------------------------------------
    if args.use_truth:
        print('Using ground-truth n_gal for colouring.')
        y_plot = y_truth
        colorbar_label = r'$n_{\rm gal}$ (truth)'
    else:
        print(f'Loading model from {args.checkpoint} ...')
        model = load_model(args.checkpoint)
        with torch.no_grad():
            y_plot = model.get_prediction(torch.Tensor(x)).numpy().ravel()
        print(f'  Predictions range: [{y_plot.min():.4e}, {y_plot.max():.4e}]')
        colorbar_label = r'$n_{\rm gal}$ (predicted)'

    # ------------------------------------------------------------------
    # Triangle plot
    # ------------------------------------------------------------------
    cosmo_label = f'c{args.cosmo_idx:03}' if args.cosmo_idx is not None else 'all cosmologies'
    title = f'Galaxy number density — HOD parameter space ({cosmo_label})'

    print('Making triangle plot...')
    fig = make_triangle_plot(
        x=x,
        y=y_plot,
        param_names=param_names,
        skip_cosmo=not args.include_cosmo,
        ndim=args.ndim,
        cmap_name=args.cmap,
        point_size=args.point_size,
        colorbar_label=colorbar_label,
        title=title,
        target_density=args.target_density if args.target_density != 0 else None,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=150)
    print(f'Saved figure to {output}')


if __name__ == '__main__':
    main()
