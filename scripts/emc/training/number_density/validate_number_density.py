"""
Validate the accuracy of the galaxy number density emulator.

Loads a trained FCN checkpoint and the original simulation data, runs
predictions over the full data set, computes summary error statistics, and
produces a set of diagnostic plots:

  1. Predicted vs. truth scatter (1-to-1 diagonal).
  2. Fractional residual distribution (histogram).
  3. Fractional residual as a function of the true n_gal.
  4. Per-cosmology MAE summary.

Usage
-----
    python validate_ngal.py --checkpoint /path/to/epoch=X-step=Y.ckpt
    python validate_ngal.py --checkpoint model.ckpt --output-dir ./validation/
    python validate_ngal.py --checkpoint model.ckpt --cosmo-idx 0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

from acm.utils.abacus import load_abacus_cosmologies
from acm.utils.default import cosmo_list
from sunbird.emulators import FCN

# ---------------------------------------------------------------------------
# Paths  (mirror train_ngal.py)
# ---------------------------------------------------------------------------
COSMO_FILE = Path('/pscratch/sd/e/epaillas/emc/AbacusSummit.csv')
HOD_PARAMS_FILE = Path('/pscratch/sd/n/ntbfin/emulator/hods/hod_params.npy')
NGAL_FILE = Path('/pscratch/sd/n/ntbfin/emulator/hods/n_gal.npy')

COSMO_PARAM_NAMES = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
COSMO_PARAMS_MAPPING = {'alpha_s': 'nrun'}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_x(cosmo_indices=None):
    """Build the input feature matrix for the given cosmologies.

    Parameters
    ----------
    cosmo_indices : list of int or None
        Subset of cosmology indices to load. Defaults to the full
        ``cosmo_list``.

    Returns
    -------
    x : np.ndarray, shape (N, N_params)
    cosmo_labels : np.ndarray of int, shape (N,)
        Cosmology index for each row, useful for per-cosmology diagnostics.
    """
    if cosmo_indices is None:
        cosmo_indices = cosmo_list

    cosmo_params = load_abacus_cosmologies(
        COSMO_FILE,
        cosmologies=cosmo_indices,
        parameters=COSMO_PARAM_NAMES,
        mapping=COSMO_PARAMS_MAPPING,
    )

    x_cosmo_names = COSMO_PARAM_NAMES.copy()
    for key, value in COSMO_PARAMS_MAPPING.items():
        x_cosmo_names[x_cosmo_names.index(key)] = value

    hod_params = np.load(HOD_PARAMS_FILE, allow_pickle=True).item()

    x, cosmo_labels = [], []
    for cosmo_idx in cosmo_indices:
        x_hod_dict = hod_params[f'c{cosmo_idx:03}']
        x_hod = np.array([x_hod_dict[p] for p in x_hod_dict.keys()]).T

        x_cosmo = cosmo_params[f'c{cosmo_idx:03}']
        x_cosmo = np.array([x_cosmo[p] for p in x_cosmo_names])
        x_cosmo = np.repeat(x_cosmo.reshape(1, -1), x_hod.shape[0], axis=0)

        x.append(np.concatenate([x_cosmo, x_hod], axis=1))
        cosmo_labels.append(np.full(x_hod.shape[0], cosmo_idx, dtype=int))

    return np.concatenate(x), np.concatenate(cosmo_labels)


def get_y(cosmo_indices=None):
    """Load the true galaxy number density values.

    Parameters
    ----------
    cosmo_indices : list of int or None
        Subset of cosmology indices. Defaults to the full ``cosmo_list``.

    Returns
    -------
    np.ndarray, shape (N,)
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
        Path to the ``.ckpt`` file.

    Returns
    -------
    FCN
        Model in evaluation mode on CPU.
    """
    model = FCN.load_from_checkpoint(
        checkpoint_path, strict=True, weights_only=False
    )
    model.eval().to('cpu')
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute scalar accuracy metrics.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)

    Returns
    -------
    dict with keys: mae, rmse, rel_err_median, rel_err_p16, rel_err_p84, r2
    """
    residuals = y_pred - y_true
    frac_residuals = residuals / np.abs(y_true)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    return {
        'mae':            float(np.mean(np.abs(residuals))),
        'rmse':           float(np.sqrt(np.mean(residuals ** 2))),
        'rel_err_median': float(np.median(frac_residuals) * 100),
        'rel_err_p16':    float(np.percentile(frac_residuals, 16) * 100),
        'rel_err_p84':    float(np.percentile(frac_residuals, 84) * 100),
        'r2':             float(1.0 - ss_res / ss_tot),
    }


def print_metrics(metrics: dict, label: str = 'All cosmologies') -> None:
    """Pretty-print a metrics dict returned by :func:`compute_metrics`."""
    print(f'\n--- {label} ---')
    print(f"  MAE                : {metrics['mae']:.4e}")
    print(f"  RMSE               : {metrics['rmse']:.4e}")
    print(f"  Rel. error (median): {metrics['rel_err_median']:+.3f} %")
    print(f"  Rel. error (16-84) : {metrics['rel_err_p16']:.3f} % to {metrics['rel_err_p84']:.3f} %")
    print(f"  R²                 : {metrics['r2']:.6f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pred_vs_truth(y_true, y_pred, output_path, title=''):
    """Scatter plot of predicted vs. true n_gal with residual panel.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N,)
    output_path : Path
    title : str
    """
    frac_res = (y_pred - y_true) / np.abs(y_true) * 100  # percent

    fig, axes = plt.subplots(2, 1, figsize=(5, 7),
                             gridspec_kw={'height_ratios': [3, 1.2]},
                             sharex=False)

    # --- Top panel: pred vs truth ---
    ax = axes[0]
    lims = [min(y_true.min(), y_pred.min()) * 0.98,
            max(y_true.max(), y_pred.max()) * 1.02]
    ax.plot(lims, lims, 'k--', lw=1, label='1:1')
    ax.scatter(y_true, y_pred, s=1.5, alpha=0.4, rasterized=True, color='steelblue')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r'$n_{\rm gal}$ (truth)', fontsize=11)
    ax.set_ylabel(r'$n_{\rm gal}$ (predicted)', fontsize=11)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)

    # Annotate with R²
    metrics = compute_metrics(y_true, y_pred)
    ax.text(0.05, 0.93, f"$R^2 = {metrics['r2']:.5f}$",
            transform=ax.transAxes, fontsize=9, va='top')

    # --- Bottom panel: fractional residuals vs truth ---
    ax2 = axes[1]
    ax2.axhline(0, color='k', lw=1, ls='--')
    ax2.scatter(y_true, frac_res, s=1.5, alpha=0.4, rasterized=True, color='steelblue')
    ax2.set_xlabel(r'$n_{\rm gal}$ (truth)', fontsize=11)
    ax2.set_ylabel(r'$\Delta n_{\rm gal} / n_{\rm gal}$ [%]', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {output_path}')


def plot_residual_histogram(y_true, y_pred, output_path, title=''):
    """Histogram of fractional residuals.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N,)
    output_path : Path
    title : str
    """
    frac_res = (y_pred - y_true) / np.abs(y_true) * 100  # percent

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(frac_res, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.axvline(np.percentile(frac_res, 16), color='tomato', lw=1, ls=':', label='16th / 84th pct')
    ax.axvline(np.percentile(frac_res, 84), color='tomato', lw=1, ls=':')
    ax.axvline(np.median(frac_res), color='darkorange', lw=1.5, ls='-', label='Median')
    ax.set_xlabel(r'$(n_{\rm gal,pred} - n_{\rm gal,true})\,/\,n_{\rm gal,true}$ [%]', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {output_path}')


def plot_per_cosmo_mae(y_true, y_pred, cosmo_labels, cosmo_indices, output_path, title=''):
    """Bar chart of per-cosmology MAE.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N,)
    cosmo_labels : np.ndarray of int, shape (N,)
    cosmo_indices : list of int
    output_path : Path
    title : str
    """
    maes = []
    for cidx in cosmo_indices:
        mask = cosmo_labels == cidx
        if mask.sum() == 0:
            maes.append(np.nan)
            continue
        maes.append(np.mean(np.abs(y_pred[mask] - y_true[mask])))

    fig, ax = plt.subplots(figsize=(max(6, len(cosmo_indices) * 0.3), 4))
    x_pos = np.arange(len(cosmo_indices))
    bars = ax.bar(x_pos, maes, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'c{i:03}' for i in cosmo_indices], rotation=90, fontsize=6)
    ax.set_xlabel('Cosmology', fontsize=10)
    ax.set_ylabel(r'MAE ($n_{\rm gal}$)', fontsize=10)
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {output_path}')


def plot_residual_vs_ngal(y_true, y_pred, output_path, title='', n_bins=30):
    """Binned fractional residual (median ± 1σ band) as a function of n_gal.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N,)
    output_path : Path
    title : str
    n_bins : int
        Number of bins along the n_gal axis.
    """
    frac_res = (y_pred - y_true) / np.abs(y_true) * 100  # percent
    edges = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    centres, medians, p16, p84 = [], [], [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() < 2:
            continue
        centres.append(0.5 * (lo + hi))
        medians.append(np.median(frac_res[mask]))
        p16.append(np.percentile(frac_res[mask], 16))
        p84.append(np.percentile(frac_res[mask], 84))

    centres, medians, p16, p84 = map(np.array, (centres, medians, p16, p84))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(0, color='k', lw=1, ls='--')
    ax.fill_between(centres, p16, p84, alpha=0.3, color='steelblue', label='16–84th pct')
    ax.plot(centres, medians, color='steelblue', lw=1.5, label='Median')
    ax.set_xlabel(r'$n_{\rm gal}$ (truth)', fontsize=11)
    ax.set_ylabel(r'$\Delta n_{\rm gal} / n_{\rm gal}$ [%]', fontsize=10)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        ``checkpoint`` : str
            Path to the trained ``.ckpt`` file.
        ``output_dir`` : str
            Directory for diagnostic plots and metrics (default: ``./validation``).
        ``cosmo_idx`` : int or None
            Restrict validation to a single cosmology index.
        ``no_plots`` : bool
            If set, skip figure generation and only print metrics.
    """
    parser = argparse.ArgumentParser(
        description='Validate the n_gal FCN emulator against simulation data.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the trained FCN checkpoint (.ckpt).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./validation',
        help='Directory to write diagnostic plots (default: ./validation).',
    )
    parser.add_argument(
        '--cosmo-idx',
        type=int,
        default=None,
        metavar='IDX',
        help='Restrict validation to a single AbacusSummit cosmology index.',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        default=False,
        help='Skip figure generation; only print metrics to stdout.',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the validation pipeline."""
    args = parse_args()

    # ------------------------------------------------------------------
    # Cosmology selection
    # ------------------------------------------------------------------
    if args.cosmo_idx is not None:
        if args.cosmo_idx not in cosmo_list:
            raise ValueError(
                f'cosmo_idx {args.cosmo_idx} is not in cosmo_list. '
                f'Available indices: {cosmo_list}'
            )
        cosmo_indices = [args.cosmo_idx]
        suffix = f'c{args.cosmo_idx:03}'
    else:
        cosmo_indices = list(cosmo_list)
        suffix = 'all'

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print('Loading data...')
    x, cosmo_labels = get_x(cosmo_indices=cosmo_indices)
    y_true = get_y(cosmo_indices=cosmo_indices)
    print(f'  x: {x.shape}  |  y: {y_true.shape}')

    # ------------------------------------------------------------------
    # Load model and run predictions
    # ------------------------------------------------------------------
    print(f'Loading model from {args.checkpoint} ...')
    model = load_model(args.checkpoint)

    with torch.no_grad():
        y_pred = model.get_prediction(torch.Tensor(x)).numpy().ravel()
    print(f'  Predictions range: [{y_pred.min():.4e}, {y_pred.max():.4e}]')

    # ------------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, label=f'Global ({suffix})')

    # ------------------------------------------------------------------
    # Per-cosmology metrics (only when >1 cosmology)
    # ------------------------------------------------------------------
    if len(cosmo_indices) > 1:
        per_cosmo_maes = {}
        for cidx in cosmo_indices:
            mask = cosmo_labels == cidx
            if mask.sum() == 0:
                continue
            m = compute_metrics(y_true[mask], y_pred[mask])
            per_cosmo_maes[cidx] = m['mae']

        worst = max(per_cosmo_maes, key=per_cosmo_maes.get)
        best  = min(per_cosmo_maes, key=per_cosmo_maes.get)
        print(f'\n  Best  cosmology: c{best:03}  (MAE = {per_cosmo_maes[best]:.4e})')
        print(f'  Worst cosmology: c{worst:03}  (MAE = {per_cosmo_maes[worst]:.4e})')

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if args.no_plots:
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f'n_gal emulator — {suffix}'

    print('\nGenerating plots...')

    plot_pred_vs_truth(
        y_true, y_pred,
        output_path=out_dir / f'pred_vs_truth_{suffix}.pdf',
        title=title,
    )
    plot_residual_histogram(
        y_true, y_pred,
        output_path=out_dir / f'residual_hist_{suffix}.pdf',
        title=title,
    )
    plot_residual_vs_ngal(
        y_true, y_pred,
        output_path=out_dir / f'residual_vs_ngal_{suffix}.pdf',
        title=title,
    )
    if len(cosmo_indices) > 1:
        plot_per_cosmo_mae(
            y_true, y_pred, cosmo_labels, cosmo_indices,
            output_path=out_dir / f'per_cosmo_mae_{suffix}.pdf',
            title=title,
        )

    print(f'\nAll outputs written to {out_dir}')


if __name__ == '__main__':
    main()
