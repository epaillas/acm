"""
Train an emulator for the galaxy number density.

The emulator is a fully connected network (FCN) that maps cosmological and HOD
parameters to the galaxy number density. Training data are drawn from the
AbacusSummit simulation HOD catalogs constructed with the ACM pipeline.
"""

import argparse
import numpy as np
from pathlib import Path
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
from acm.utils.default import cosmo_list  # List of cosmologies in AbacusSummit
from acm.utils.abacus import load_abacus_cosmologies
import torch
torch.set_float32_matmul_precision('high')

# ---------------------------------------------------------------------------
# Hardcoded paths
# ---------------------------------------------------------------------------
COSMO_FILE = Path('/pscratch/sd/e/epaillas/emc/AbacusSummit.csv')
HOD_PARAMS_FILE = Path('/pscratch/sd/n/ntbfin/emulator/hods/hod_params.npy')
NGAL_FILE = Path('/pscratch/sd/n/ntbfin/emulator/hods/n_gal.npy')

# ---------------------------------------------------------------------------
# Cosmological parameter configuration
# ---------------------------------------------------------------------------
COSMO_PARAM_NAMES = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
COSMO_PARAMS_MAPPING = {'alpha_s': 'nrun'}

# ---------------------------------------------------------------------------
# Statistic label
# ---------------------------------------------------------------------------
STATISTIC = 'number_density_downsampled'


def get_x():
    """Build the input feature matrix X for the emulator.

    For each cosmology in ``cosmo_list``, the function concatenates the
    cosmological parameters (broadcast to every HOD realisation) with the
    corresponding HOD parameters, producing one row per (cosmology, HOD) pair.

    Returns
    -------
    x : np.ndarray
        2-D array of shape ``(N_samples, N_cosmo_params + N_hod_params)``
        containing the concatenated input features.
    cosmo_labels : np.ndarray of int, shape (N_samples,)
        Cosmology index for each row, used to build cosmology-level splits.
    """
    cosmo_params = load_abacus_cosmologies(
        COSMO_FILE,
        cosmologies=cosmo_list,
        parameters=COSMO_PARAM_NAMES,
        mapping=COSMO_PARAMS_MAPPING,
    )

    # Enforce parameter ordering after renaming via the mapping
    x_cosmo_names = COSMO_PARAM_NAMES.copy()
    for key, value in COSMO_PARAMS_MAPPING.items():
        x_cosmo_names[x_cosmo_names.index(key)] = value

    hod_params = np.load(HOD_PARAMS_FILE, allow_pickle=True).item()

    x, cosmo_labels = [], []
    for cosmo_idx in cosmo_list:
        # HOD parameters: dict → array of shape (N_hod, N_hod_params)
        x_hod = hod_params[f'c{cosmo_idx:03}']
        x_hod = np.array([x_hod[param] for param in x_hod.keys()]).T

        # Cosmological parameters: broadcast to match HOD realisations
        x_cosmo = cosmo_params[f'c{cosmo_idx:03}']
        x_cosmo = np.array([x_cosmo[param] for param in x_cosmo_names])
        x_cosmo = np.repeat(x_cosmo.reshape(1, -1), x_hod.shape[0], axis=0)

        x.append(np.concatenate([x_cosmo, x_hod], axis=1))
        cosmo_labels.append(np.full(x_hod.shape[0], cosmo_idx, dtype=int))

    return np.concatenate(x), np.concatenate(cosmo_labels)


def get_y():
    """Load the target galaxy number density values.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(N_samples, 1)`` containing the galaxy number
        density for each (cosmology, HOD) pair.
    """
    data = np.load(NGAL_FILE, allow_pickle=True).item()
    y = np.concatenate([data[f'c{cosmo_idx:03}'] for cosmo_idx in cosmo_list])
    return y.reshape(-1, 1)


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the following attributes:

        ``model_dir`` : str
            Directory where model checkpoints and logs are saved.
        ``max_epochs`` : int
            Maximum number of training epochs.
        ``batch_size`` : int
            Mini-batch size used during training.
        ``learning_rate`` : float
            Initial learning rate for the Adam optimiser.
        ``devices`` : int
            Number of GPU (or CPU) devices to use for training.
        ``n_hidden`` : list of int
            Hidden layer widths of the FCN.
        ``val_cosmo_fraction`` : float
            Fraction of cosmologies held out for validation.
        ``weight_decay`` : float
            L2 regularisation strength.
    """
    parser = argparse.ArgumentParser(
        description='Train an FCN emulator for the galaxy number density.'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='./',
        help='Directory to save model checkpoints and TensorBoard logs (default: ./)',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=5000,
        help='Maximum number of training epochs (default: 5000)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Mini-batch size (default: 512)',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Initial learning rate (default: 1e-3)',
    )
    parser.add_argument(
        '--devices',
        type=int,
        default=1,
        help='Number of devices (GPUs/CPUs) to use for training (default: 1)',
    )
    parser.add_argument(
        '--transform',
        type=str,
        choices=['log', 'arcsinh'],
        default=None,
        help='Output transform to apply to n_gal before training (default: none).',
    )
    parser.add_argument(
        '--n-hidden',
        type=int,
        nargs='+',
        default=[256, 256, 256],
        metavar='N',
        help='Hidden layer widths, e.g. --n-hidden 256 256 256 (default: 256 256 256)',
    )
    parser.add_argument(
        '--val-cosmo-fraction',
        type=float,
        default=0.1,
        help='Fraction of cosmologies to hold out for validation (default: 0.1)',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0,
        help='L2 regularisation strength (default: 0.0)',
    )
    return parser.parse_args()


def main():
    """Train the galaxy number density emulator end-to-end."""
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    x, cosmo_labels = get_x()
    y = get_y()
    print(f'Loaded data with shape: x={x.shape}, y={y.shape}')

    # ------------------------------------------------------------------
    # Cosmology-level train/val split
    # ------------------------------------------------------------------
    n_val_cosmo = max(1, int(len(cosmo_list) * args.val_cosmo_fraction))
    val_cosmos  = set(cosmo_list[:n_val_cosmo])
    train_cosmos = set(cosmo_list[n_val_cosmo:])
    val_idx   = np.where(np.isin(cosmo_labels, list(val_cosmos)))[0].tolist()
    train_idx = np.where(np.isin(cosmo_labels, list(train_cosmos)))[0].tolist()
    print(f'Validation cosmologies ({n_val_cosmo}): {sorted(val_cosmos)}')
    print(f'Training cosmologies  ({len(train_cosmos)}): {sorted(train_cosmos)}')

    # ------------------------------------------------------------------
    # Optional output transform
    # ------------------------------------------------------------------
    if args.transform == 'log':
        transform = LogTransform()
    elif args.transform == 'arcsinh':
        transform = ArcsinhTransform()
    else:
        transform = None

    if transform is not None:
        print(f'Applying {args.transform} transform to outputs.')
        y = transform.transform(y)

    # ------------------------------------------------------------------
    # Set up PyTorch Lightning data module
    # ------------------------------------------------------------------
    data = ArrayDataModule(
        x=torch.Tensor(x),
        y=torch.Tensor(y),
        val_idx=val_idx,
        batch_size=args.batch_size,
        num_workers=0,
    )
    data.setup()

    # ------------------------------------------------------------------
    # Compute normalisation statistics from the training split only
    # ------------------------------------------------------------------
    x_train = data.ds_train.tensors[0].numpy()
    y_train = data.ds_train.tensors[1].numpy()
    mean_x, std_x = x_train.mean(axis=0), x_train.std(axis=0)
    mean_y, std_y = y_train.mean(axis=0), y_train.std(axis=0)

    # ------------------------------------------------------------------
    # Build the fully connected network
    # ------------------------------------------------------------------
    model = FCN(
        n_input=data.n_input,
        n_output=data.n_output,
        n_hidden=args.n_hidden,
        dropout_rate=0.0,
        learning_rate=args.learning_rate,
        scheduler_patience=100,
        scheduler_factor=0.5,
        scheduler_threshold=1e-5,
        weight_decay=args.weight_decay,
        act_fn='learned_sigmoid',
        loss='mae',
        training=True,
        transform_output=transform,
        mean_input=mean_x,
        std_input=std_x,
        mean_output=mean_y,
        std_output=std_y,
        standarize_output=True,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    val_loss, model, early_stop_callback = train.fit(
        data=data,
        model=model,
        model_dir=str(model_dir),
        max_epochs=args.max_epochs,
        devices=args.devices,
        logger='tensorboard',
        log_dir=str(model_dir),
    )

    print(f'Training complete. Best validation loss: {val_loss:.6f}')
    print(f'Model saved to: {model_dir}')


if __name__ == '__main__':
    main()