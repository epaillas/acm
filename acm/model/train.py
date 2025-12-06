import numpy as np
from pathlib import Path
import logging

import torch
from pytorch_lightning import seed_everything
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule

from acm.utils.logging import setup_logging
setup_logging()
logger = logging.getLogger('ACM trainer')

def TrainFCN(
    # Data from DataObservable
    lhc_y: np.ndarray,
    lhc_x: np.ndarray,
    lhc_x_names: list,
    covariance_matrix: np.ndarray,
    stat_name: str,
    model_dir: str,
    n_test: int|list,
    # Hyperparameters
    learning_rate: float,
    n_hidden: list,
    dropout_rate: float,
    weight_decay: float,
    act_fn: str = 'learned_sigmoid',
    loss: str = 'mae',
    # Training
    max_epochs: int = 5000,
    log_dir: str = None,
    seed: int = None,
    # Data transforms
    transform = None, 
    )-> float:
    """
    Train a Fully Connected Neural Network (FCN) emulator for the given statistic, with the given hyperparameters.
    This function expects the LHC data and the covariance matrix to be in the same format as the one used in the ACM pipeline.

    Parameters
    ----------
    lhc_y : np.ndarray
        LHC y data for the statistic to train on (outputs)
    lhc_x : np.ndarray
        LHC x data for the statistic to train on (inputs).
    lhc_x_names : list
        List of the names of the input parameters.
    covariance_matrix : np.ndarray
        Covariance matrix for the statistic to train on.
    stat_name : str
        Statistic to train on.
    model_dir : str, optional
        Directory to save the model. 
    n_test : int|list
        Number of training samples to select from the LHC data. Must be smaller than the total number of samples.
        If a list is provided, those indexes are used to select the test samples (excluded from the training set).
        If an integer is provided, the first n_test samples are used for testing.
        Set to 0 to use all the samples for training.
    learning_rate : float
        Learning rate for the optimizer.
    n_hidden : list
        List of integers, number of neurons in each hidden layer.
    dropout_rate : float
        Dropout rate for the hidden layers.
    weight_decay : float
        Weight decay for the optimizer.
    act_fn : str, optional
        Activation function for the hidden layers. Defaults to 'learned_sigmoid'.
    loss : str, optional
        Loss function to use. Defaults to 'mae'.
    max_epochs : int, optional
        Maximum number of epochs to train the model. Defaults to 5000.
    log_dir : str, optional
        Directory to save the pytorch lightning logs.
        If set to None, the logs are saved in the current directory. Defaults to None.
    transform : callable, optional
        Transform to apply to the output features, from the `sunbird.data.transforms` or `sunbird.data.transforms_array` modules. Defaults to None.

    Returns
    -------
    float
        Validation loss of the model.
    """
    logger.info(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    logger.info(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    if transform: 
        logger.info(f'Applying transform: {type(transform).__name__}')
        try: # Handle sunbird.data.transforms
            lhc_y = transform.fit_transform(lhc_y)
        except: # Handle sunbird.data.transforms_array
            lhc_y = transform.transform(lhc_y) 
    
    # Set the first n_test samples to the testing set 
    n_tot = len(lhc_y) # Total number of data points
    if isinstance(n_test, int):
        idx_train = list(range(n_test, n_tot))
    elif isinstance(n_test, list):
        idx_train = list(set(range(n_tot)) - set(n_test))
    
    if len(idx_train) > n_tot:
        raise ValueError(f'Number of training samples ({n_test=}) is larger than the total number of samples ({n_tot=})')

    logger.info(f'Using {len(idx_train)} last samples for training')

    lhc_train_x = lhc_x[idx_train]
    lhc_train_y = lhc_y[idx_train]

    train_mean = np.mean(lhc_train_y, axis=0)
    train_std = np.std(lhc_train_y, axis=0)

    train_mean_x = np.mean(lhc_train_x, axis=0)
    train_std_x = np.std(lhc_train_x, axis=0)

    data = ArrayDataModule(
        x=torch.Tensor(lhc_train_x),
        y=torch.Tensor(lhc_train_y), 
        val_fraction=0.2, # NOTE : Hardcoded values here : Ok ?
        batch_size=128,
        num_workers=0)
    data.setup()
    
    if seed is not None:
        logger.info(f'Setting seed to {seed}')
        seed_everything(seed, workers=True)

    model = FCN(
            n_input=data.n_input,
            n_output=data.n_output,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate, 
            learning_rate=learning_rate,
            scheduler_patience=10, # NOTE : Hardcoded values here : Ok ?
            scheduler_factor=0.5,
            scheduler_threshold=1.e-6,
            weight_decay=weight_decay,
            act_fn=act_fn,
            loss=loss,
            training=True,
            mean_output=train_mean,
            std_output=train_std,
            mean_input=train_mean_x,
            std_input=train_std_x,
            transform_output=transform,
            standarize_output=True,
            coordinates=lhc_x_names,
            covariance_matrix=covariance_matrix,
        )
    
    if model_dir is not None: # To avoid some errors with Path
        model_dir = Path(model_dir) / f'{stat_name}/'
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    val_loss, model, early_stop_callback = train.fit(
        data=data, model=model,
        model_dir=model_dir,
        max_epochs=max_epochs,
        log_dir=log_dir,
        devices=1,
    )
    
    return val_loss


# NOTE : toy example to test the function
if __name__ == '__main__':
    
    from sunbird.data.transforms import Log
    transform = Log()
    
    # Set the paths
    from acm.observables.emc import GalaxyCorrelationFunctionMultipoles
    tpcf = GalaxyCorrelationFunctionMultipoles(
        # No filters for now
    )
    stat_name = tpcf.stat_name
    lhc_y = tpcf.y
    lhc_x = tpcf.x
    lhc_x_names = tpcf.x_names
    covariance_matrix = tpcf.get_covariance_matrix()
    model_dir = tpcf.paths['model_dir']
    
    logger.info(f'Training {stat_name}')
    
    # Training parameters
    n_test = 600 # 6 first cosmologies
    
    # Hyperparameters
    learning_rate = 1.0e-3
    n_hidden = [512, 512, 512, 512]
    dropout_rate = 0
    weight_decay = 0
    
    val_loss = TrainFCN(
        lhc_y = lhc_y,
        lhc_x = lhc_x,
        lhc_x_names = lhc_x_names,
        covariance_matrix = covariance_matrix,
        stat_name = stat_name,
        model_dir = model_dir,
        n_test = n_test,
        learning_rate = learning_rate,
        n_hidden = n_hidden,
        dropout_rate = dropout_rate,
        weight_decay = weight_decay,
        transform=transform,
    )
    
    logger.info(f'Best validation loss for {stat_name}: {val_loss}')