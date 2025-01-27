import numpy as np
from pathlib import Path
import logging

import torch
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule

from acm.data.io_tools import read_lhc, read_covariance
from acm.data.default import summary_coords_dict

from acm.utils import setup_logging
setup_logging()
logger = logging.getLogger('ACM trainer')

def TrainFCN(
    # Data
    statistic: str,
    lhc_dir: str,
    covariance_dir: str,
    model_dir: str,
    n_test: int,
    # Hyperparameters
    learning_rate: float,
    n_hidden: list,
    dropout_rate: float,
    weight_decay: float,
    act_fn: str = 'learned_sigmoid',
    loss: str = 'mae',
    # Training
    final_model: bool = False,
    max_epochs: int = 5000,
    logger_dir: str = None,
    # Data transforms
    transform = None, 
    select_filters: dict = None,
    slice_filters: dict = None,
    summary_coords_dict: dict = summary_coords_dict,
    )-> float:
    """
    Train a Fully Connected Neural Network (FCN) emulator for the given statistic, with the given hyperparameters.
    This function expects the LHC data and the covariance matrix to be in the same format as the one used in the ACM pipeline.

    Parameters
    ----------
    statistic : str
        Statistic to train on.
    lhc_dir : str
        Directory containing the LHC data.
    covariance_dir : str
        Directory containing the covariance matrix.
    model_dir : str, optional
        Directory to save the model. 
    n_test : int
        Number of training samples to select from the LHC data. Must be smaller than the total number of samples.
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
    final_model : bool, optional
        Whether to train the final model or not. If False, the first six cosmologies are used for testing, otherwise all the data is used for training.
        Defaults to False
    max_epochs : int, optional
        Maximum number of epochs to train the model. Defaults to 5000.
    logger_dir : str, optional
        Directory to save the pytorch lightning logs.
        If set to None, the logs are saved in the current directory. Defaults to None.
    transform : callable, optional
        Transform to apply to the output features, from the `sunbird.data.transforms` or `sunbird.data.transforms_array` modules. Defaults to None.
    select_filters : dict, optional
        Filters to select values in coordinates. Defaults to None.
    slice_filters : dict, optional
        Filters to slice values in coordinates. Defaults to None.
    summary_coords_dict : dict, optional
        Dictionary containing the summary coordinates for each statistic. 
        It also contains the comology indexes, the number of HODs, parameters and phases. 
        Defaults to summary_coords_dict from `acm.data.default`.

    Returns
    -------
    float
        Validation loss of the model.
        
    Example
    -------
    ```python
    slice_filters = {'bin_values': (0, 0.5),} 
    select_filters = {'multipoles': [0, 2],}
    ```
    will return the summary statistics for `0 < bin_values < 0.5` and multipoles 0 and 2
    """
    lhc_x, lhc_y, coords = read_lhc(statistics=[statistic],
                                    data_dir=lhc_dir,
                                    select_filters=select_filters,
                                    slice_filters=slice_filters,
                                    summary_coords_dict=summary_coords_dict,
                                    ) 
    logger.info(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    covariance_matrix, n_sim = read_covariance(statistics=[statistic],
                                               data_dir=covariance_dir,
                                               select_filters=select_filters,
                                               slice_filters=slice_filters,
                                               summary_coords_dict=summary_coords_dict,
                                               )
    logger.info(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    if transform: 
        logger.info(f'Applying transform: {type(transform).__name__}')
        try: # Handle sunbird.data.transforms
            lhc_y = transform.fit_transform(lhc_y)
        except: # Handle sunbird.data.transforms_array
            lhc_y = transform.transform(lhc_y) 
    
    n_tot = len(lhc_y) # Total number of data points
    if n_test > n_tot:
        raise ValueError(f'Number of training samples ({n_test=}) is larger than the total number of samples ({n_tot=})')

    # Set the first n_test samples to the testing set 
    if final_model:
        idx_train = list(range(n_tot))
    else:
        idx_train = list(range(n_test, n_tot))

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
            coordinates=coords,
            covariance_matrix=covariance_matrix,
        )
    
    if model_dir is not None: # To avoid some errors with Path
        model_dir = Path(model_dir) / f'{statistic}/'
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    val_loss, model, early_stop_callback = train.fit(
        data=data, model=model,
        model_dir=model_dir,
        max_epochs=max_epochs,
        logger_dir=logger_dir,
        devices=1,
    )
    
    return val_loss


# NOTE : toy example to test the function
if __name__ == '__main__':
    statistic = 'tpcf'
    logger.info(f'Training {statistic}')
    
    from sunbird.data.transforms import Log
    transform = Log()
    
    # Set the paths
    from acm.data.paths import emc_paths
    lhc_dir = emc_paths['lhc_dir']
    covariance_dir = emc_paths['covariance_dir']
    model_dir = emc_paths['model_dir']
    
    # Training parameters
    n_test = 600 # 6 first cosmologies
    
    # Hyperparameters
    learning_rate = 1.0e-3
    n_hidden = [512, 512, 512, 512]
    dropout_rate = 0
    weight_decay = 0
    
    val_loss = TrainFCN(
        statistic = statistic,
        lhc_dir = lhc_dir,
        covariance_dir = covariance_dir,
        model_dir = model_dir,
        n_test = n_test,
        learning_rate = learning_rate,
        n_hidden = n_hidden,
        dropout_rate = dropout_rate,
        weight_decay = weight_decay,
        transform=transform,
    )
    
    logger.info(f'Best validation loss for {statistic}: {val_loss}')