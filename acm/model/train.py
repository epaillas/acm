import torch
import wandb
import logging
import numpy as np
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from sunbird.emulators import FCN
from sunbird.data import ArrayDataModule

class FCNTrainer(Trainer):
    """
    Trainer class for Fully Connected Neural Network (FCN) models using PyTorch Lightning.
    """
    def __init__(self, callbacks: list = None, logger: str = None, log_dir: str = None, **kwargs):
        """
        Initialize the FCNTrainer with specified callbacks and logger.
        
        Parameters
        ----------
        callbacks : list | None
            List of callbacks to use during training. If None, default callbacks will be set up.
        logger : str | None
            Type of logger to use ('wandb', 'tensorboard', or None).
        log_dir : str | None
            Directory to save logs.
        **kwargs
            Additional keyword arguments for the Trainer.
        """
        
        # Set up default callbacks if none are provided
        patience = kwargs.pop('early_stop_patience', 30)
        min_delta = kwargs.pop('early_stop_threshold', 1.e-7)
        checkpoint_filename = kwargs.pop('checkpoint_filename', '{epoch:02d}-{step}-{val_loss:.5f}')
        checkpoint_dir = kwargs.pop('checkpoint_dir', None)
        if callbacks is None:
            early_stop_callback = self.early_stop_callback(
                monitor="val_loss", 
                patience=patience, 
                min_delta=min_delta,
            )
            checkpoint_callback = self.checkpoint_callback(
                monitor='val_loss',
                checkpoint_filename=checkpoint_filename,
                checkpoint_dir=checkpoint_dir,
            )
            lr_monitor = LearningRateMonitor(logging_interval='step')
            progress_bar = RichProgressBar()
            
            callbacks = [
                early_stop_callback,
                checkpoint_callback,
                lr_monitor,
                progress_bar,
            ]
            callbacks = [cb for cb in callbacks if cb is not None] # Remove None callbacks
            
        logger = self.get_logger(logger=logger, log_dir=log_dir)
        
        gradient_clip_val = kwargs.pop('gradient_clip_val', 0.5)
        log_every_n_steps = kwargs.pop('log_every_n_steps', 1)
        super().__init__(
            callbacks = callbacks,
            logger = logger,
            gradient_clip_val = gradient_clip_val,
            log_every_n_steps = log_every_n_steps,
            **kwargs
        )
        
    def fit(self, *args, **kwargs):
        """
        Fit the model using the Trainer's fit method and return the best validation loss.
        
        Parameters
        ----------
        *args
            Positional arguments to pass to the Trainer's fit method.
        **kwargs
            Keyword arguments to pass to the Trainer's fit method.
        
        Returns
        -------
        best_val_loss : float
            Best validation loss achieved during training.
        """
        super().fit(*args, **kwargs)
        
        # Retrieve the best validation loss from the EarlyStopping callback
        best_val_loss = None
        for cb in self.callbacks:
            if isinstance(cb, EarlyStopping):
                best_val_loss = cb.best_score.item()
                break
        if best_val_loss is None:
            best_val_loss = self.callback_metrics.get('val_loss', None)
        
        return best_val_loss
    
    @staticmethod
    def get_logger(logger: str = None, log_dir: str = None):
        """
        Get the logger instance based on the specified type.
        
        Parameters
        ----------
        logger : str | None
            Logger type. Can be 'wandb', 'tensorboard', or None.
        log_dir : str | None
            Directory to save logs. Required if logger is not None.
        
        Returns
        -------
        logger : Logger | None
            Configured logger instance or None.
        """
        if logger == 'wandb':
            wandb.init()
            logger = WandbLogger(log_model="all", project="sunbird",)
        elif logger == 'tensorboard':
            logger = TensorBoardLogger(log_dir, name="optuna")
        elif logger is None:
            logger = None
        return logger
    
    @staticmethod
    def early_stop_callback(
        monitor: str = "val_loss", 
        patience: int = 30, 
        min_delta: float = 1.e-7,
    ) -> EarlyStopping:
        """
        Sets up an EarlyStopping callback for the trainer.

        Parameters
        ----------
        monitor : str
            Metric to monitor for early stopping.
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        min_delta : float
            Minimum change in the monitored metric to qualify as an improvement.

        Returns
        -------
        EarlyStopping
            Configured EarlyStopping callback.
        """
        esc = EarlyStopping(
            monitor = monitor, 
            patience = patience, 
            min_delta = min_delta,
            mode = "min", 
            verbose = True, 
            check_on_train_epoch_end = True,
        )
        return esc
    
    @staticmethod
    def checkpoint_callback(
        monitor: str = "val_loss",
        checkpoint_filename: str = '{epoch:02d}-{step}-{val_loss:.5f}',
        checkpoint_dir: str = None,
    ) -> ModelCheckpoint:
        """
        Sets up a ModelCheckpoint callback for the trainer.
        
        Parameters
        ----------
        monitor : str
            Metric to monitor for saving checkpoints.
        checkpoint_filename : str
            Filename template for the saved checkpoints.
        checkpoint_dir : str | None
            Directory to save the checkpoints. If None, no checkpoints are saved.
        
        Returns
        -------
        ModelCheckpoint | None
            Configured ModelCheckpoint callback or None if checkpoint_dir is None.
        """
        if checkpoint_dir is None: 
            return None
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        mcp = ModelCheckpoint(
            monitor = monitor,
            dirpath = checkpoint_dir,
            filename = checkpoint_filename,
            auto_insert_metric_name = True,
            save_last = 'link',
            mode = 'min',
        )
        return mcp


def TrainFCN(
    lhc_x: np.ndarray,
    lhc_y: np.ndarray,
    lhc_x_names: list,
    n_hidden: list,
    dropout_rate: float,
    learning_rate: float,
    weight_decay: float,
    act_fn: str = 'learned_sigmoid',
    loss: str = 'mae',
    transform = None,
    val_fraction: float = 0.1,
    checkpoint_dir: str = None,
    checkpoint_filename: str = '{epoch:02d}-{step}-{val_loss:.5f}',
    train_logger: str = None,
    log_dir: str = None,
    return_trainer: bool = False,
    **kwargs,
) -> tuple[float, FCNTrainer]:
    """
    Train a Fully Connected Neural Network (FCN) model with the given hyperparameters
    and return the validation loss.
    
    Parameters
    ----------
    lhc_x : np.ndarray
        Input features for training.
    lhc_y : np.ndarray
        Target values for training.
    lhc_x_names : list
        Names of the input features.
    n_hidden : list
        List specifying the number of hidden units in each layer.
    dropout_rate : float
        Dropout rate for regularization.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for the optimizer.
    act_fn : str
        Activation function to use in the model.
    loss : str
        Loss function to use for training.
    transform : callable | None
        Data transform to apply to the target values.
    val_fraction : float
        Fraction of data to use for validation.
    max_epochs : int
        Maximum number of epochs to train the model.
    checkpoint_dir : str | None
        Directory to save model checkpoints.
    checkpoint_filename : str
        Filename template for saving checkpoints.
    train_logger : str | None
        Type of logger to use ('wandb', 'tensorboard', or None).
    log_dir : str | None
        Directory to save logs.
    return_trainer: bool
        Whether to return the trainer instance along with the validation loss.
    **kwargs
        Additional keyword arguments for the FCNTrainer.
    
    Returns
    -------
    val_loss : float
        Validation loss after training.
    trainer : FCNTrainer, optional
        The trained FCNTrainer instance (returned if return_trainer is True).
    """
    logger = logging.getLogger('TrainFCN')
    logger.info(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    if transform: 
        logger.info(f'Applying transform: {type(transform).__name__}')
        try: # Handle sunbird.data.transforms
            lhc_y = transform.fit_transform(lhc_y)
        except: # Handle sunbird.data.transforms_array
            lhc_y = transform.transform(lhc_y) 

    train_mean = np.mean(lhc_y, axis=0)
    train_std = np.std(lhc_y, axis=0)

    train_mean_x = np.mean(lhc_x, axis=0)
    train_std_x = np.std(lhc_x, axis=0)

    data = ArrayDataModule(
        x=torch.Tensor(lhc_x),
        y=torch.Tensor(lhc_y), 
        val_fraction=val_fraction, 
        batch_size=128, # NOTE : Hardcoded values here : Ok ?
        num_workers=0,
    )
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
        mean_input=train_mean_x,
        std_input=train_std_x,
        mean_output=train_mean,
        std_output=train_std,
        standarize_input=True,
        standarize_output=True,
        transform_input=None,
        transform_output=transform,
        coordinates=lhc_x_names,
    )
    
    trainer = FCNTrainer(
        logger = train_logger,
        log_dir = log_dir,
        checkpoint_dir = checkpoint_dir,
        checkpoint_filename = checkpoint_filename,
        **kwargs,
    )
    
    val_loss = trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    
    if return_trainer:
        return val_loss, trainer
    return val_loss


#%% Example usage
if __name__ == "__main__":
    from acm.utils.logging import setup_logging
    setup_logging(level='info')
    
    from lightning import seed_everything
    seed_everything(42, workers=True)
    
    # Dummy data for testing
    lhc_x = np.random.rand(1000, 10)
    lhc_y = np.random.rand(1000, 5)
    lhc_x_names = [f'feature_{i}' for i in range(10)]
    
    val_loss, trainer = TrainFCN(
        lhc_x=lhc_x,
        lhc_y=lhc_y,
        lhc_x_names=lhc_x_names,
        n_hidden=[64, 64],
        dropout_rate=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        max_epochs=10,
        deterministic=True,
        devices=1,
        return_trainer=True,
        checkpoint_dir='./checkpoints',
        checkpoint_filename='test-{epoch:02d}-{val_loss:.2f}.ckpt',
    )
    print(f'Validation Loss: {val_loss}')