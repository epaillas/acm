import os
import shutil
import numpy as np
from pathlib import Path
from acm.projects.emc_new import *

import logging

# %% Old fom new

def lhc_from_old(filename: str, statistic_name: str, separation_name: str = None, save_to: str = None):
    logger = logging.getLogger(f'{statistic_name}_lhc')
    
    data = np.load(filename, allow_pickle=True).item()
    logger.info(f'old lhc keys : {data.keys()}')
    
    if separation_name is None:
        old_keys = list(data.keys())
        new_keys = ['bin_values', 'lhc_x_names', 'lhc_x', 'lhc_y', 'cov_y']
        separation_name = [key for key in old_keys if key not in new_keys]
        if len(separation_name) == 1: # Just in case
            separation_name = separation_name[0]
            logger.info(f'Assuming separation_name: {separation_name}')
    
    bin_values = data[separation_name]
    lhc_x_names = data['lhc_x_names']
    lhc_x = data['lhc_x']
    lhc_y = data['lhc_y']
    cov_y = data['cov_y']
    
    # Save the new file
    cout = {
        'bin_values': bin_values,
        'lhc_x_names': lhc_x_names,
        'lhc_x': lhc_x,
        'lhc_y': lhc_y,
        'cov_y': cov_y,
    }
    
    if save_to:
        new_path = save_to + f'{statistic_name}_lhc.npy'
        np.save(new_path, cout)
        logger.info(f'File saved at {new_path}')
    
    return cout
    

if __name__ == '__main__':

    stat_dict = {
        'tpcf': {
            'class': GalaxyCorrelationFunctionMultipoles,
            'model_fn': '/pscratch/sd/e/epaillas/emc/v1.1/trained_models/tpcf/cosmo+hod/optuna_log/epoch=298-step=14950.ckpt',
        },
        'dsc_conf': {
            'class': DensitySplitCorrelationFunctionMultipoles,
            'model_fn': '/pscratch/sd/e/epaillas/emc/trained_models/dsc_conf/cosmo+hod/aug9/last-v1.ckpt',
        },
        'bispectrum': {
            'class': GalaxyBispectrumMultipoles,
            'model_fn': '/pscratch/sd/e/epaillas/emc/v1.1/trained_models/GalaxyBispectrumMultipoles/cosmo+hod/optuna/last.ckpt',
        },
        'pk': {
            'class': GalaxyPowerSpectrumMultipoles,
            'model_fn': '/pscratch/sd/e/epaillas/emc/trained_models/pk/cosmo+hod/optuna/last-v31.ckpt',
        },
        'mst': {
            'class': MinimumSpanningTree,
            'model_fn': '/pscratch/sd/e/epaillas/emc/trained_models/mst/cosmo+hod/optuna/last-v8.ckpt',
        },
    }
    
    for stat in stat_dict.keys():
        # logging
        logger = logging.getLogger(f'{stat}_lhc')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(name)-28s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.info(f'Creating LHC for {stat}')
        obs = stat_dict[stat]['class']()
        
        # Check the stat name just in case
        assert stat == obs.stat_name, f'Error: {stat} != {obs.stat_name}'
        
        # Create the lhc file
        try:
            obs.create_lhc(save_to=obs.paths['lhc_dir'])
        except:
            logger.info(f'No LHC creation method for {obs.stat_name}, using old lhc instead')
            old_filename = f'/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/{obs.stat_name}.npy'
            lhc_from_old(filename=old_filename, statistic_name=obs.stat_name, save_to=obs.paths['lhc_dir'])
        
        # Copy the best model
        model_fn = stat_dict[obs.stat_name]['model_fn']
        model_fn = Path(model_fn).resolve()
        copy_to = obs.paths['model_dir'] + f'{obs.stat_name}/' # ACM standard storage (see train and io_tools)
        Path(copy_to).mkdir(parents=True, exist_ok=True) # Check if the directory exists, if not create it
        model_fn = shutil.copy(model_fn, copy_to) # Copy the model to the desired path

        # Create the symlink to the emulator
        symlink = Path(copy_to) / 'last.ckpt'
        symlink.unlink(missing_ok=True) # Remove the symlink if it already exists
        os.symlink(model_fn, symlink)
        
        # Create the error file
        n_test = 6 * obs.summary_coords_dict['hod_number'] # 6 cosmologies w/ all HODs
        obs.create_emulator_error(n_test=n_test, save_to=obs.paths['error_dir'])

        logger.info(f'Created LHC with shape: {obs.lhc_x.shape}, {obs.lhc_y.shape}')
        logger.info(f'Created covariance with shape: {obs.covariance_y.shape}')
        logger.info(f'Created error with shape: {obs.emulator_error.shape}')
