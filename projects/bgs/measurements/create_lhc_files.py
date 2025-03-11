from acm.projects.bgs import *

import logging


if __name__ == '__main__':
    
    stat_dict = {
        'tpcf': {
            'class': GalaxyCorrelationFunctionMultipoles,
        },
        'dsc_conf': {
            'class': DensitySplitCorrelationFunctionMultipoles,
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
            logger.info(f'No LHC creation method for {obs.stat_name}')
        
        # Create the error file
        n_test = 6 * obs.summary_coords_dict['hod_number'] # 6 cosmologies w/ all HODs
        obs.create_emulator_error(n_test=n_test, save_to=obs.paths['error_dir'])

        logger.info(f'Created LHC with shape: {obs.lhc_x.shape}, {obs.lhc_y.shape}')
        logger.info(f'Created covariance with shape: {obs.covariance_y.shape}')
        logger.info(f'Created error with shape: {obs.emulator_error.shape}')