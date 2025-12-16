"""
Compress measurement files for BGS observables.

Usage:
    python compress_files.py --measurements tpcf ds_xigg ds_xiqg --output /pscratch/sd/s/sbouchar/acm/bgs/input_data/ --log_level INFO    
"""
import logging
import argparse
from acm.utils.modules import get_class_from_module
from acm.utils.logging import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default='acm.observables.bgs', help='Base module path for observables')
    parser.add_argument('--measurements', type=str, nargs='+', help='List of measurements to process')
    parser.add_argument('--output', type=str, default='/pscratch/sd/s/sbouchar/acm/bgs/input_data/', help='Output directory for compressed files') # TODO: Change default
    parser.add_argument('--add_covariance', action='store_true', help='Whether to add covariance to the compressed files')
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__file__.split('/')[-1])

    for stat_name in args.measurements:
        try:
            cls = get_class_from_module(args.module, stat_name)
        except ImportError as e:
            logger.error(f"Could not import class for measurement '{stat_name}': {e}")
            continue
        instance = cls()
        instance.compress_data(add_covariance=args.add_covariance, save_to=args.output)