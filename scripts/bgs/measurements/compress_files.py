"""
Compress measurement files for BGS observables.

Usage:
    python compress_files.py \
    -md /pscratch/sd/s/sbouchar/acm/bgs-20/measurements \
    -pd /pscratch/sd/s/sbouchar/acm/bgs-20/parameters/cosmo+hod_params \
    --measurements tpcf ds_xiqq ds_xiqg \
    -tf "{'cosmo_idx': [0, 1, 2, 3, 4, 13]}" \
    --output /pscratch/sd/s/sbouchar/acm/bgs-20/input_data/ \
    --add_covariance \
    --compress_kwargs "{'hod_idx': 157}" \
    --log_level info
"""
import logging
import argparse
from acm.utils.modules import get_class_from_module
from acm.utils.logging import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default='acm.observables.bgs', help='Base module path for observables')
    parser.add_argument('--measurements_dir', '-md', type=str, default=None, required=True, help='Directory containing measurement files')
    parser.add_argument('--param_dir', '-pd', type=str, default=None, required=True, help='Directory containing HOD parameter files')
    parser.add_argument('--test_filters', '-tf', type=str, default=None, help='Expression evaluated to define the test set (e.g., "{\'cosmo_idx\': 0}")')
    parser.add_argument('--measurements', type=str, nargs='+', help='List of measurements to process')
    parser.add_argument('--output', type=str, default=None, help='Output directory for compressed files')
    parser.add_argument('--add_covariance', action='store_true', help='Whether to add covariance to the compressed files')
    parser.add_argument('--log_level', type=str, default='warning', help='Set logging level (e.g., DEBUG, INFO)')
    parser.add_argument('--compress_kwargs', '-ck', type=str, default=None, help='Additional keyword arguments for compression, evaluated as a dictionary (e.g., "{\'key\': value}")')
    args = parser.parse_args()
    
    logger = logging.getLogger(__file__.split('/')[-1])
    setup_logging(level=args.log_level)
    
    paths = dict(
        measurements_dir = args.measurements_dir,
        param_dir = args.param_dir
    )
    
    compress_kwargs = {}
    if args.compress_kwargs:
        compress_kwargs = eval(args.compress_kwargs)
        if not isinstance(compress_kwargs, dict):
            logger.error("compress_kwargs should be a dictionary. Please check the input.")
            compress_kwargs = {}
        else:
            logger.info(f"Additional compression kwargs: {compress_kwargs}")
            compress_kwargs.update(compress_kwargs)
    
    test_filters = None
    if args.test_filters:
        test_filters = eval(args.test_filters)

    for stat_name in args.measurements:
        try:
            cls = get_class_from_module(args.module, stat_name)
        except ImportError as e:
            logger.error(f"Could not import class for measurement '{stat_name}': {e}")
            continue
        cls.compress_data(paths=paths, add_covariance=args.add_covariance, save_to=args.output, test_filters=test_filters, **compress_kwargs)