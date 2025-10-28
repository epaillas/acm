import argparse
import importlib
import logging

from acm.utils.logging import setup_logging
setup_logging()
logger = logging.getLogger(__file__.split('/')[-1])

mapping = {
    'density': 'TODO', # TODO: Add correct mapping
    'tpcf': 'GalaxyCorrelationFunctionMultipoles',
    'wp': 'GalaxyProjectedCorrelationFunction',
    'dsc_conf': 'DensitySplitCorrelationFunctionMultipoles',
}

def get_class_from_module(module_path, class_name):
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default='acm.observables.bgs', help='Base module path for observables')
    parser.add_argument('--measurements', type=str, nargs='+', help='List of measurements to process')
    parser.add_argument('--output', type=str, default='/pscratch/sd/s/sbouchar/acm/bgs/input_data/', help='Output directory for compressed files') # TODO: Change default
    args = parser.parse_args()
    
    if 'density' in args.measurements:
        cls = get_class_from_module(args.module, mapping['density'])
        instance = cls()
        # TODO : Implement density class
        
    if 'tpcf' in args.measurements:
        cls = get_class_from_module(args.module, mapping['tpcf'])
        instance = cls()
        instance.compress_data(add_covariance=True, save_to=args.output)
    
    if 'wp' in args.measurements:
        cls = get_class_from_module(args.module, mapping['wp'])
        instance = cls()
        instance.compress_data(add_covariance=True, save_to=args.output)
        
    if 'dsc_conf' in args.measurements:
        cls = get_class_from_module(args.module, mapping['dsc_conf'])
        instance = cls()
        instance.compress_data(add_covariance=True, save_to=args.output)
    
    if any(m not in mapping for m in args.measurements):
        unknown = [m for m in args.measurements if m not in mapping]
        logger.error(f"Unknown measurements: {unknown}. Available options are: {list(mapping.keys())}")