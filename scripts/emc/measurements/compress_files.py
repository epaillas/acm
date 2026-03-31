import argparse

from acm import setup_logging
from acm.utils.modules import get_class_from_module
from acm.utils.paths import lookup_registry_path


ALIASES = [
    "bispectrum",
    "dd_knn",
    "ds_xiqg",
    "ds_xiqq",
    "minkowski",
    "pdf",
    "projected_tpcf",
    "recon_spectrum",
    "spectrum",
    "sv_recon_vsf",
    "sv_recon_xivg",
    "sv_recon_xivv",
    "sv_vsf",
    "sv_xivg",
    "sv_xivv",
    "tpcf",
    "vide_ccf",
    "vide_vsf",
    "wst",
]

LEGACY_TO_ALIAS = {
    "GalaxyBispectrumMultipoles": "bispectrum",
    "DDkNN": "dd_knn",
    "DensitySplitQuantileGalaxyCorrelationFunctionMultipoles": "ds_xiqg",
    "DensitySplitQuantileCorrelationFunctionMultipoles": "ds_xiqq",
    "MinkowskiFunctionals": "minkowski",
    "GalaxyOverdensityPDF": "pdf",
    "ProjectedGalaxyCorrelationFunction": "projected_tpcf",
    "ReconstructedGalaxyPowerSpectrumMultipoles": "recon_spectrum",
    "GalaxyPowerSpectrumMultipoles": "spectrum",
    "VERSUSVoidSizeFunction": "sv_vsf",
    "ReconstructedVERSUSVoidSizeFunction": "sv_recon_vsf",
    "VERSUSVoidGalaxyCorrelationFunctionMultipoles": "sv_xivg",
    "ReconstructedVERSUSVoidGalaxyCorrelationFunctionMultipoles": "sv_recon_xivg",
    "VERSUSVoidAutoCorrelationFunctionMultipoles": "sv_xivv",
    "ReconstructedVERSUSVoidAutoCorrelationFunctionMultipoles": "sv_recon_xivv",
    "GalaxyCorrelationFunctionMultipoles": "tpcf",
    "VIDEVoidGalaxyCorrelationFunctionMultipoles": "vide_ccf",
    "VIDEVoidSizeFunction": "vide_vsf",
    "WaveletScatteringTransform": "wst",
}


def resolve_statistic_name(statistic: str) -> str:
    statistic = LEGACY_TO_ALIAS.get(statistic, statistic)
    if statistic not in ALIASES:
        valid = ", ".join(ALIASES)
        raise ValueError(
            f"Unknown EMC observable alias '{statistic}'. Expected one of: {valid}"
        )
    return statistic


parser = argparse.ArgumentParser(description='Compress EMC measurement files.')
parser.add_argument(
    '--module',
    type=str,
    default='acm.observables.emc',
    help='Module to load the observable classes from.',
)
parser.add_argument(
    '-s',
    '--statistic',
    type=str,
    choices=ALIASES,
    help='Observable alias to compress.',
    default='spectrum',
)
parser.add_argument('--n_hod', type=int, default=500, help='Number of HOD realizations to use for compression.')
parser.add_argument('--add_covariance', action='store_true', help='Whether to add covariance to the compressed data.')

args = parser.parse_args()
statistic = resolve_statistic_name(args.statistic)
module = args.module
n_hod = args.n_hod
add_covariance = args.add_covariance

setup_logging()

paths = lookup_registry_path('projects.yaml', 'emc')

observable = get_class_from_module(module, statistic)
observable.compress_data(paths=paths, save_to=paths['data_dir'], add_covariance=add_covariance, n_hod=n_hod)
