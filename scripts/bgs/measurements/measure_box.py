"""
Script to measure clustering statistics from HOD catalogs generated with AbacusHOD.
The script allows to compute various statistics such as the two-point correlation function (2PCF) and density split statistics.
It supports loading HOD catalogs from a specified directory, applying redshift space distortions (RSD) and Alcock-Paczynski (AP) effects, and saving the results to disk
for different cosmologies, phases, and seeds.

Usage:
    python measure_box.py -h

Directories and files:
- HOD catalogs should be stored in a directory structure like:
  root/cXXX_phYYY/seedZ/hodAAA/galaxies.fits
  where XXX is the cosmology index, YYY is the phase index, Z is the seed, and AAA is the HOD index.
- Results will be saved in a similar structure under their respective name and formats.
"""
import sys
import yaml
import fitsio
import logging
import argparse
import numpy as np
from pathlib import Path
from pycorr import TwoPointCorrelationFunction
from pypower import CatalogFFTPower

from acm.hod import BoxHOD
from acm.utils.logging import setup_logging
from acm.utils.paths import get_Abacus_dirs
from acm.estimators.galaxy_clustering.density_split import DensitySplit

#%% Box loading functions
def get_save_fn(
    save_dir: str|Path,
    measurement: str,
    los: str = None,
    extension: str = 'npy',
    mkdir: bool = True,
    exist_ok: bool = True,
) -> Path:
    """
    Get the filename to save the measurement, with the 
    `save_dir/measurement_los_Y.ext` format.
    If the file already exists and exist_ok is False, returns None.

    Parameters
    ----------
    save_dir : str or Path
        The base directory to save the measurements.
    hod_idx : int
        The HOD index.
    measurement : str
        The type of measurement (e.g., 'tpcf', 'density_split').
    los : str, optional
        The line-of-sight direction. If None, no los is added to the filename. Defaults to None.
    extension : str, optional
        The file extension. Defaults to 'npy'.
    mkdir : bool, optional
        Whether to create the directory if it does not exist. Defaults to True.
    exist_ok : bool, optional
        If False and the file already exists, returns None. Defaults to True.

    Returns
    -------
    Path
        The full path to the file where the measurement will be saved.
    """
    extension = extension.lstrip('.') # Remove leading dot if present just in case
    fn = measurement
    if los is not None:
        fn += f'_los_{los}'
    fn += f'.{extension}'
    save_fn = Path(save_dir) / fn
    if mkdir:
        save_fn.parent.mkdir(parents=True, exist_ok=True)
    if not exist_ok and save_fn.exists():
        return None
    return save_fn

#%% Statistics computation functions
def compute_number_density(
    catalog: dict,
    boxsize: float,
    save_fn: Path = None,
) -> float:
    """
    Compute the number density of the galaxies in the catalog.

    Parameters
    ----------
    catalog : dict
        The HOD catalog.
    boxsize : float or array-like
        The size of the simulation box. 
        If a float is provided, the box is assumed to be cubic. 
        If an array-like of shape (3,) is provided, it is assumed to be the box size along each axis.
    save_fn : Path, optional
        The filename to save the density value as a numpy array. If None, the density is not saved. Defaults to None.

    Returns
    -------
    float
        The number density of the galaxies in h^3 Mpc^-3.
    """
    if isinstance(catalog, dict):
        # Sanity check on catalog lengths
        keys = list(catalog.keys())
        values = list(catalog.values())
        lengths = [len(val) for val in values]
        if len(set(lengths)) != 1:
            logger.warning(f"Catalog columns have different lengths. Assuming the {keys[0]} column length for density computation.")
        n_galaxies = lengths[0]
    else:
        n_galaxies = len(catalog)
        
    # Sanity check on n_galaxies
    if n_galaxies < 20:
        logger.warning(f"Number of galaxies is very low ({n_galaxies}). Are you sure you passed the correct catalog?")
        
    if isinstance(boxsize, (list, tuple, np.ndarray)):
        volume = np.prod(boxsize)
    else:
        volume = boxsize**3

    density = n_galaxies / volume
    if save_fn is not None:
        np.save(save_fn, density)
    return density

def compute_tpcf(
    positions, 
    edges,
    boxsize: float,
    los: str = 'z', 
    save_fn: Path = None, 
    **kwargs
) -> TwoPointCorrelationFunction:
    """
    Compute the two-point correlation function (2PCF) in (s, mu) bins
    using the pycorr package.
    
    Parameters
    ----------
    positions : np.ndarray
        The positions of the galaxies, with shape (N, 3).
    edges : tuple of np.ndarray
        The edges of the bins for the 2PCF. Should be a tuple of two arrays: (s_edges, mu_edges).
    boxsize : float
        The size of the simulation box.
    los : str, optional
        The line-of-sight direction. Defaults to 'z'.
    save_fn : Path, optional
        The filename to save the 2PCF trough the `save` method of the `pycorr` object. If None, the 2PCF is not saved. Defaults to None.
    **kwargs
        Additional keyword arguments to pass to the TwoPointCorrelationFunction constructor.
    
    Returns
    -------
    TwoPointCorrelationFunction
        The computed two-point correlation function.
    """
    tpcf = TwoPointCorrelationFunction(
        data_positions1 = positions,
        edges = edges,
        boxsize = boxsize,
        los = los,
        mode = 'smu',
        position_type = 'pos', 
        compute_sepsavg = False, # Ensure consistency with different computations
        **kwargs
    )
    if save_fn is not None:
        tpcf.save(save_fn)
    
    return tpcf

def compute_density_split(
    positions,
    edges,
    boxsize: float,
    los: str = 'z',
    cellsize: float = 5.0,
    smoothing_radius: float = 10.0,
    nquantiles: int = 5,
    save_fn_ccf: Path = None,
    save_fn_acf: Path = None,
    **kwargs
) -> tuple[list, list]:
    """
    Compute the density split statistics: the cross-correlation between
    the quantile regions and the data, and the auto-correlation of the
    quantile regions.
    
    Parameters
    ----------
    positions : np.ndarray
        The positions of the galaxies, with shape (N, 3).
    edges : tuple of np.ndarray
        The edges of the bins for the correlation functions. Should be a tuple of two arrays:
        (s_edges, mu_edges).
    boxsize : float
        The size of the simulation box.
    los : str, optional
        The line-of-sight direction. Defaults to 'z'.
    cellsize : float, optional
        The size of the cells for the density field. Defaults to 5.0.
    smoothing_radius : float, optional
        The radius for smoothing the density field. Defaults to 10.0.
    nquantiles : int, optional
        The number of quantiles to split the density field into. Defaults to 5.
    save_fn_ccf : Path, optional
        The filename to save the cross-correlation function as a list of `pycorr` objects. If None, the CCF is not saved. Defaults to None.
    save_fn_acf : Path, optional
        The filename to save the auto-correlation function as a list of `pycorr` objects. If None, the ACF is not saved. Defaults to None.
    **kwargs
        Additional keyword arguments to pass to the correlation function computations.
    
    Returns
    -------
    tuple of list
        A tuple containing the cross-correlation function and the auto-correlation functions for each quantile.
    """
    ds = DensitySplit(data_positions=positions, boxsize=boxsize, boxcenter=boxsize/2, cellsize=cellsize)
    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(nquantiles=nquantiles, query_method='randoms')

    quantile_data_correlation = ds.quantile_data_correlation(
        data_positions = positions,
        edges = edges,
        los = los,
        compute_sepsavg = False,
        **kwargs
    )
    if save_fn_ccf is not None:
        np.save(save_fn_ccf, quantile_data_correlation)
        ds.logger.info(f'Saved {save_fn_ccf}')
    
    quantile_correlation = ds.quantile_correlation(
        edges = edges,
        los = los,
        compute_sepsavg = False,
        **kwargs
    )
    if save_fn_acf is not None:
        np.save(save_fn_acf, quantile_correlation)
        ds.logger.info(f'Saved {save_fn_acf}')
    
    return quantile_data_correlation, quantile_correlation

def compute_power_spectrum(
    positions, 
    edges,
    boxsize: float,
    los: str = 'z', 
    save_fn: Path = None, 
    **kwargs
):
    pk = CatalogFFTPower(
        data_positions1 = positions,
        edges = edges,
        boxsize = boxsize,
        los = los,
        position_type = 'pos',
        **kwargs,
    )
    if save_fn is not None:
        pk.save(save_fn)
    
    return pk

def compute_density_split_power(
    positions,
    edges, 
    boxsize: float,
    los: str = 'z',
    cellsize: float = 5.0,
    smoothing_radius: float = 10.0,
    nquantiles: int = 5,
    save_fn_ccf: Path = None,
    save_fn_acf: Path = None,
    **kwargs
):
    """
    Compute the power spectra for the DensitySplit quantiles

    Parameters
    ----------
    positions : np.ndarray
        The positions of the galaxies, with shape (N, 3).
    edges : tuple of np.ndarray
        The edges of the bins for the power spectrum. Should be a tuple of two arrays:
        (k_edges, mu_edges).
    boxsize : float
        The size of the simulation box.
    los : str, optional
        The line-of-sight direction. Defaults to 'z'.
    cellsize : float, optional
        The size of the cells for the density field. Defaults to 5.0.
    smoothing_radius : float, optional
        The radius for smoothing the density field. Defaults to 10.0.
    nquantiles : int, optional
        The number of quantiles to split the density field into. Defaults to 5.
    save_fn_ccf : Path, optional
        The filename to save the cross-correlation function as a list of poles from `pypower`. If None, the CCF is not saved. Defaults to None.
    save_fn_acf : Path, optional
        The filename to save the auto-correlation function as a list of poles from `pypower`. If None, the ACF is not saved. Defaults to None.
    **kwargs
        Additional keyword arguments to pass to the power spectra computations.
        
    Returns
    -------
    _type_
        _description_
    """
    ds = DensitySplit(data_positions=positions, boxsize=boxsize, boxcenter=boxsize/2, cellsize=cellsize)
    ds.set_density_contrast(smoothing_radius=smoothing_radius)
    ds.set_quantiles(nquantiles=nquantiles, query_method='randoms')
    
    quantile_data_power = ds.quantile_data_power(
        data_positions = positions,
        edges = edges,
        los = los,
        **kwargs,
    )
    if save_fn_ccf is not None:
        np.save(save_fn_ccf, quantile_data_power)
        ds.logger.info(f'Saved {save_fn_ccf}')
    
    quantile_power = ds.quantile_power(
        edges = edges,
        los = los,
        **kwargs,
    )
    if save_fn_acf is not None:
        np.save(save_fn_acf, quantile_power)
        ds.logger.info(f'Saved {save_fn_acf}')

    return quantile_data_power, quantile_power

#%% Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Measure clustering statistics from HOD catalogs.")
    parser.add_argument('--config', type=str, default=None, help='Path to a configuration file (YAML format) with the parameters below. Command line arguments override config file settings.')
    parser.add_argument('--dump_config', action='store_true', help='If set, dumps the current configuration in the console and exits.')
    parser.add_argument('-c', '--cosmologies', type=int, nargs='+', help='List of cosmology indices to process.')
    parser.add_argument('-p', '--phases', type=int, nargs='+', help='List of phase indices to process.')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', help='List of seeds to process.')
    parser.add_argument('--hods', type=int, nargs='+', default=None, help='List of HOD indices to process. If None, processes all HODs in the catalog file.')
    parser.add_argument('-t', '--sim_type', type=str, default='base', help='Simulation type (e.g., base, small).')
    parser.add_argument('--abacus_tracer', type=str, default='BGS', help='Tracer type for Abacus catalogs loading (e.g., BGS, LRG), see `acm.utils.paths.get_Abacus_dirs`.') # NOTE: Should be temporary ?
    parser.add_argument('-z', '--redshift', type=float, default=0.2, help='Redshift of the simulations to load.')
    parser.add_argument('-n', '--n_hod', type=int, default=100, help='Number of HODs to run per cosmology, phase and seed.')
    parser.add_argument('--hod_start', type=int, default=None, help='Starting index for HODs to process (for resuming interrupted runs).')
    parser.add_argument('--hod_dir', type=str, help='Directory containing the HOD catalogs.')
    parser.add_argument('-nz', '--target_density', type=float, default=1e-2, help='Target density for the tracer in h^3 Mpc^-3.')
    parser.add_argument('-ns', '--target_density_sigma', type=float, default=1e-5, help='Allowed sigma around the target density in h^3 Mpc^-3.')
    parser.add_argument('--process_underdense', action='store_true', help='Whether to process HODs that are underdense compared to the target density.')
    parser.add_argument('--add_rsd', action='store_true', help='Whether to add RSD (redshift space distortion) effects.')
    parser.add_argument('--add_ap', action='store_true', help='Whether to add AP (Alcock-Paczynski) effects.')
    parser.add_argument('--measurements', type=str, nargs='+', help='List of measurements to compute. See details below.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration if available.')
    parser.add_argument('--nthreads', type=int, default=4, help='Number of threads to use for computations.')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the measurements. If None, measurements are not saved.')
    parser.add_argument('--save_galaxies', action='store_true', help='Whether to save the generated galaxy catalogs.')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing measurement files.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (e.g., DEBUG, INFO, WARNING, ERROR).')
    parser.add_argument('--log_file', type=str, default=None, help='File to save logs. If None, logs are printed to console.')
    
    parser.epilog = """
    Measurements options:
    - 'density': Compute and save the number density of the tracer.
    - 'tpcf': Compute and save the two-point correlation function (2PCF in (s, mu) bins).
    - 'density_split': Compute and save the density split statistics (cross-correlation and auto-correlation).
    - 'power_spectrum': Compute and save the power spectrum (in (k, mu) bins).
    - 'density_split_power': Compute and save the density split power spectra (cross-power and auto-power).
    You can specify multiple measurements by providing a list, e.g., --measurements density tpcf power_spectrum density_split_power
    """
    
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
            parser.set_defaults(**config_args)
        args = parser.parse_args() # Re-parse arguments with config defaults
    if args.dump_config:
        parser.print_help(sys.stdout)
        print("\nCurrent configuration:")
        print("----------------------")
        tmp_args = vars(args).copy()
        del tmp_args['config']
        del tmp_args['dump_config']
        for arg in tmp_args:
            print(f"{arg}: {getattr(args, arg)}")
        sys.exit(-1)
    
    # Setup argument parameters
    cosmologies = args.cosmologies
    phases = args.phases
    seeds = args.seeds
    sim_type = args.sim_type
    abacus_tracer = args.abacus_tracer
    redshift = args.redshift
    n_hod = args.n_hod
    hod_dir = args.hod_dir
    
    tracer_density = [args.target_density - args.target_density_sigma, args.target_density] if args.target_density else None
    process_underdense = args.process_underdense
    add_rsd = args.add_rsd
    add_ap = args.add_ap

    measurements = args.measurements
    gpu = args.gpu
    nthreads = args.nthreads
    save_galaxies = args.save_galaxies
    overwrite = args.overwrite
    
    # Setup logging
    setup_logging(level=args.log_level, filename=args.log_file)
    logger = logging.getLogger(__file__.split('/')[-1])

    for cosmo_idx in cosmologies:
        hod_file = np.genfromtxt(Path(hod_dir) / f'Bouchard25_c{cosmo_idx:03d}.csv', delimiter=',', names=True)
        hod_params = list(hod_file.dtype.names)
        hods = range(len(hod_file)) if args.hods is None else args.hods # Get the indexes of the HODs to process
        if args.hod_start is not None:
            hods = [h for h in hods if h >= args.hod_start]
        if len(hods) == 0:
            logger.warning(f'No HODs to process for cosmology {cosmo_idx}. Skipping...')
            continue
        
        for phase_idx in phases:
            logger.info(f"Processing c{cosmo_idx:03d}_ph{phase_idx:03d}")
            
            abacus = BoxHOD(
                varied_params = hod_params,
                cosmo_idx = cosmo_idx,
                phase_idx = phase_idx,
                sim_type = sim_type,
                redshift = redshift,
                DM_DICT = get_Abacus_dirs(tracer=abacus_tracer, simtype='box'), # TODO : Maybe change this if using Hanyu's profile version (no need for BGS-specific catalog ?)
            )
            
            for seed in seeds:
                N = 1
                for hod_idx in hods:
                    if N > n_hod: continue
                    hod = {key: hod_file[key][hod_idx] for key in hod_params}
                    
                    save_dir = Path(args.save_dir) / sim_type / f'c{cosmo_idx:03d}_ph{phase_idx:03d}' / f'seed{seed}' / f'hod{hod_idx:03d}'
                    # No mkdir here to avoid creating empty directories
                    # NOTE: args.save_dir can't be None !
                    
                    save_fn = get_save_fn(save_dir, measurement='galaxies', extension='fits', mkdir=False, exist_ok=overwrite)
                    if save_fn is None:
                        save_fn = get_save_fn(save_dir, measurement='galaxies', extension='fits')
                        logger.info(f'HOD file exists. Loading catalog from file...')
                        catalog, header = fitsio.read(save_fn, header=True)
                        # NOTE: in theory, all these parameters exist already in the BoxHOD instance above, 
                        # but to be consistent with the file, we'll use the header values
                        boxsize = BoxHOD.get_boxsize(header['BOXSIZE'], add_ap=add_ap, los='x', q_par=header.get('Q_PAR', 1.0), q_perp=header.get('Q_PERP', 1.0))
                        density = compute_number_density(catalog, boxsize)
                        abacus.in_density = density > min(tracer_density) if tracer_density else True
                    else:
                        catalog = abacus.run(
                            hod,
                            nthreads = nthreads,
                            tracer_density = tracer_density,
                            process_underdense = process_underdense,
                            seed = seed,
                            add_ap = add_ap,
                            save_fn = save_fn if save_galaxies else None,
                        )
                        catalog = catalog['LRG'] # NOTE : assuming LRG tracer here
                    if not abacus.in_density and not process_underdense: continue
                    N += abacus.in_density

                    if 'density' in measurements:
                        save_fn = get_save_fn(save_dir, measurement='density')
                        boxsize = BoxHOD.get_boxsize(boxsize=abacus.boxsize, add_ap=add_ap, los='x', q_par=abacus.q_par, q_perp=abacus.q_perp)
                        density = compute_number_density(catalog, boxsize, save_fn=save_fn)
                        logger.info(f"Density for HOD {hod_idx:03d}: {density:.4e} h^3 Mpc^-3")

                    for los in ['x', 'y', 'z']:
                        logger.info(f'Computing measurements for HOD {hod_idx:03d}, seed {seed}, los {los}')
                        boxsize = BoxHOD.get_boxsize(
                            boxsize=abacus.boxsize, 
                            add_ap=add_ap, 
                            los=los, 
                            q_par=abacus.q_par, 
                            q_perp=abacus.q_perp
                        )
                        positions = BoxHOD.get_positions(
                            catalog, 
                            los=los,
                            add_rsd=add_rsd,
                            hubble=abacus.hubble,
                            az=abacus.az,
                            boxsize=abacus.boxsize,
                            add_ap=add_ap,
                            q_par=abacus.q_par,
                            q_perp=abacus.q_perp,
                        )
                        
                        logger.debug(f'Positions shape: {positions.shape}')
                        logger.info(f'Box size: {boxsize}')
                        
                        if 'tpcf' in measurements:
                            save_fn = get_save_fn(save_dir, measurement='tpcf', los=los, exist_ok=overwrite)
                            if save_fn is None: # Only if the file exists and overwrite is False
                                logger.info(f'TPCF file exists. Skipping...')
                            else:
                                # NOTE : hardcoded settings for simplification
                                kwargs = dict( # Dict for code readability
                                    edges = (np.arange(0, 150, 1), np.linspace(-1, 1, 120)), # s and mu edges for the correlation functions
                                    boxsize = boxsize,
                                    los = los,
                                    save_fn = save_fn,
                                    nthreads = nthreads,
                                    gpu = gpu,
                                )
                                compute_tpcf(positions, **kwargs)

                        if 'density_split' in measurements:
                            save_fn_ccf = get_save_fn(save_dir, measurement='quantile_data_correlation', los=los, exist_ok=overwrite)
                            save_fn_acf = get_save_fn(save_dir, measurement='quantile_correlation', los=los, exist_ok=overwrite)
                            if save_fn_ccf is None and save_fn_acf is None:
                                logger.info(f'Quantile files exist. Skipping...')
                            else:
                                # NOTE : hardcoded settings for simplification
                                kwargs = dict(
                                    edges = (np.arange(0, 31, 1), np.linspace(-1, 1, 120)),
                                    cellsize = 5.0,
                                    smoothing_radius = 10.0,
                                    nquantiles = 5,
                                    boxsize = boxsize,
                                    los = los,
                                    save_fn_ccf = save_fn_ccf,
                                    save_fn_acf = save_fn_acf,
                                    nthreads = nthreads,
                                    gpu = gpu,
                                )
                                compute_density_split(positions, **kwargs)

                        if 'power_spectrum' in measurements:
                            save_fn = get_save_fn(save_dir, measurement='power_spectrum', los=los, exist_ok=overwrite)
                            if save_fn is None:
                                logger.info(f'Power spectrum file exists. Skipping...')
                            else:
                                # NOTE : hardcoded settings for simplification
                                kwargs = dict( # Dict for code readability
                                    edges = (np.arange(0.01, 0.5, 0.01), np.linspace(-1, 1, 120)), # k and mu edges for the power spectrum
                                    boxsize = boxsize,
                                    los = los,
                                    save_fn = save_fn,
                                    nthreads = nthreads,
                                    gpu = gpu,
                                )
                                compute_power_spectrum(positions, **kwargs)
                        
                        if 'density_split_power' in measurements:
                            save_fn_ccf = get_save_fn(save_dir, measurement='quantile_data_power', los=los, exist_ok=overwrite)
                            save_fn_acf = get_save_fn(save_dir, measurement='quantile_power', los=los, exist_ok=overwrite)
                            if save_fn_ccf is None and save_fn_acf is None:
                                logger.info(f'Quantile power files exist. Skipping...')
                            else:
                                # NOTE : hardcoded settings for simplification
                                kwargs = dict(
                                    edges = (np.arange(0.01, 0.5, 0.01), np.linspace(-1, 1, 120)),
                                    cellsize = 5.0,
                                    smoothing_radius = 10.0,
                                    nquantiles = 5,
                                    boxsize = boxsize,
                                    los = los,
                                    save_fn_ccf = save_fn_ccf,
                                    save_fn_acf = save_fn_acf,
                                    nthreads = nthreads,
                                    gpu = gpu,
                                )
                                compute_density_split_power(positions, **kwargs)

# TODO : check if looping over los one time is faster than looping over los for each observable