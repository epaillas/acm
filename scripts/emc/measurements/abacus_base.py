import fitsio
from pathlib import Path
import numpy as np
import time


def get_positions(cosmo_idx=0, hod_idx=30, phase_idx=0, los='z', redshift=0.5):
    """Load HOD positions from disk and apply RSD along the specified line of sight."""
    from cosmoprimo.fiducial import AbacusSummit
    hod_dir = '/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/z0.5/'
    hod_dir += f'yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
    hod_fn = Path(hod_dir) / f'hod{hod_idx:03}.fits'
    hod = fitsio.read(hod_fn)
    pos = np.c_[hod['X'], hod['Y'], hod['Z']]
    cosmo = AbacusSummit(cosmo_idx)
    hubble = 100 * cosmo.efunc(redshift)
    scale_factor = 1 / (1 + redshift)
    if los == 'x':
        pos[:, 0] += hod['VX'] / (hubble * scale_factor)
    elif los == 'y':
        pos[:, 1] += hod['VY'] / (hubble * scale_factor)
    elif los == 'z':
        pos[:, 2] += hod['VZ'] / (hubble * scale_factor)
    return pos

def compute_power_spectrum(positions, los='z'):
    """Compute the power spectrum of a set of positions using jaxpower."""
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    from jaxpower import MeshAttrs, ParticleField, BinMesh2Spectrum, compute_mesh2_spectrum
    t0 = time.time()
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
    attrs = MeshAttrs(meshsize=options['nmesh'], boxsize=boxsize, boxcenter=boxcenter)
    bins = BinMesh2Spectrum(attrs, edges={'step': options['dk']}, ells=(0, 2, 4))
    data = ParticleField(positions, attrs=attrs, exchange=True)
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mesh = mesh / mesh.mean()
    shotnoise = attrs.boxsize.prod() / data.sum()
    pk = jitted_compute_mesh2_spectrum(mesh, bin=bins, los=options['los'])
    pk = pk.clone(num_shotnoise=shotnoise * pk.norm)
    pk.attrs.update(mesh=dict(mesh.attrs), los=options['los'])
    jax.block_until_ready(pk)
    print(f'Power spectrum in ellapsed time {time.time() - t0:.2f} s')
    if options['save_fn']:
        pk.save(options['save_fn'])

def compute_bispectrum(positions, options):
    """Compute the bispectrum of a set of positions using jaxpower."""
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    from jaxpower import MeshAttrs, ParticleField, BinMesh3Spectrum, compute_mesh3_spectrum
    t0 = time.time()
    jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])
    attrs = MeshAttrs(meshsize=options['nmesh'], boxsize=boxsize, boxcenter=boxcenter)
    bins = BinMesh3Spectrum(attrs, edges={'step': options['dk']}, ells=(0, 2, 4), basis='scoccimarro')
    data = ParticleField(positions, attrs=attrs, exchange=True)
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mesh = mesh / mesh.mean() - 1
    bk = jitted_compute_mesh3_spectrum(mesh, bin=bins, los=options['los'])
    if options['save_fn']:
        bk.save(options['save_fn'])
    print(f'Bispectrum computed in {time.time() - t0:.2f} s')



if __name__ == '__main__':

    from cosmoprimo.fiducial import AbacusSummit

    fid_cosmo = AbacusSummit(0)
    boxsize = 2000
    boxcenter = 0
    
    todo_stats = ['power_spectrum', 'bispectrum']
    todo_cosmos = [0, 1, 2]
    todo_hods = [30]

    for cosmo_idx in todo_cosmos:
        for hod_idx in todo_hods:
            print(f'Processing cosmo {cosmo_idx}, hod {hod_idx}')

            positions = get_positions(cosmo_idx=0, hod_idx=30, los='z')

            if 'power_spectrum' in todo_stats:
                options = {'nmesh': 512, 'dk': 0.001, 'los': 'z', 'save_fn': None}
                compute_power_spectrum(positions, options)

            if 'bispectrum' in todo_stats:
                options = {'nmesh': 160, 'dk': 0.02, 'los': 'z', 'save_fn': None}
                compute_bispectrum(positions, options)
