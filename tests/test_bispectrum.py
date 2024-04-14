from acm.estimators import BasePolyBinEstimator, Bispectrum
from acm.utils import setup_logging
from pypower import CatalogMesh
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def read_mock_catalog():
    # read some random galaxy catalog
    data_dir = '/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior2/c000_ph000/'
    data_fn = Path(data_dir) / 'hod3177.npy'
    data = np.load(data_fn, allow_pickle=True).item()
    boxsize = 2000.0
    x = data['x']
    y = data['y']
    z_rsd = data['z_rsd']
    data_positions = np.c_[x, y, z_rsd]
    data_positions -= boxsize / 2  # this is needed for Polybin
    return data_positions, boxsize


def test_bispectrum():
    data_positions, boxsize = read_mock_catalog()

    mesh = CatalogMesh(data_positions, boxsize=boxsize, nmesh=128,
                       boxcenter=0, wrap=True, position_type='pos', interlacing=0).to_mesh()
    delta_mesh = mesh / np.mean(mesh) - 1

    # Define some k-bins and ell_max
    k_edges = np.arange(0.01,0.10,0.02)
    k_edges_squeeze = np.arange(0.01,0.15,0.02) # add extra high-k bins for squeezed triangles
    lmax = 2

    base = BasePolyBinEstimator(boxsize=boxsize, boxcenter=[0, 0, 0],
                                gridsize=128, sightline='global', nthreads=256)

    # Load the bispectrum class
    bs = Bispectrum(base, 
                    k_edges, # one-dimensional bin edges
                    applySinv = None, # weighting function [only needed for unwindowed estimators]
                    mask = None, # real-space mask
                    lmax = lmax, # maximum Legendre multipole
                    k_bins_squeeze = k_edges_squeeze, # squeezed bins
                    include_partial_triangles = False, # whether to include bins whose centers do not satisfy triangle conditions
                    )

    bk = bs.Bk_ideal(delta_mesh, discreteness_correction=False)

    k123 = bs.get_ks()
    cs = ['r','g']
    weight = k123.prod(axis=0)
    fig, ax = plt.subplots()
    for l in range(0,lmax+1,2):
        ax.plot(weight*bk['b%d'%l],c=cs[l//2], ls='-', label=r'$\ell=%d$'%l)
    ax.set_xlabel(r'Bin Index',fontsize=15)
    ax.set_ylabel(r'$k_1k_2k_3\,B_\ell(k_1,k_2,k_3)$',fontsize=15)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    setup_logging()
    # test_density_split()
    # test_wst()
    test_bispectrum()