from acm.estimators.galaxy_clustering import Bispectrum
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

    bspec = Bispectrum(boxsize=boxsize, boxcenter=0, nmesh=128,
                       sightline='global', nthreads=256)
    
    bspec.set_binning(
        k_bins=np.arange(0.01,0.10,0.02),
        lmax=2,
        k_bins_squeeze=np.arange(0.01,0.15,0.02),
        include_partial_triangles=False,
    )

    bspec.assign_data(positions=data_positions, wrap=True)
    bspec.set_density_contrast()

    bk = bspec.Bk_ideal(discreteness_correction=False)

    k123 = bspec.get_ks()
    cs = ['r','g']
    weight = k123.prod(axis=0)
    fig, ax = plt.subplots()
    for l in range(0, 3,2):
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