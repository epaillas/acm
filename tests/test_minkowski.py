from pathlib import Path
import glob
import fitsio
import numpy as np
import matplotlib.pyplot as plt

from acm.estimators.galaxy_clustering.jaxmf import MinkowskiFunctionals
from acm import setup_logging

def get_hod_fns(cosmo=1, phase=0, redshift=0.5, seed=0):
    """
    Get the list of HOD file names for a given cosmology,
    phase, and redshift.
    """
    base_dir = '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/'
    hod_dir = Path(base_dir) / f'c{cosmo:03}_ph{phase:03}/seed{seed}/'
    hod_fns = glob.glob(str(Path(hod_dir) / f'hod*.fits'))
    return sorted(hod_fns)

def get_hod_positions(filename, los='z'):
    """Get redshift-space positions from a HOD file."""
    hod, header = fitsio.read(filename, header=True)
    qpar, qperp = header['Q_PAR'], header['Q_PERP']
    if los == 'x':
        pos = np.c_[hod['X_RSD'], hod['Y_PERP'], hod['Z_PERP']]
        boxsize = np.array([2000/qpar, 2000/qperp, 2000/qperp])
    elif los == 'y':
        pos = np.c_[hod['X_PERP'], hod['Y_RSD'], hod['Z_PERP']]
        boxsize = np.array([2000/qperp, 2000/qpar, 2000/qperp])
    elif los == 'z':
        pos = np.c_[hod['X_PERP'], hod['Y_PERP'], hod['Z_RSD']]
        boxsize = np.array([2000/qperp, 2000/qperp, 2000/qpar])
    return pos, boxsize

def get_box_args(boxsize, cellsize):
    meshsize = (boxsize / cellsize).astype(int)
    return dict(boxsize=boxsize, boxcenter=0.0, meshsize=meshsize)

def test_minkowski():
    """
    Test Minkowski functionals computation on a HOD catalog.
    """
    # Load thresholds for different smoothing radii
    # For testing purposes, we'll create simple thresholds if the file doesn't exist
    thresholds_fn = '/pscratch/sd/e/epaillas/emc/Thresholds_for_MFs_with_Rg5_7_10_15.npy'
    
    if Path(thresholds_fn).exists():
        thresholds_all = np.load(thresholds_fn, allow_pickle=True).item()
        smoothing_radii = [5, 7, 10, 15]
    else:
        # Create simple test thresholds
        print(f"Warning: {thresholds_fn} not found. Using test thresholds.")
        smoothing_radii = [10]
        thresholds_all = {f"Thresholds_Rg{r}": np.linspace(-2, 2, 10) for r in smoothing_radii}
    
    # Get HOD catalog
    box_args = get_box_args(boxsize, cellsize=3.9)
    
    # Initialize Minkowski functionals estimator
    mf = MinkowskiFunctionals(data_positions=positions, thres_mask=-5, **box_args)
    
    # Store results
    mfs3d_all = {}
    
    # Compute for each smoothing radius
    for smoothing_radius in smoothing_radii:
        print(f"\nComputing Minkowski functionals for smoothing radius = {smoothing_radius} Mpc/h")
        thresholds = thresholds_all[f"Thresholds_Rg{smoothing_radius}"]
        
        # Set density contrast with smoothing
        mf.set_density_contrast(smoothing_radius=smoothing_radius)
        
        # Compute Minkowski functionals
        mf3d = mf.run(thresholds=thresholds)
        
        # Store results
        mfs3d_all[f'Rg{smoothing_radius}'] = mf3d
        mfs3d_all[f'thresholds_Rg{smoothing_radius}'] = thresholds
        
        print(f"  MF shape: {mf3d.shape}")
        print(f"  MF0 range: [{mf3d[:, 0].min():.4f}, {mf3d[:, 0].max():.4f}]")
        print(f"  MF1 range: [{mf3d[:, 1].min():.4f}, {mf3d[:, 1].max():.4f}]")
        print(f"  MF2 range: [{mf3d[:, 2].min():.4f}, {mf3d[:, 2].max():.4f}]")
        print(f"  MF3 range: [{mf3d[:, 3].min():.4f}, {mf3d[:, 3].max():.4f}]")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    mf_labels = [r'$V_0$ (Volume)', r'$V_1$ (Surface)', r'$V_2$ (Curvature)', r'$V_3$ (Euler)']
    
    for i, ax in enumerate(axes):
        for smoothing_radius in smoothing_radii:
            thresholds = mfs3d_all[f'thresholds_Rg{smoothing_radius}']
            mf3d = mfs3d_all[f'Rg{smoothing_radius}']
            
            ax.plot(thresholds, mf3d[:, i], marker='o', markersize=3, 
                   label=f'$R_s={smoothing_radius}$ Mpc/h', linewidth=1.5)
        
        ax.set_xlabel(r'Threshold $\nu$', fontsize=10)
        ax.set_ylabel(mf_labels[i], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('minkowski_test.png', bbox_inches='tight', dpi=300)
    print("\nPlot saved to minkowski_test.png")
    
    # Additional diagnostic plot: check monotonicity for V0 (should be monotonic)
    fig, ax = plt.subplots(figsize=(6, 4))
    for smoothing_radius in smoothing_radii:
        thresholds = mfs3d_all[f'thresholds_Rg{smoothing_radius}']
        mf3d = mfs3d_all[f'Rg{smoothing_radius}']
        ax.plot(thresholds, mf3d[:, 0], marker='o', markersize=4, 
               label=f'$R_s={smoothing_radius}$ Mpc/h')
    
    ax.set_xlabel(r'Threshold $\nu$', fontsize=12)
    ax.set_ylabel(r'$V_0$ (Volume Fraction)', fontsize=12)
    ax.set_title('Volume Fraction vs Threshold (should be monotonically decreasing)', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('minkowski_v0_test.png', bbox_inches='tight', dpi=300)
    print("Volume fraction plot saved to minkowski_v0_test.png")
    
    return mfs3d_all


if __name__ == '__main__':
    setup_logging()
    
    # Load HOD catalog
    hod_fn = get_hod_fns(cosmo=0, phase=0, redshift=0.5)[0]
    positions, boxsize = get_hod_positions(hod_fn, los='z')
    
    print(f"Loaded HOD catalog: {hod_fn}")
    print(f"Number of galaxies: {len(positions)}")
    print(f"Box size: {boxsize}")
    
    # Run test
    results = test_minkowski()
    
    print("\nTest completed successfully!")
