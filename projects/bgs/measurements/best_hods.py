# For each cosmology, finds the index of the HOD parameters that minimizes the chi2 between the statistics and the reference data.
# This assumes the reference data statistics are provided within the script.

import numpy as np
from pathlib import Path
from acm.observables import BaseCombinedObservable as CombinedObservable
from acm.projects.bgs import *
from acm.projects.bgs.default import cosmo_list

# TODO : Define how to read the reference ?
# Idea : Create a BaseCombinedObservable class that reads only the reference data ?
reference = None

def get_chi2(diff, covariance_matrix):
    inv_cov = np.linalg.inv(covariance_matrix)
    chi2 = diff @ inv_cov @ diff
    return chi2

if __name__ == '__main__':
    
    # Load the LHC & covariance matrix
    stats = CombinedObservable([
        GalaxyCorrelationFunctionMultipoles(),
        DensitySplitCorrelationFunctionMultipoles(),
    ])
    lhc_y = stats.lhc_y
    covariance_matrix = stats.get_covariance_matrix()
    
    best_hod_dict = {}
    for i, cosmo_idx in enumerate(cosmo_list):
        best_chi2 = None
        best_hod = None
        for hod in range(100):
            idx = i*100 + hod
            hod_stat = lhc_y[idx]
            diff = hod_stat - reference
            
            chi2 = get_chi2(diff, covariance_matrix)
            
            if best_chi2 is None or chi2 < best_chi2:
                print(f'New best chi2: {chi2:.2f} for HOD {hod} and cosmology {cosmo_idx}')
                best_chi2 = chi2
                best_hod = hod

        best_hod_dict[cosmo_idx] = best_hod
    
    # Save the best HODs in the same directory as the LHC
    print(f'Best HODs: {best_hod_dict}')
    lhc_dir = Path(stats.paths[stats.observables[0].stat_name]['lhc_dir'])
    np.save(lhc_dir / 'best_hod.npy', best_hod_dict)