from pathlib import Path
import acm.observables.emc as emc
import numpy as np


observable = emc.GalaxyBispectrumMultipoles()
sep, lhc_x, lhc_x_names, lhc_y = observable.create_lhc(
    n_hod=350, cosmos=None, ells=[0, 2], phase_idx=0, seed_idx=0
)
output_dir = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/'
output_fn = Path(output_dir) / 'bk.npy'
Path(output_dir).mkdir(parents=True, exist_ok=True)
cout = {'k123': sep, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
np.save(output_fn, cout)

sep, small_box_y = observable.create_small_box_y()
output_dir = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/covariance_sets/small_box/'
output_fn = Path(output_dir) / 'bk.npy'
Path(output_dir).mkdir(parents=True, exist_ok=True)
cout = {'k123': sep, 'cov_y': small_box_y}
np.save(output_fn, cout)