import pandas as pd
from acm.hod.parameters import HODLatinHypercube
from acm.projects.bgs import Bouchard25
from acm.projects.bgs.default import cosmo_list

# Parameters
param_dir = f'/pscratch/sd/s/sbouchar/acm/bgs/parameters/'
n_samples = len(cosmo_list) * 100

print('Warning : The sigma hod parameter is sampled as log10(sigma) !')

# Select hod parameters 
ranges = Bouchard25().ranges
lhc = HODLatinHypercube(ranges)
hod_params = lhc.sample(n_samples)
hod_params = lhc.split_by_cosmo(cosmos=cosmo_list)

# Cosmology parameters
cosmo_params = pd.read_csv(f'{param_dir}/cosmo_params/AbacusSummit.csv')
acceptable_root = [f'abacus_cosm{c:03d}' for c in cosmo_list]
cosmo_params = cosmo_params[cosmo_params['root'].isin(acceptable_root)] # Remove unused cosmologies
cosmo_params = cosmo_params.iloc[:, 2:] # Remove two first columns
keys_to_keep = ['omega_cdm', 'omega_b', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld'] # Keep only relevant keys
cosmo_params_new = cosmo_params[keys_to_keep] # + copy to avoid issues
cosmo_params_new = cosmo_params_new.rename(columns={'alpha_s': 'nrun'}) # To avoid parameter name duplication with HODs
cosmo_params = cosmo_params_new

# Dicts to pandas
for i, cosmo_idx in enumerate(cosmo_list):
    hod_df = pd.DataFrame(hod_params[f'c{cosmo_idx:03d}'])
    hod_df.to_csv(f'{param_dir}/hod_params/Bouchard25_c{cosmo_idx:03d}.csv', index=False)
    
    cosmo_df = pd.DataFrame([cosmo_params.iloc[i] for _ in range(hod_df.shape[0])]) # Repeat cosmology parameters
    
    full_df = pd.concat([cosmo_df.reset_index(drop=True), hod_df.reset_index(drop=True)], axis=1) # Concatenate
    full_df.to_csv(f'{param_dir}/cosmo+hod_params/AbacusSummit_c{cosmo_idx:03d}.csv', index=False)
    
keys = list(full_df.keys())
print(f'Saved {n_samples} samples to {param_dir}, with keys: {keys}')