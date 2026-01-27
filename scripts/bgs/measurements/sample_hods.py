import argparse
from sunbird.inference.priors import Bouchard25
from acm.hod.parameters import HODLatinHypercube
from acm.utils.abacus import load_abacus_cosmologies
from acm.utils.default import cosmo_list

# Default parameters
filename = '/pscratch/sd/s/sbouchar/acm/bgs-20/parameters/cosmo_params/AbacusSummit.csv'
parameters = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
order = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'nrun', 'N_ur', 'w0_fld', 'wa_fld', 'logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa', 'alpha_c', 'alpha_s', 's', 'A_cen', 'A_sat', 'B_cen', 'B_sat']

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--number', type=int, default=85*100, help='Number of HOD to sample')
parser.add_argument('-f', '--filename', type=str, default=filename, help='Path to the AbacusSummit cosmology parameters CSV file')
parser.add_argument('-p', '--parameters', type=str, nargs='+', default=parameters, help='List of cosmological parameters to include')
parser.add_argument('-c', '--cosmologies', type=int, nargs='+', default=cosmo_list, help='List of cosmology indices to sample HODs for')
parser.add_argument('-o', '--order', type=str, nargs='+', default=order, help='Order of parameters in the final output CSV files')
parser.add_argument('-s', '--save_dir', type=str, default=None, help='Directory to save the sampled HOD parameters')
args = parser.parse_args()

n = args.number
filename = args.filename
parameters = args.parameters
cosmologies = args.cosmologies
save_dir = args.save_dir

print(f'Sampling {n} HODs for {len(cosmologies)} cosmologies from {filename} with parameters {parameters}')

ranges = Bouchard25().ranges
cosmo_params = load_abacus_cosmologies(
    filename = filename, 
    cosmologies = cosmologies, 
    parameters = parameters, 
    mapping = {'alpha_s': 'nrun'}, # map alpha_s to nrun to avoid overwriting alpha_s in HOD params
)

lhc = HODLatinHypercube(ranges=ranges, order=order)
lhc.sample(n)
save_fn = [f'{save_dir}/hod_params/Bouchard25_c{c:03d}.csv' for c in cosmologies] if save_dir else None
lhc.split_by_cosmo(cosmologies, save_fn=save_fn)
save_fn = [f'{save_dir}/cosmo+hod_params/AbacusSummit_c{c:03d}.csv' for c in cosmologies] if save_dir else None
lhc.add_cosmo_params(cosmo_params, save_fn=save_fn)

if save_dir:
    print(f'Saved files to {save_dir}')