import matplotlib.pyplot as plt
import acm.observables.emc as emc
from acm.utils.covariance import get_covariance_correction
from pathlib import Path
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
import torch
import torch.func as func
import argparse


def get_gradient(statistic):
    fiducial_parameters = stat_map[statistic].x
    fiducial_parameters = torch.tensor(fiducial_parameters.astype(np.float32), requires_grad=True,).unsqueeze(0)
    # fiducial_parameters = torch.tensor(fiducial_parameters.values.astype(np.float32), requires_grad=True,).unsqueeze(0)
    def model_fn(x_batch):
        # Add batch dimension for the model
        return stat_map[statistic].model.get_prediction(x_batch)
    return func.jacrev(model_fn)(fiducial_parameters).detach().squeeze().numpy()

def get_full_gradients(statistics):
    return np.vstack([get_gradient(stat) for stat in statistics],)

def get_precision_matrix(full_covariance_vector):
    if full_covariance_vector.shape[1] == 1:
        # For 1D case, calculate variance manually and force a 1x1 matrix
        variance = np.var(full_covariance_vector, ddof=1)
        covariance_matrix = np.array([[variance]])
    else:
        covariance_matrix = np.cov(full_covariance_vector.T)
    correction = get_covariance_correction(
        n_s=full_covariance_vector.shape[0],
        n_d=len(covariance_matrix),
        n_theta=20,
        method='percival-fisher',
    )
    precision_matrix = np.linalg.inv(correction * covariance_matrix)
    return precision_matrix

def get_fisher_log_det(statistics,):
    full_covariance_vector = np.hstack([stat_map[stat].covariance_y for stat in statistics])
    precision_matrix = get_precision_matrix(full_covariance_vector)
    gradients = get_full_gradients(statistics)
    fisher_matrix = np.dot(gradients.T, np.dot(precision_matrix, gradients))
    sign, fisher_log_det = np.linalg.slogdet(fisher_matrix)
    return fisher_matrix, fisher_log_det

def precompute_derivatives_and_covariance(statistics=['tpcf', 'bk']):
    precomputed = {}
    precomputed['derivatives'] = {}
    for stat_name in statistics:
        precomputed['derivatives'][stat_name] = get_gradient(stat_name)
    
    precomputed['covariance_data'] = {}
    for stat_name in statistics:
        precomputed['covariance_data'][stat_name] = stat_map[stat_name].covariance_y
    precomputed['emulator_error'] = {}
    for stat_name in statistics:
        precomputed['emulator_error'][stat_name] = stat_map[stat_name].get_emulator_error()**2

    
    precomputed['bin_counts'] = {
        stat_name: precomputed['derivatives'][stat_name].shape[1] 
        for stat_name in statistics
    }
    return precomputed

def generate_augmented_gradients(selected_bin_gradients, available_bin_gradients):

    if selected_bin_gradients.shape[0] == 0:
        return available_bin_gradients
    else:
        return np.array([
            np.vstack([selected_bin_gradients, available_bin_gradients[i]])
            for i in range(len(available_bin_gradients))
        ])


def compute_precision_matrices(selected_bin_data, available_bin_data, selected_bin_emulator_error=None, available_bin_emulator_error=None,add_emulator_error=False,):
    if selected_bin_data.shape[1] == 0:
        covariance_mocks = available_bin_data
    else:
        covariance_mocks = np.array([
            np.hstack([selected_bin_data, available_bin_data[i]])
            for i in range(len(available_bin_data))
        ])
    n_mocks = covariance_mocks.shape[1]
    n_dim = covariance_mocks.shape[-1]
    n_options = len(covariance_mocks)
    correction = get_covariance_correction(
        n_s=n_mocks,
        n_d=n_dim,
        n_theta=20,
        method='percival',
    )
    
    precision_matrices = np.zeros((n_options, n_dim, n_dim))
    for i in range(n_options):
        covariance_matrix = np.atleast_2d(np.cov(covariance_mocks[i].T))
        if add_emulator_error:
            error = np.hstack((selected_bin_emulator_error, available_bin_emulator_error[i]))
            covariance_matrix += np.diag(error)
        precision_matrices[i] = np.linalg.inv(correction * covariance_matrix)
    return precision_matrices 

def get_batch_fisher_matrices(gradients, precision_matrices):
    temp = np.einsum('mij,mjk->mik', precision_matrices, gradients)
    return np.einsum('mki,mij->mkj', gradients.transpose(0, 2, 1), temp)

def get_batch_fisher_information(fisher_matrices):
    _, logabsdet = np.linalg.slogdet(fisher_matrices)
    return logabsdet

def get_maximum_fisher_idx(selected_bin_gradients, available_bin_gradients, selected_bin_data, available_bin_data, selected_bin_emulator_error=None, available_bin_emulator_error=None, add_emulator_error=False,):
    augmented_gradients = generate_augmented_gradients(
        selected_bin_gradients,
        available_bin_gradients[:,None,:],
    )
    precision_matrices = compute_precision_matrices(
        selected_bin_data,
        available_bin_data.T[...,None],
        selected_bin_emulator_error=selected_bin_emulator_error,
        available_bin_emulator_error=available_bin_emulator_error,
        add_emulator_error=add_emulator_error,
    )
    fisher_matrices = get_batch_fisher_matrices(augmented_gradients, precision_matrices)
    fisher_information = get_batch_fisher_information(fisher_matrices)
    max_fisher_idx = np.argmax(fisher_information)
    return max_fisher_idx, fisher_information[max_fisher_idx]

def greedy_bin_selection_vectorized(precomputed, max_bins=10, add_emulator_error=False,):
    derivatives = precomputed['derivatives']
    covariance_data = precomputed['covariance_data']
    emulator_error = precomputed['emulator_error']
    print(emulator_error['pk'][0])
    statistics = list(derivatives.keys())
    
    available_bins = {stat: list(range(derivatives[stat].shape[0])) for stat in statistics}
    selected_bins = {stat: [] for stat in statistics}

    n_params = derivatives[statistics[0]].shape[1]
    selected_bin_gradients = np.zeros((0, n_params))
    selected_bin_data = np.zeros((covariance_data[statistics[0]].shape[0], 0))
    selected_bin_emulator_error = np.zeros((0,))
    
    current_fisher = float('-inf')
    all_fisher_values = []
    total_selected_bins = 0
    
    while total_selected_bins < max_bins:
        best_stat = None
        best_idx = None
        best_fisher = current_fisher
        
        for stat_name in statistics:
            if not available_bins[stat_name]:
                continue
                
            stat_indices = available_bins[stat_name]
            stat_gradients = derivatives[stat_name][stat_indices]
            stat_covs = covariance_data[stat_name][:, stat_indices]
            stat_emulator_error = emulator_error[stat_name][stat_indices]

            max_idx_local, max_fisher = get_maximum_fisher_idx(
                selected_bin_gradients=selected_bin_gradients,
                available_bin_gradients=stat_gradients,
                selected_bin_data=selected_bin_data,
                available_bin_data=stat_covs,  
                selected_bin_emulator_error=selected_bin_emulator_error,
                available_bin_emulator_error=stat_emulator_error,
                add_emulator_error=add_emulator_error,
            )
            
            bin_idx = stat_indices[max_idx_local]
            
            if max_fisher > best_fisher:
                best_fisher = max_fisher
                best_stat = stat_name
                best_idx = bin_idx
        
        improvement = best_fisher - current_fisher
        if improvement < 1e-6 or best_stat is None:
            print("No significant improvement found, stopping early")
            break
            
        selected_bins[best_stat].append(best_idx)
        available_bins[best_stat].remove(best_idx)
        
        bin_gradient = derivatives[best_stat][best_idx:best_idx+1]
        bin_data = covariance_data[best_stat][:, best_idx:best_idx+1]
        bin_emulator_error = emulator_error[best_stat][best_idx:best_idx+1]
        
        if total_selected_bins == 0:
            selected_bin_gradients = bin_gradient
            selected_bin_data = bin_data
            selected_bin_emulator_error = bin_emulator_error
        else:
            selected_bin_gradients = np.vstack([selected_bin_gradients, bin_gradient])
            selected_bin_data = np.hstack([selected_bin_data, bin_data])
            selected_bin_emulator_error = np.hstack([selected_bin_emulator_error, bin_emulator_error])
        
        total_selected_bins += 1
        current_fisher = best_fisher
        all_fisher_values.append(current_fisher)
        
        if total_selected_bins % 5 == 0 or total_selected_bins == 1 or total_selected_bins == max_bins:
            print(f"Selected {total_selected_bins}/{max_bins} bins, Fisher: {current_fisher:.4f}")
            distribution = ", ".join([f"{stat}: {len(bins)}" for stat, bins in selected_bins.items()])
            print(f"Distribution: {distribution}")
            print(f"Added {best_stat}:{best_idx} with improvement {improvement:.4f}")
    
    return selected_bins, current_fisher, all_fisher_values

def run_optimization_vectorized(statistics=['tpcf', 'bk'], max_bins=100, add_emulator_error=False):
    print(f"Precomputing data for statistics: {statistics}")
    precomputed_data = precompute_derivatives_and_covariance(statistics)
    
    print(f"\nRunning greedy selection with max_bins={max_bins}")
    selected_bins, final_fisher, all_fisher_values = greedy_bin_selection_vectorized(precomputed_data, max_bins=max_bins, add_emulator_error=add_emulator_error)
    
    print("\nFinal selection:")
    for stat, bins in selected_bins.items():
        print(f"{stat}: {len(bins)} bins selected")
    print(f"Total: {sum(len(bins) for bins in selected_bins.values())} bins")
    print(f"Final Fisher log-determinant: {final_fisher:.4f}")
    
    return selected_bins, final_fisher, all_fisher_values


if __name__== '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--statistics', nargs='+', help='List of statistics to use, e.g. GalaxyPowerSpectrumMultipoles')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the results.')

    args = parser.parse_args()

    paths = {
        'data_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/compressed/',
        'measurements_dir': '/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/',
        'hod_dir': '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/',
        'param_dir': None
    }

    select_filters={'cosmo_idx': [0], 'hod_idx': [0,],}
    kwargs = dict(numpy_output=True, squeeze_output=True, paths=paths, select_filters=select_filters)

    stat_map = {
        'tpcf': emc.GalaxyCorrelationFunctionMultipoles(**kwargs),
        'bk': emc.GalaxyBispectrumMultipoles(**kwargs),
        'pk' : emc.GalaxyPowerSpectrumMultipoles(**kwargs),
        'minkowski': emc.MinkowskiFunctionals(**kwargs),
        'wst': emc.WaveletScatteringTransform(**kwargs),
        'dsc_xi': emc.DensitySplitQuantileGalaxyCorrelationFunctionMultipoles(**kwargs),
        'wp': emc.ProjectedGalaxyCorrelationFunction(**kwargs),
    }

    to_combine = ['pk', 'bk', 'minkowski'] 
    selected_bins, final_fisher, all_fisher_values = run_optimization_vectorized(
        to_combine,
        max_bins=200,
        add_emulator_error=True
    )

    np.save(Path(args.output_dir) / 'selected_bins.npy', selected_bins)
    np.save(Path(args.output_dir) / 'final_fisher.npy', final_fisher)
    np.save(Path(args.output_dir) / 'all_fisher_values.npy', all_fisher_values)