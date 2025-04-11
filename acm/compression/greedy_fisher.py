import numpy as np
import torch
import acm.observables.emc as emc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

def is_well_conditioned(matrix, threshold=1e10):
    condition_number = np.linalg.cond(matrix)
    return condition_number < threshold, condition_number

def safe_inverse(matrix,): 
    """Safely invert a potentially ill-conditioned matrix"""
    # if matrix is 1x1 return the inverse directly
    if matrix.shape == (1,1):
        return 1.0 / matrix[0,0]
    is_stable, cond_num = is_well_conditioned(matrix)
    if is_stable:
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            print("Regular inversion failed despite good condition number")
            pass
    return np.linalg.pinv(matrix)

def get_pseudodeterminant(matrix, epsilon=1.e-10): 
    U, s, Vh = np.linalg.svd(matrix)
    threshold = np.max(s) * epsilon
    significant_s = s[s > threshold]
    return np.sum(np.log(significant_s))

def safe_log_determinant(matrix, epsilon=1e-10):
    is_stable, cond_num = is_well_conditioned(matrix)
    if is_stable:
        try:
            sign, logdet = np.linalg.slogdet(matrix)
            if sign > 0:
                return logdet
        except np.linalg.LinAlgError:
            print("Regular log-determinant failed despite good condition number")
            pass
    return get_pseudodeterminant(matrix, epsilon)


def get_gradient(statistic):
    fiducial_parameters = statistic.lhc_x
    fiducial_parameters = torch.tensor(fiducial_parameters.astype(np.float32), requires_grad=True,).unsqueeze(0)
    def model_fn(x_batch):
        return statistic.model.get_prediction(x_batch)
    return torch.func.jacrev(model_fn)(fiducial_parameters).detach().squeeze().numpy()


def get_individual_fisher_information(statistic, add_inverse_correction=True, add_emulator_error=True,):
    small_box_y = statistic.small_box_y
    fiducial_parameters = statistic.lhc_x
    covariance_matrix = np.cov(small_box_y.T)
    if add_emulator_error:
        covariance_matrix += statistic.get_emulator_error()**2
    if add_inverse_correction:
        correction = statistic.get_covariance_correction(
            n_s = len(small_box_y),
            n_d = len(covariance_matrix),
            n_theta = fiducial_parameters.shape[-1],
            method='percival',
        )
    else:
        correction = 1.
    print('correction = ', correction)
    precision_matrix = safe_inverse(correction * covariance_matrix)
    gradients = get_gradient(statistic)
    fisher_matrix = np.dot(gradients.T, np.dot(precision_matrix, gradients))
    return safe_log_determinant(fisher_matrix)

def precompute_derivatives_and_covariance_simulations(statistics):
    precomputed = {
        'derivatives': {},
        'covariance_simulations': {},
        'emulator_error': {},
    }
    precomputed['derivatives'] = {}
    for stat_str, statistic in statistics.items():
        precomputed['derivatives'][stat_str] = get_gradient(statistic)
        precomputed['covariance_simulations'][stat_str] = statistic.small_box_y
        precomputed['emulator_error'][stat_str] = statistic.get_emulator_error()**2
        precomputed['bin_counts'] = precomputed['derivatives'][stat_str].shape[1]
    return precomputed


def get_all_gradients(selected_bin_gradients, available_bin_gradients):
    if selected_bin_gradients.shape[0] == 0:
        return available_bin_gradients
    else:
        return np.array([
            np.vstack([selected_bin_gradients, available_bin_gradients[i]])
            for i in range(len(available_bin_gradients))
        ])

def compute_precision_matrices(
        statistics,
        selected_bin_data, 
        available_bin_data, 
        selected_bin_emulator_error=None, 
        available_bin_emulator_error=None,
        add_emulator_error=False,
        add_inverse_correction=True,
    ):
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

    if add_inverse_correction:
        correction = list(statistics.values())[0].get_covariance_correction(
            n_s=n_mocks,
            n_d=n_dim,
            n_theta=list(statistics.values())[0].lhc_x.shape[-1],
            method='percival',
        )
    else:
        correction = 1.
    precision_matrices = np.zeros((n_options, n_dim, n_dim))
    for i in range(n_options):
        covariance_matrix = np.atleast_2d(np.cov(covariance_mocks[i].T))
        if add_emulator_error:
            error = np.hstack((selected_bin_emulator_error, available_bin_emulator_error[i]))
            covariance_matrix += np.diag(error)
        precision_matrices[i] = safe_inverse(correction * covariance_matrix)
    return precision_matrices 

def get_batch_fisher_matrices(gradients, precision_matrices):
    temp = np.einsum('mij,mjk->mik', precision_matrices, gradients)
    return np.einsum('mki,mij->mkj', gradients.transpose(0, 2, 1), temp)


def get_maximum_fisher_idx(
        statistics,
        selected_bin_gradients, 
        available_bin_gradients, 
        selected_bin_data, 
        available_bin_data, 
        selected_bin_emulator_error=None, 
        available_bin_emulator_error=None, 
        add_emulator_error=False, 
        add_inverse_correction=True,
):
    augmented_gradients = get_all_gradients(
        selected_bin_gradients,
        available_bin_gradients[:,None,:],
    )
    precision_matrices = compute_precision_matrices(
        statistics,
        selected_bin_data,
        available_bin_data.T[...,None],
        selected_bin_emulator_error=selected_bin_emulator_error,
        available_bin_emulator_error=available_bin_emulator_error,
        add_emulator_error=add_emulator_error,
        add_inverse_correction=add_inverse_correction,
    )
    fisher_matrices = get_batch_fisher_matrices(augmented_gradients, precision_matrices)
    fisher_information = np.array(
        [safe_log_determinant(fisher_matrices[i]) for i in range(fisher_matrices.shape[0])]
    )
    max_fisher_idx = np.argmax(fisher_information)
    return max_fisher_idx, fisher_information[max_fisher_idx]


def greedy_bin_selection(
    statistics, 
    max_bins=10,
    add_emulator_error=False,
    add_inverse_correction=True,
    patience=5,           
    min_improvement=0.001  
):
    precomputed = precompute_derivatives_and_covariance_simulations(statistics)
    available_bins = {stat: list(range(precomputed['derivatives'][stat].shape[0])) for stat in statistics}
    print('Total available bins:', sum([len(bins) for _, bins in available_bins.items()]))
    selected_bins = {stat: [] for stat in statistics}
    
    # Initialize selected data structures
    n_params = precomputed['derivatives'][list(statistics.keys())[0]].shape[1]
    selected_bin_gradients = np.zeros((0, n_params))
    selected_bin_data = np.zeros(
        (precomputed['covariance_simulations'][list(statistics.keys())[0]].shape[0], 0)
    )
    selected_emulator_error = np.zeros((0,))

    # Track Fisher 
    current_fisher = float('-inf')
    all_fisher_values = []
    all_fisher_values_by_statistic = {stat: 0 for stat in statistics}
    total_selected_bins = 0
    
    best_fisher_seen = float('-inf')
    best_bins_configuration = {stat: [] for stat in statistics}
    best_iteration = 0
    
    # For restoring the best configuration at the end
    best_selected_data = {
        'gradients': None,
        'bin_data': None,
        'emulator_error': None
    }
    
    iterations_without_improvement = 0
    
    while total_selected_bins < max_bins:
        best_stat = None
        best_idx = None
        best_fisher = float('-inf') 
        
        for stat_str, stat in statistics.items():
            if not available_bins[stat_str]:
                continue
                
            stat_indices = available_bins[stat_str]
            stat_gradients = precomputed['derivatives'][stat_str][stat_indices]
            stat_covs = precomputed['covariance_simulations'][stat_str][:, stat_indices]
            stat_emulator_error = precomputed['emulator_error'][stat_str][stat_indices]

            max_idx, max_fisher = get_maximum_fisher_idx(
                statistics=statistics,
                selected_bin_gradients=selected_bin_gradients,
                available_bin_gradients=stat_gradients,
                selected_bin_data=selected_bin_data,
                available_bin_data=stat_covs,
                selected_bin_emulator_error=selected_emulator_error,
                available_bin_emulator_error=stat_emulator_error,
                add_emulator_error=add_emulator_error,
                add_inverse_correction=add_inverse_correction,
            )
            
            if len(stat_indices) > 0:  
                bin_idx = stat_indices[max_idx]
                if max_fisher >= best_fisher:
                    best_fisher = max_fisher
                    best_stat = stat_str
                    best_idx = bin_idx
        
        if best_stat is None:
            print("No further bins can be added, stopping early")
            break
            
        improvement = best_fisher - current_fisher
        selected_bins[best_stat].append(best_idx)
        available_bins[best_stat].remove(best_idx)

        # Update selected data
        bin_gradient = precomputed['derivatives'][best_stat][best_idx:best_idx+1]
        bin_data = precomputed['covariance_simulations'][best_stat][:, best_idx:best_idx+1]
        bin_emulator_error = precomputed['emulator_error'][best_stat][best_idx:best_idx+1]
        
        if total_selected_bins == 0:
            selected_bin_gradients = bin_gradient
            selected_bin_data = bin_data
            selected_bin_emulator_error = bin_emulator_error
        else:
            selected_bin_gradients = np.vstack([selected_bin_gradients, bin_gradient])
            selected_bin_data = np.hstack([selected_bin_data, bin_data])
            selected_bin_emulator_error = np.hstack([selected_bin_emulator_error, bin_emulator_error])
        
        # Update tracking variables
        total_selected_bins += 1
        all_fisher_values_by_statistic[best_stat] += (best_fisher - current_fisher)
        current_fisher = best_fisher
        all_fisher_values.append(current_fisher)
        
        # Check if this is the best configuration seen so far
        if current_fisher > best_fisher_seen:
            best_fisher_seen = current_fisher
            best_iteration = total_selected_bins
            iterations_without_improvement = 0
            
            best_bins_configuration = {
                stat: list(bins) for stat, bins in selected_bins.items()
            }
            
            best_selected_data = {
                'gradients': selected_bin_gradients.copy(),
                'bin_data': selected_bin_data.copy(),
                'emulator_error': selected_bin_emulator_error.copy()
            }
        else:
            relative_improvement = abs(current_fisher - best_fisher_seen) / abs(best_fisher_seen) if best_fisher_seen != 0 else float('inf')
            if relative_improvement < min_improvement:
                iterations_without_improvement += 1
            else:
                iterations_without_improvement = 0
        
        if iterations_without_improvement >= patience:
            print(f"No significant improvement for {patience} iterations, stopping early")
            break
        
        if total_selected_bins % 5 == 0 or total_selected_bins == 1 or total_selected_bins == max_bins:
            print(f"Selected {total_selected_bins}/{max_bins} bins, Fisher: {current_fisher:.4f}")
            distribution = ", ".join([f"{stat}: {len(bins)}" for stat, bins in selected_bins.items()])
            print(f"Distribution: {distribution}")
            print(f"Added {best_stat}:{best_idx} with improvement {improvement:.4f}")
            print(f"Best Fisher so far: {best_fisher_seen:.4f} (at iteration {best_iteration})")
            print(f"Iterations without improvement: {iterations_without_improvement}/{patience}")
    
    # Check if we need to revert to the best configuration
    if best_fisher_seen > current_fisher:
        print(f"\nFinal Fisher ({current_fisher:.4f}) is lower than the best seen ({best_fisher_seen:.4f})")
        print(f"Reverting to best configuration from iteration {best_iteration}")
        
        selected_bins = best_bins_configuration
        selected_bin_gradients = best_selected_data['gradients']
        selected_bin_data = best_selected_data['bin_data']
        selected_bin_emulator_error = best_selected_data['emulator_error']
        current_fisher = best_fisher_seen
    
    return selected_bins, current_fisher, all_fisher_values

def run_greedy_fisher(statistics, max_bins=100, add_emulator_error=False, add_inverse_correction=True,):
    print(f"Precomputing data for statistics: {statistics}")
    
    print(f"\nRunning greedy selection with max_bins={max_bins}")
    selected_bins, final_fisher, all_fisher_values = greedy_bin_selection(
        statistics, 
        max_bins=max_bins, 
        add_emulator_error=add_emulator_error,
        add_inverse_correction=add_inverse_correction,
    )
    
    print("\nFinal selection:")
    for stat, bins in selected_bins.items():
        print(f"{stat}: {len(bins)} bins selected")
    print(f"Total: {sum(len(bins) for bins in selected_bins.values())} bins")
    print(f"Final Fisher log-determinant: {final_fisher:.4f}")
    
    return selected_bins, final_fisher, all_fisher_values

if __name__ == '__main__':
    select_mocks={'cosmo_idx': [0], 'hod_idx': [30,],}
    statistics = {
        'tpcf': emc.GalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        'bk': emc.GalaxyBispectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'pk' : emc.GalaxyPowerSpectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'minkowski': emc.MinkowskiFunctionals(
            select_mocks=select_mocks,
        ),
        'wst': emc.WaveletScatteringTransform(
            select_mocks=select_mocks,
        ),
        'dsc_pk': emc.DensitySplitPowerSpectrumMultipoles(
            select_mocks=select_mocks,
        ),
        'wp': emc.GalaxyProjectedCorrelationFunction(
            select_mocks=select_mocks,
        ),

        'vg_voids': emc.VoxelVoidGalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),
        'dt_voids': emc.DTVoidGalaxyCorrelationFunctionMultipoles(
            select_mocks=select_mocks,
        ),

    }


    for stat_str, statistic in statistics.items():
        print(stat_str)
        selected_bins, final_fisher, all_fisher_values = run_greedy_fisher(
            {stat_str: statistic},
            max_bins=100, 
            add_emulator_error=True,
            add_inverse_correction=True,
        )
        print(f"Greedy Fisher log-determinant: {final_fisher:.4f}")
        fisher_information = get_individual_fisher_information(
            statistic,
            add_inverse_correction=True,
            add_emulator_error=True,
        )
        print(f"Individual Fisher information: {stat_str} = {fisher_information:.4f}")
        print('*'*100)
