import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

def is_well_conditioned(matrix, threshold=1e10):
    condition_number = np.linalg.cond(matrix)
    return condition_number < threshold, condition_number

def safe_inverse(matrix,): 
    """Safely invert a potentially ill-conditioned matrix"""
    is_stable, cond_num = is_well_conditioned(matrix)
    if is_stable:
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            print("Regular inversion failed despite good condition number")
            pass
    print("Falling back to pseudo-inverse")
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
    print("Falling back to pseudo-determinant")
    return get_pseudodeterminant(matrix, epsilon)


def get_gradient(statistic):
    fiducial_parameters = statistic.lhc_x
    fiducial_parameters = torch.tensor(fiducial_parameters.astype(np.float32), requires_grad=True,).unsqueeze(0)
    def model_fn(x_batch):
        return statistic.model.get_prediction(x_batch)
    return torch.func.jacrev(model_fn)(fiducial_parameters).detach().squeeze().numpy()


def get_fisher_information(statistic,):
    small_box_y = statistic.small_box_y
    covariance_matrix = np.cov(small_box_y.T)
    correction = statistic.get_covariance_correction(
        n_s = len(small_box_y),
        n_d = len(covariance_matrix),
        n_theta = 20,
        method='percival',
    )
    precision_matrix = safe_inverse(correction * covariance_matrix)
    gradients = get_gradient(statistic)
    fisher_matrix = np.dot(gradients.T, np.dot(precision_matrix, gradients))
    return safe_log_determinant(fisher_matrix)