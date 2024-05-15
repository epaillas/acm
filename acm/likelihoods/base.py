import numpy as np
import torch
from collections.abc import Iterable


class BaseGaussianLikelihood(object):
    """
    Base class for Gaussian likelihoods.
    """
    def __init__(self):
        pass

    def log_likelihood(self, theta):
        # print(np.shape(self.flatdata), np.shape(self.flattheory(theta)), np.shape(self.inverse_covariance))
        flatdiff = self.flatdata - self.flattheory(theta)
        return -0.5 * flatdiff @ self.inverse_covariance @ flatdiff


class ObservablesGaussianLikelihood(BaseGaussianLikelihood):
    """
    Gaussian likelihood for observables.
    """

    def __init__(self, observables, covariance=None):
        if not isinstance(observables, Iterable):
            observables = [observables]
        self.observables = observables
        self.covariance = covariance
        self.inverse_covariance = np.linalg.inv(self.covariance)

        self.flatdata = np.concatenate([obs.data for obs in self.observables], axis=0)
        super().__init__()

    def flattheory(self, theta):
        # return np.concatenate([obs.theory.predictions_np(theta)[0] for obs in self.observables], axis=0)

        with torch.no_grad():
            pred_test_y = [obs.theory.get_prediction(torch.Tensor(theta)) for obs in self.observables]
            return [pred_test_y[i].numpy() for i in range(len(pred_test_y))]
            # pred_test_y = pred_test_y.numpy()
    
    