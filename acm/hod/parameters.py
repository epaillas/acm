import pandas
import numpy as np
from scipy.stats import qmc
from acm.utils.default import cosmo_list

class HODLatinHypercube:
    """
    Sample HOD parameters from a prior and distribute them on
    a Latin hypercube.

    Parameters
    ----------
    ranges : dict
        Dictionary with the prior ranges for each parameter.
    seed : int
        Seed for the random number generator.
    """
    def __init__(self, ranges, seed=42):
        self.ranges = ranges
        self.sampler = qmc.LatinHypercube(d=len(ranges), seed=seed)
        self.pmins = np.array([ranges[key][0] for key in ranges])
        self.pmaxs = np.array([ranges[key][1] for key in ranges])

    def sample(self, n: int, save_fn: str = None):
        """
        Sample HOD parameters from the prior.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        params : dict
            Dictionary with the sampled parameters.
        """
        # randomly sample n points in the unit hypercube
        params = self.sampler.random(n=n)
        # scale to the prior range and turn into dict
        params = self.pmins + params * (self.pmaxs - self.pmins)
        self.params = {key: list(params[:, i]) for i, key in enumerate(self.ranges)}
        self.is_split = False # Flag to indicate if params have been split by cosmology
        if save_fn:
            self.save_params(save_fn)
        return self.params

    def split_by_cosmo(self, cosmos: list = None, save_fn: list = None):
        """
        Split the sampled parameters by cosmology.

        Parameters
        ----------
        cosmos : list
            List of cosmologies to split the parameters for. If none are provided,
            the default AbacusSummit list of cosmologies is used
        save_fn : list
            List of filenames to save the split parameters to.

        Returns
        -------
        split_params : dict
            Dictionary with the split parameters.
        """
        if cosmos is None:
            cosmos = cosmo_list
        split_params = {}
        for i, cosmo in enumerate(cosmos):
            split = [np.array_split(self.params[key], len(cosmos))[i] for key in self.params]
            split_params[f'c{cosmo:03}'] = {key: list(split[i]) for i, key in enumerate(self.params)}
        self.params = split_params
        self.is_split = True
        if save_fn:
            if len(cosmos) != len(save_fn):
                raise ValueError('Number of filenames must match number of cosmologies.')
            for cosmo, save_fn in zip(cosmos, save_fn):
                self.save_params(save_fn, split_params[f'c{cosmo:03}'])
        return self.params

    def add_cosmo_params(self, cosmo_params: dict, save_fn: list = None):
        """
        Add cosmology parameters to HOD parameters for each key in self.params.

        Parameters
        ----------
        cosmo_params : dict
            Dictionary with parameters for each cosmology. Must have the same keys as self.params.
        save_fn : list, optional
            List of filenames to save the updated parameters to. If None, the parameters will not be saved. Defaults to None.
        
        Returns
        -------
        params : dict
            Dictionary with the updated parameters.
        """
        for key, hod in self.params.items():
            n_hod = len(hod[next(iter(hod))])  # number of HOD samples for this cosmology
            cosmo = {k: [v] * n_hod for k, v in cosmo_params[key].items()} # Repeat each cosmology parameter n_hod times
            cosmo.update(hod)
            self.params[key] = cosmo
        if save_fn:
            if len(self.params) != len(save_fn):
                raise ValueError('Number of filenames must match number of cosmologies.')
            for key, save_fn in zip(self.params, save_fn):
                self.save_params(save_fn, self.params[key])
        return self.params
    
    def enforce_ordering(self, param: list):
        """
        Enforce ordering of the keys in param, given as a list of keys.
        Not recommended, as any transformation of the resulting dictionary
        may not preserve the ordering. This is just a safeguard.

        Parameters
        ----------
        param : list
            list of keys to enforce ordering on.
        """
        if self.is_split:
            for key, hod in self.params.items():
                self.params[key] = {k: hod[k] for k in param if k in hod}
        else:
            self.params = {k: self.params[k] for k in param if k in self.params}
        return self.params

    def save_params(self, save_fn: str):
        """
        Save the sampled parameters to disk.

        Parameters
        ----------
        save_fn : str
            File to save the parameters to.
        """
        df = pandas.DataFrame(self.params)
        df.to_csv(save_fn, index=False, float_format='%.5f')


if __name__ == '__main__':
    from sunbird.inference.priors import Yuan23
    ranges = Yuan23().ranges

    lhc = HODLatinHypercube(ranges)
    params = lhc.sample(50_000)  # number of HOD variations

    params = lhc.split_by_cosmo(cosmos=[0, 1])
    lhc.save_params('./')