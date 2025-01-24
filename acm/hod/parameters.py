from scipy.stats import qmc
import numpy as np
from pathlib import Path


class HODLatinHypercube:
    """
    Sample HOD parameters from a prior and distribute them on
    a Latin hypercube.

    Parameters

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

    def sample(self, n, save_fn=None):
        """
        Sample HOD parameters from the prior.

        Parameters

        n : int
            Number of samples to draw.

        Returns

        params : dict
            Dictionary with the sampled parameters.
        """
        # randomly sample n points in the unit hypercube
        params = self.sampler.random(n=n)
        # scale to the prior range and turn into dict
        params = self.pmins + params * (self.pmaxs - self.pmins)
        self.params = {key: list(params[:, i]) for i, key in enumerate(self.ranges)}
        if save_fn:
            self.save_params(save_fn)
        return self.params

    def split_by_cosmo(self, cosmos=None, save_fn=None):
        """
        Split the sampled parameters by cosmology.

        Parameters

        cosmos : list
            List of cosmologies to split the parameters for. If none are provided,
            the default AbacusSummit list of cosmologies is used
        save_fn : list
            List of filenames to save the split parameters to.

        Returns

        split_params : dict
            Dictionary with the split parameters.
        """
        if cosmos is None:
            cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
        split_params = {}
        for i, cosmo in enumerate(cosmos):
            split = [np.array_split(self.params[key], len(cosmos))[i] for key in self.params]
            split_params[f'c{cosmo:03}'] = {key: list(split[i]) for i, key in enumerate(self.params)}
        self.params = split_params
        if save_fn:
            if len(cosmos) != len(save_fn):
                raise ValueError('Number of filenames must match number of cosmologies.')
            for cosmo, save_fn in zip(cosmos, save_fn):
                self.save_params(save_fn, split_params[f'c{cosmo:03}'])
        return self.params

    def save_params(self, filename, save_fn=None):
        """
        Save the sampled parameters to disk.

        Parameters

        filename : str
            Directory to save the parameters to.
        """
        import pandas
        df = pandas.DataFrame(self.params)
        df.to_csv(filename, index=False, float_format='%.5f')

if __name__ == '__main__':
    from sunbird.inference.ranges import Yuan23
    ranges = Yuan23().ranges

    lhc = HODLatinHypercube(ranges)
    params = lhc.sample(50_000)  # number of HOD variations

    params = lhc.split_by_cosmo(cosmos=[0, 1])
    lhc.save_params('./')