import numpy as np
import pandas
from scipy.stats import qmc

from acm.utils.default import cosmo_list


class HODLatinHypercube:
    """
    Sample HOD parameters from a prior and distribute them on
    a Latin hypercube.
    """

    def __init__(
        self,
        ranges,
        seed: int = 42,
        order: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        ranges : dict
            Dictionary with the prior ranges for each parameter.
        seed : int
            Seed for the random number generator.
        order : list, optional
            See save_params method for details.
        """
        self.ranges = ranges
        self.sampler = qmc.LatinHypercube(d=len(ranges), seed=seed)
        self.pmins = np.array([ranges[key][0] for key in ranges])
        self.pmaxs = np.array([ranges[key][1] for key in ranges])
        self.order = order

    def sample(self, n: int, save_fn: str | None = None) -> dict:
        """
        Sample HOD parameters from the prior.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        save_fn : str, optional
            Path to save the sampled parameters. If None, the parameters are not saved. Defaults to None.

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
        self.is_split = False  # Flag to indicate if params have been split by cosmology
        if save_fn:
            self.save_params(save_fn)
        return self.params

    def split_by_cosmo(
        self, cosmos: list | None = None, save_fn: list | None = None
    ) -> dict:
        """
        Split the sampled parameters by cosmology.

        Parameters
        ----------
        cosmos : list | None
            List of cosmologies to split the parameters for. If none are provided,
            the default AbacusSummit list of cosmologies is used
        save_fn : list | None
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
            split = [
                np.array_split(self.params[key], len(cosmos))[i] for key in self.params
            ]
            split_params[f"c{cosmo:03}"] = {
                key: list(split[i]) for i, key in enumerate(self.params)
            }
        self.params = split_params
        self.is_split = True
        if save_fn:
            self.save_params(save_fn)
        return self.params

    def add_cosmo_params(self, cosmo_params: dict, save_fn: list | None = None) -> dict:
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
            n_hod = len(
                hod[next(iter(hod))]
            )  # number of HOD samples for this cosmology
            cosmo = {
                k: [v] * n_hod for k, v in cosmo_params[key].items()
            }  # Repeat each cosmology parameter n_hod times
            cosmo.update(hod)
            self.params[key] = cosmo  # ty:ignore[invalid-assignment]
        if save_fn:
            self.save_params(save_fn)
        return self.params

    def save_params(
        self, save_fn: str | list[str], order: list[str] | None = None
    ) -> None:
        """
        Save the sampled parameters to disk.

        Parameters
        ----------
        save_fn : str|list[str]
            File to save the parameters to or list of files if params are split by cosmology.
        order : list[str], optional
            List of keys to enforce ordering on.
            If None, tries to access self.order; otherwise the default order of self.params is used.
            Any keys not in self.params will be ignored. Keys not in order will be dropped.
            Defaults to None.
        """
        if order is None:
            order = getattr(self, "order", None)  # on-the-fly attribute access

        if self.is_split:
            if len(self.params) != len(save_fn):
                raise ValueError(
                    "Number of filenames must match number of cosmologies."
                )
            for key, save_fn in zip(self.params, save_fn):
                df = pandas.DataFrame(self.params[key])
                df = df[[k for k in order if k in df.columns]] if order else df
                df.to_csv(save_fn, index=False, float_format="%.5f")
        else:
            df = pandas.DataFrame(self.params)
            df = df[[k for k in order if k in df.columns]] if order else df
            df.to_csv(save_fn, index=False, float_format="%.5f")


if __name__ == "__main__":
    from sunbird.inference.priors import Yuan23

    ranges = Yuan23().ranges

    lhc = HODLatinHypercube(ranges)
    params = lhc.sample(50_000)  # number of HOD variations
    params = lhc.split_by_cosmo(cosmos=[0, 1])  # split by cosmology

    lhc.save_params(
        save_fn=["hod_params_c{cosmo:03}.csv".format(cosmo=cosmo) for cosmo in [0, 1]],
        order=[
            "logM_cut",
            "logM_1",
            "sigma",
            "alpha",
            "kappa",
            "alpha_c",
            "alpha_s",
            "s",
            "A_cen",
            "A_sat",
            "B_cen",
            "B_sat",
        ],
    )
