import importlib
from typing import cast

import numpy as np
import pandas
from scipy.stats import qmc

from acm.utils.default import cosmo_list

Numeric = int | float
ParameterTable = dict[str, list[Numeric]]
SplitParameterTable = dict[str, ParameterTable]


class HODLatinHypercube:
    """
    Sample HOD parameters from a prior and distribute them on
    a Latin hypercube.
    """

    def __init__(
        self, ranges, seed: int = 42, order: list[str] | None = None
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
        self.params: ParameterTable | SplitParameterTable = {}
        self.is_split = False

    def sample(self, n: int, save_fn: str | None = None) -> ParameterTable:
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
        self.is_split = False  # Flag to indicate if params have been split by cosmology
        if save_fn:
            self.save_params(save_fn)
        return self.params

    def split_by_cosmo(
        self, cosmos: list[int] | None = None, save_fn: list[str] | None = None
    ) -> SplitParameterTable:
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
        params = cast(ParameterTable, self.params)
        split_params: SplitParameterTable = {}
        for i, cosmo in enumerate(cosmos):
            split = [
                np.array_split(params[key], len(cosmos))[i] for key in params
            ]
            split_params[f"c{cosmo:03}"] = {
                key: list(split[i]) for i, key in enumerate(params)
            }
        self.params = split_params
        self.is_split = True
        if save_fn:
            self.save_params(save_fn)
        return self.params

    def add_cosmo_params(
        self, cosmo_params: dict[str, dict[str, float]], save_fn: list[str] | None = None
    ) -> SplitParameterTable:
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
        if not self.is_split:
            raise ValueError("Parameters must be split by cosmology before adding cosmology parameters.")

        params = cast(SplitParameterTable, self.params)
        for key, hod in params.items():
            n_hod = len(
                hod[next(iter(hod))]
            )  # number of HOD samples for this cosmology
            cosmo: ParameterTable = {
                k: [v] * n_hod for k, v in cosmo_params[key].items()
            }  # Repeat each cosmology parameter n_hod times
            cosmo.update(hod)
            params[key] = cosmo
        self.params = params
        if save_fn:
            self.save_params(save_fn)
        return params

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
            order = self.order

        if self.is_split:
            if not isinstance(save_fn, list):
                raise ValueError(
                    "A list of filenames is required when parameters are split by cosmology."
                )
            if len(self.params) != len(save_fn):
                raise ValueError(
                    "Number of filenames must match number of cosmologies."
                )
            for key, save_fn in zip(self.params, save_fn):
                df = pandas.DataFrame(self.params[key])
                df = df[[k for k in order if k in df.columns]] if order else df
                df.to_csv(save_fn, index=False, float_format="%.5f")
        else:
            if isinstance(save_fn, list):
                raise ValueError(
                    "A single filename is required when parameters are not split by cosmology."
                )
            df = pandas.DataFrame(self.params)
            df = df[[k for k in order if k in df.columns]] if order else df
            df.to_csv(save_fn, index=False, float_format="%.5f")


if __name__ == "__main__":
    Yuan23 = importlib.import_module("sunbird.inference.priors").Yuan23

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
