"""Utils module containing the LatinHyperCube sampler for HOD parameters."""
import numpy as np
import pandas as pd
from scipy.stats import qmc


class LatinHyperCubeSampler:
    """Sample a Latin Hypercube in a given range."""

    def __init__(self, ranges: dict, seed: int = 42) -> None:
        self.sampler = qmc.LatinHypercube(d=len(ranges), seed=seed)
        self.pmins = np.array([ranges[key][0] for key in ranges])
        self.pmaxs = np.array([ranges[key][1] for key in ranges])
        self.ranges = ranges

    def sample(self, n: int) -> pd.DataFrame:
        """Sample n points in the Latin Hypercube and scale to the given ranges."""
        params = self.sampler.random(n=n)
        params = self.pmins + params * (self.pmaxs - self.pmins)
        return pd.DataFrame(params, columns=list(self.ranges))  # ty: ignore[invalid-argument-type]

    @staticmethod
    def split(sample: pd.DataFrame, keys: list) -> dict:
        """Split the sampled parameters in quantiles set by the keys."""
        pass
