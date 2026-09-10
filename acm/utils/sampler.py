"""Utils module containing the LatinHyperCube sampler for HOD parameters."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


class LatinHyperCubeSampler:
    """Sample a Latin Hypercube in a given range."""

    def __init__(self, ranges: dict[str, tuple[float, float]], seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.sampler = qmc.LatinHypercube(d=len(ranges), rng=rng)
        self.pmins = np.array([ranges[key][0] for key in ranges])
        self.pmaxs = np.array([ranges[key][1] for key in ranges])
        self.ranges = ranges

    def sample(self, n: int) -> pd.DataFrame:
        """Sample n points in the Latin Hypercube and scale to the given ranges."""
        params = self.sampler.random(n=n)
        params = self.pmins + params * (self.pmaxs - self.pmins)
        return pd.DataFrame(params, columns=list(self.ranges))

    @staticmethod
    def split(sample: pd.DataFrame, keys: list[str]) -> dict[str, pd.DataFrame]:
        """Split the sampled parameters in quantiles set by the keys."""
        n_splits = len(keys)
        splits_arr = np.array_split(sample, n_splits)
        splits_df = [pd.DataFrame(arr, columns=sample.columns) for arr in splits_arr]
        return {keys[i]: splits_df[i] for i in range(n_splits)}

    @staticmethod
    def add_columns(sample: pd.DataFrame, extra_params: pd.DataFrame) -> pd.DataFrame:
        """Add the extra parameters to each row of the sampled parameters."""
        nrows = sample.shape[0]
        extra_df = pd.DataFrame(
            {k: np.repeat(v, nrows) for k, v in extra_params.items()}
        )
        extra_df = extra_df.reset_index(drop=True)  # All rows have index 0
        return pd.concat([sample, extra_df], axis=1)

    @staticmethod
    def save(
        sample: pd.DataFrame | dict[str, pd.DataFrame],
        save_fn: str | Path,
        order: list[str] | None = None,
    ) -> None:
        """Save the sampled parameters to a CSV file."""

        def _save(df: pd.DataFrame, fn: str | Path) -> None:
            if order is not None:
                df = df[order]
            Path(fn).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(fn, index=False, float_format="%.5f")

        if isinstance(sample, pd.DataFrame):
            _save(sample, save_fn)
        elif isinstance(sample, dict):
            for key, df in sample.items():
                fn = str(save_fn).format(key=key)
                _save(df, fn)
