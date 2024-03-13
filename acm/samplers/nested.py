from typing import Dict
import numpy as np
import pandas as pd
import dill
import dynesty
import dynesty.utils
dynesty.utils.pickle_module = dill
from .base import BasePosteriorSampler


class NestedSampler(BasePosteriorSampler):
    """Run nested sampling using dynesty"""

    def get_prior_from_cube(self, cube: np.array) -> np.array:
        """Transform a cube of uniform priors into the desired distribution

        Args:
            cube (np.array): uniform cube

        Returns:
            np.array: prior
        """
        transformed_cube = np.array(cube)
        for n, param in enumerate(self.param_names):
            transformed_cube[n] = self.priors[param].ppf(cube[n])
        return transformed_cube

    def get_loglikelihood_for_params(self, params: np.array) -> float:
        """Get loglikelihood for a set of parameters

        Args:
            params (np.array): input parameters

        Returns:
            float: log likelihood
        """
        # prediction = self.get_model_prediction(params)
        params = dict(zip(list(self.priors.keys()), params))
        for i, fixed_param in enumerate(self.fixed_parameters.keys()):
            params[fixed_param] = self.fixed_parameters[fixed_param]
        for key in params.keys():
            params[key] = [params[key]]
        return self.likelihood.log_likelihood(params)
        # return self.get_loglikelihood_for_prediction(
        #     prediction=prediction,
        # )

    def __call__(
        self,
        num_live_points: int = 500,
        dlogz: float = 0.01,
        max_iterations: int = 50_000,
        max_calls: int = 1_000_000,
    ):
        """Run nested sampling

        Args:
            num_live_points (int, optional): number of live points. Defaults to 500.
            dlogz (float, optional): allowed error on evidence. Defaults to 0.01.
            max_iterations (int, optional): maximum number of iterations. Defaults to 50_000.
            max_calls (int, optional): maximum number of calls. Defaults to 1_000_000.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sampler = dynesty.NestedSampler(
            self.get_loglikelihood_for_params,
            self.get_prior_from_cube,
            ndim=self.n_dim,
            nlive=num_live_points,
        )
        sampler.run_nested(
            checkpoint_file=str(self.output_dir / "dynasty.save"),
            dlogz=dlogz,
            maxiter=max_iterations,
            maxcall=max_calls,
        )
        results = sampler.results
        self.store_results(results)

    def store_results(self, results: Dict):
        """Store inference results

        Args:
            results (Dict): dictionary with chain and summary statistics
        """
        df = self.convert_results_to_df(results=results)
        df.to_csv(self.output_dir / "results.csv", index=False)

    def convert_results_to_df(self, results: Dict) -> pd.DataFrame:
        """Convert dynesty results to pandas dataframe

        Args:
            results (Dict): dynesty results

        Returns:
            pd.DataFrame: summarised results
        """
        log_like = results.logl
        log_weights = results.logwt
        log_evidence = results.logz
        log_evidence_err = results.logzerr
        samples = results.samples
        df = pd.DataFrame(
            {
                "log_likelihood": log_like,
                "log_weights": log_weights,
                "log_evidence": log_evidence,
                "log_evidence_err": log_evidence_err,
            }
        )
        for i, param in enumerate(self.priors):
            df[param] = samples[:, i]
        return df

    def get_results(
        self,
    ) -> pd.DataFrame:
        """
        Read results from file
        """
        return pd.read_csv(self.output_dir / "results.csv")