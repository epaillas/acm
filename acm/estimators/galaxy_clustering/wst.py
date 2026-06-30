import logging
import time
from pathlib import Path
from typing import Any

import lsstypes
import matplotlib.pyplot as plt
import numpy as np
from kymatio import HarmonicScattering3D

from acm.typing import LsstypeObject

from .backends.base import EstimatorBackend
from .base import BaseEstimator

# NOTE: 2 lazy imports in _torch and _jax to avoid unnecessary imports if not using those backends
logger = logging.getLogger(__name__)


class WaveletScatteringTransform(BaseEstimator):
    """Class to compute the wavelet scattering transform with :mod:`kymatio`."""

    def __init__(
        self,
        backend: str | EstimatorBackend,
        data_positions: np.ndarray,
        randoms_positions: np.ndarray | None = None,
        data_weights: np.ndarray | None = None,
        randoms_weights: np.ndarray | None = None,
        J: int = 4,  # noqa: N803
        L: int = 4,  # noqa: N803
        sigma: float = 0.8,
        frontend: str = "torch",  # FIXME: is this the backend of frontend ? Docs imply frontend but TBC w/ Georgios
        kymatio_object: Any | None = None,  # noqa: ANN401
        **kwargs,
    ) -> None:
        """
        Initialize the WaveletScatteringTransform estimator.

        Parameters
        ----------
        backend: str | EstimatorBackend
            The backend to use for the estimator.
        data_positions: np.ndarray
            Positions of data galaxies, of shape (N, 3).
        randoms_positions: np.ndarray, optional
            Positions of random catalog, of shape (M, 3).
        data_weights: np.ndarray, optional
            Weights for data galaxies, of shape (N,).
        randoms_weights: np.ndarray, optional
            Weights for randoms, of shape (M,).
        J: int, optional
            Scale of the scattering transform. Default is 4.
        L: int, optional
            Number of angles for the scattering transform. Default is 4.
        sigma: float, optional
            Standard deviation of the Gaussian window for the scattering transform. Default is 0.8.
        frontend: str, optional
            The frontend to use for the scattering transform. Default is "torch".
        kymatio_object: Any, optional
            Pre-loaded Kymatio HarmonicScattering3D object. If provided, it will be used instead of initializing a new one. Default is None.
        **kwargs
            Additional keyword arguments for the backend initialization.
        """
        super().__init__(
            backend,
            data_positions,
            randoms_positions,
            data_weights,
            randoms_weights,
            **kwargs,
        )

        if kymatio_object is not None:
            logger.info("Using pre-loaded Kymatio object.")
            S = kymatio_object
        else:
            logger.info("Initializing WaveletScatteringTransform.")
            S = self.initialize_kymatio(
                J=J,
                L=L,
                sigma_0=sigma,
                shape=self.backend.meshsize,
                frontend=frontend,  # NOTE: dynamic frontend/backend selection
                # FIXME: do we want to pass extra kwargs to kymatio here ? e.g., max_order, rotation_covariant, etc. They need to be explicit.
            )

        logger.info(
            f"Initialized HarmonicScattering3D ({S.backend}) with J={S.J}, L={S.L}, sigma_0={S.sigma_0}, max_order={S.max_order}."
        )

        # Register private attributes
        self._S = S

    @staticmethod
    def initialize_kymatio(
        J: int,  # noqa: N803
        shape: list[int] | tuple[int, ...] | np.ndarray,
        **kwargs
    ) -> HarmonicScattering3D:
        """
        Initialize the Kymatio HarmonicScattering3D object.

        Parameters
        ----------
        J: int
            Scale of the scattering transform.
        shape: list[int] | tuple[int, ...] | np.ndarray
            Shape of the input data.
        **kwargs
            Additional keyword arguments for the HarmonicScattering3D initialization, including 'frontend'.
            See :class:`~kymatio.scattering3d.HarmonicScattering3D` for details.

        Returns
        -------
        S: HarmonicScattering3D
            The initialized Kymatio HarmonicScattering3D object. Can be saved as a pickled object for later use.
        """
        t0 = time.time()
        S = HarmonicScattering3D(
            J=J,
            shape=shape,
            **kwargs,
        )
        logger.info(f"Initialized Kymatio in {time.time() - t0:.2f} s.")
        return S

    def compute(
        self,
        q: float = 0.8,
        method: str = "lattice",
        resampler: str = 'cic',
        **kwargs,
    ) -> lsstypes.ObservableLeaf:
        """
        Compute the wavelet scattering transform coefficients from the density contrast field.

        Parameters
        ----------
        q: float, optional
            The exponent for the L^q norm. Default is 0.8.
        method: str, optional
            The method for querying positions. Default is "lattice".
        resampler: str, optional
            The resampling method for reading density contrast. Default is 'cic'.
        **kwargs
            Additional keyword arguments for the density contrast computation.
            See :func:`~acm.estimators.galaxy_clustering.backends.base.EstimatorBackend.read_density_contrast` for details.

        Returns
        -------
        leaf: lsstypes.ObservableLeaf
            An :class:`~lsstypes.ObservableLeaf` object containing the WST coefficients and associated metadata.
        """
        query_positions = self.backend.get_query_positions(method=method, **kwargs)
        density_contrast = self.backend.read_density_contrast(query_positions, resampler=resampler)

        density_contrast = density_contrast.reshape(self.backend.meshsize)

        t0 = time.time()
        logger.info("Computing wavelet scattering transform.")
        if not hasattr(self, f"_{self._S.backend}"):
            raise ValueError(f"Unsupported Kymatio backend: {self._S.backend}")
        _callable = getattr(self, f"_{self._S.backend}")
        smatavg: np.ndarray = _callable(density_contrast, q)
        logger.info(f"Computed WST coefficients in {time.time() - t0:.2f} s.")

        attrs = dict(  # FIXME: Choose which attributes to keep !
            J=self._S.J,
            L=self._S.L,
            sigma_0=self._S.sigma_0,
            max_order=self._S.max_order,
            frontend=self._S.backend,  # frontend usually matches backend in kymatio
            q=q,
            boxsize=list(self.backend.boxsize),
            boxcenter=list(self.backend.boxcenter),
            meshsize=list(self.backend.meshsize),
        )
        leaf = lsstypes.ObservableLeaf(
            coefficients=smatavg,
            index=np.arange(len(smatavg)),
            coords=["index"],
            attrs=attrs,
        )
        return leaf

    def _torch(self, density_contrast: np.ndarray, q: float) -> np.ndarray:
        """Run the wavelet scattering transform with Torch backend."""
        import torch  # noqa: PLC0415
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._S.to(device)
        dc_torch = torch.from_numpy(density_contrast, dtype=torch.float32).to(device)

        s0 = torch.sum(torch.abs(dc_torch) ** q)
        smat_orders_12 = self._S(dc_torch)
        smat = torch.abs(smat_orders_12[:, :, 0]).flatten()
        smatavg = torch.cat([s0.unsqueeze(0), smat])
        smatavg /= np.prod(self.backend.meshsize)
        return smatavg.cpu().numpy()

    def _jax(self, density_contrast: np.ndarray, q: float) -> np.ndarray:
        """Run the wavelet scattering transform with JAX backend."""
        import jax.numpy as jnp  # noqa: PLC0415

        s0 = jnp.sum(jnp.abs(density_contrast) ** q)
        smat_orders_12 = self._S(density_contrast)
        smat = jnp.abs(smat_orders_12[:, :, 0]).flatten()
        smatavg = jnp.concatenate([jnp.array([s0]), smat])
        smatavg /= np.prod(self.backend.meshsize)
        return np.asarray(smatavg)

    @staticmethod
    def load(filename: str | Path) -> lsstypes.ObservableLeaf:
        """Load a :class:`~lsstypes.ObservableLeaf` object with WST coefficients from file."""
        obj: lsstypes.ObservableLeaf = lsstypes.read(filename)
        return obj

    @staticmethod
    def plot(obj: LsstypeObject, **kwargs) -> tuple:
        """
        Plot the wavelet scattering transform coefficients from a :class:`~lsstypes.ObservableLeaf` object.

        Parameters
        ----------
        obj: lsstypes.ObservableLeaf
            The :class:`~lsstypes.ObservableLeaf` object containing the WST coefficients to plot.
        **kwargs
            Additional keyword arguments for the plot. See :func:`matplotlib.pyplot.subplots` for details.

        Returns
        -------
        tuple
            A tuple containing the figure and axes objects of the plot.
        """
        fig = kwargs.pop("fig", None)
        ax = kwargs.pop("ax", None)
        figsize = kwargs.pop("figsize", (8, 6))
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
            ax.set_xlabel("WST Coefficient index")
            ax.set_ylabel("WST Coefficient value")
        ax.plot(obj.index, obj.coefficients, **kwargs)
        return fig, ax
