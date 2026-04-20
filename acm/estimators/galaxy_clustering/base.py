import logging
import time
from pathlib import Path

import lsstypes as types
import numpy as np
import numpy.typing as npt
import xarray as xr

logger = logging.getLogger(__name__)


class BaseEstimator:
    """
    Base estimator class.
    """

    def __init__(self, backend: str = "jaxpower", **kwargs) -> None:
        logger.info(f"Initializing {self.__class__.__name__}.")
        # Lazy import of backend classes to avoid forcing installation of all backends
        if backend == "jaxpower":
            from .backends.jaxpower import JaxpowerBackend

            self.backend = JaxpowerBackend(**kwargs)
        elif backend == "pypower":
            from .backends.pypower import PypowerBackend

            self.backend = PypowerBackend(**kwargs)
        elif backend == "pyrecon":
            from .backends.pyrecon import PyreconBackend

            self.backend = PyreconBackend(**kwargs)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Available backends: 'jaxpower', 'pypower', 'pyrecon'"
            )

    def read_density_contrast(
        self, positions: npt.NDArray, resampler: str = "cic"
    ) -> npt.NDArray:
        """
        Get the density contrast at the input positions.

        Parameters
        ----------
        positions : array_like
            Input positions.
        resampler : str, optional
            Resampling scheme. Default is 'cic'.

        Returns
        -------
        delta : array_like
            Density contrast at the input positions.
        """
        t0 = time.time()
        if self.backend.name == "jaxpower":
            delta = self.backend.delta_mesh.read(positions, resampler=resampler)
        elif self.backend.name == "pypower":
            offset = self.boxcenter - self.boxsize / 2.0
            delta = self.backend.delta_mesh.readout(
                positions - offset, resampler=resampler
            )
        elif self.backend.name == "pyrecon":
            if resampler != "cic":
                raise NotImplementedError(
                    "Pyrecon backend only supports CIC resampling."
                )
            delta = self.backend.delta_mesh.read_cic(positions)
        logger.info(f"Read density contrast in {time.time() - t0:.2f} s.")
        return delta

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the backend.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        attribute
            Attribute from the backend.
        """
        return self.backend.__getattribute__(name)

    @staticmethod
    def read(filename: str | Path, **kwargs):
        """
        Read estimator output from a file.

        Format is automatically determined from file extension:
        - .hdf5, .h5 -> lsstypes
        - .nc -> xarray NetCDF
        - .zarr -> xarray Zarr
        - .npy -> numpy

        Parameters
        ----------
        filename : str | Path
            Path to the saved file.
        **kwargs
            Additional keyword arguments for the specific file format reader.

        Returns
        -------
        data : ObservableLeaf, xarray.DataArray, xarray.Dataset, or numpy.ndarray
            The loaded data.
            - lsstypes (.hdf5, .h5): ObservableLeaf object with .value() and .coords() methods
            - xarray NetCDF (.nc): DataArray with .values and coordinate attributes
            - xarray Zarr (.zarr): DataArray (if data_var specified) or Dataset
            - numpy (.npy): plain ndarray

        Examples
        --------
        >>> data = MyEstimator.read('output.hdf5')  # lsstypes format
        >>> data = MyEstimator.read('output.nc')     # xarray NetCDF format
        >>> data = MyEstimator.read('output.zarr', data_var='wst')  # xarray Zarr with specific variable
        >>> coeffs = MyEstimator.read('output.npy')  # numpy array

        Raises
        ------
        ValueError
            If the file extension is not recognized.
        """
        path = Path(filename)

        # Determine format from file extension
        if path.suffix in [".hdf5", ".h5"]:
            return types.read(filename)
        if path.suffix == ".nc":
            return xr.open_dataarray(filename, **kwargs)
        if str(filename).endswith(".zarr"):
            return xr.open_zarr(filename, **kwargs)
        if path.suffix == ".npy":
            return np.load(filename, **kwargs)
        raise ValueError(
            f"Unrecognized file extension '{path.suffix}' for file: {filename}. "
            f"Supported extensions: .hdf5, .h5 (lsstypes), .nc (xarray NetCDF), "
            f".zarr (xarray Zarr), .npy (numpy)."
        )
