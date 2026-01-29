from typing import Any
import numpy.typing as npt


class BaseEstimator:
    """
    Base estimator class.
    """
    def __init__(self, backend: str = 'jaxpower', **kwargs: Any) -> None:
        self.backend_name = backend
        
        # Lazy import of backend classes to avoid forcing installation of all backends
        if backend == 'jaxpower':
            from .backends.jaxpower import JaxpowerBackend
            self.backend = JaxpowerBackend(**kwargs)
            self._JaxpowerBackend = JaxpowerBackend
        elif backend == 'pypower':
            from .backends.pypower import PypowerBackend
            self.backend = PypowerBackend(**kwargs)
            self._PypowerBackend = PypowerBackend
        elif backend == 'pyrecon':
            from .backends.pyrecon import PyreconBackend
            self.backend = PyreconBackend(**kwargs)
            self._PyreconBackend = PyreconBackend
        else:
            raise ValueError(f"Unknown backend '{backend}'. Available backends: 'jaxpower', 'pypower', 'pyrecon'")
        
    def read_density_contrast(self, positions: npt.NDArray, resampler: str = 'cic') -> npt.NDArray:
        """
        Get the density contrast at the input positions.
z
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
        if hasattr(self, '_JaxpowerBackend') and isinstance(self.backend, self._JaxpowerBackend):
            return self.backend.delta_mesh.read(positions, resampler=resampler)
        elif hasattr(self, '_PypowerBackend') and isinstance(self.backend, self._PypowerBackend):
            offset = self.boxcenter - self.boxsize/2.
            return self.backend.delta_mesh.readout(positions - offset, resampler=resampler)
        elif hasattr(self, '_PyreconBackend') and isinstance(self.backend, self._PyreconBackend):
            if resampler != 'cic':
                raise NotImplementedError('Pyrecon backend only supports CIC resampling.')
            return self.backend.delta_mesh.read_cic(positions)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the backend.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        attribute : any
            Attribute from the backend.
        """
        return self.backend.__getattribute__(name)
        