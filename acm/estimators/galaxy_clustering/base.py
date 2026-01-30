import logging

import numpy.typing as npt


class BaseEstimator:
    """
    Base estimator class.
    """
    def __init__(self, backend: str = 'jaxpower', **kwargs) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f'Initializing {self.__class__.__name__}.')
        
        # Lazy import of backend classes to avoid forcing installation of all backends
        if backend == 'jaxpower':
            from .backends.jaxpower import JaxpowerBackend
            self.backend = JaxpowerBackend(**kwargs)
        elif backend == 'pypower':
            from .backends.pypower import PypowerBackend
            self.backend = PypowerBackend(**kwargs)
        elif backend == 'pyrecon':
            from .backends.pyrecon import PyreconBackend
            self.backend = PyreconBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Available backends: 'jaxpower', 'pypower', 'pyrecon'")
        
    def read_density_contrast(self, positions: npt.NDArray, resampler: str = 'cic') -> npt.NDArray:
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
        if self.backend.name == 'jaxpower':
            return self.backend.delta_mesh.read(positions, resampler=resampler)
        elif self.backend.name == 'pypower':
            offset = self.boxcenter - self.boxsize/2.
            return self.backend.delta_mesh.readout(positions - offset, resampler=resampler)
        elif self.backend.name == 'pyrecon':
            if resampler != 'cic':
                raise NotImplementedError('Pyrecon backend only supports CIC resampling.')
            return self.backend.delta_mesh.read_cic(positions)

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
        