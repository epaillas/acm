from jaxpower import MeshAttrs, ParticleField, FKPField, ComplexMeshField, RealMeshField, get_mesh_attrs
from jax import numpy as jnp
import jax
import numpy as np
import time

from .backends import JaxpowerBackend, PypowerBackend


_BACKENDS = {
    "jaxpower": JaxpowerBackend,
    "pypower": PypowerBackend,
}

class BaseEstimator:
    """
    Base estimator class.
    """
    def __init__(self, backend='jaxpower', **kwargs):
        self.backend_name = backend
        self.backend = _BACKENDS[backend](**kwargs)

    def set_density_contrast(self, **kwargs):
        """
        Set the density contrast on the rectangular mesh
        """
        self.backend.set_density_contrast(**kwargs)

    def read_density_contrast(self, positions, resampler='cic'):
        """
        Get the density contrast at the input positions.

        Parameters
        ----------
        positions : array_like
            Input positions.

        Returns
        -------
        delta_query : array_like
            Density contrast at the input positions.
        """
        if self.backend.name == 'jaxpower':
            return self.backend.delta_mesh.read(positions, resampler=resampler).reshape(self.backend.meshsize)
        elif self.backend.name == 'pypower':
            return self.backend.delta_mesh.readout(positions, resampler=resampler).reshape(self.backend.meshsize)
        