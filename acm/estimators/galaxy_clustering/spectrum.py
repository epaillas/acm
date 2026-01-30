from jaxpower import (
    MeshAttrs, ParticleField, FKPField,
    BinMesh2SpectrumPoles, get_mesh_attrs,
    compute_mesh2_spectrum, compute_fkp2_shotnoise,
    compute_box2_normalization
)
import jax
import logging
from typing import Any, Optional, Union
from .base import BaseEstimator


class PowerSpectrumMultipoles(BaseEstimator):
    """
    Calculate the power spectrum multipoles using jaxpower.
    https://github.com/adematti/jax-power/
    """
    
    def __init__(self, **kwargs: Any) -> None:
        self.logger = logging.getLogger(__name__)
        kwargs.setdefault('backend', 'jaxpower')
        super().__init__(**kwargs)
        
        if not isinstance(self.backend, self._JaxpowerBackend):
            raise ValueError("PowerSpectrumMultipoles only supports the 'jaxpower' backend.")

        self.jitted_compute_mesh2_spectrum = jax.jit(
            compute_mesh2_spectrum,
            static_argnames=['los'],
            donate_argnums=[0]
        )

    def compute_spectrum(self, edges: dict = {'step': 0.001}, ells: tuple = (0, 2, 4), los: str = 'z', save_fn: Optional[str] = None):
        self.bin = BinMesh2SpectrumPoles(
            self.mattrs,
            edges=edges,
            ells=ells,
        )
        norm = compute_box2_normalization(self.data_mesh, bin=self.bin)
        num_shotnoise = compute_fkp2_shotnoise(self.data_mesh, bin=self.bin)
        spectrum = self.jitted_compute_mesh2_spectrum(self.delta_mesh * self.mean, bin=self.bin, los=los)
        self.spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

        if save_fn and jax.process_index() == 0:
            self.logger.info(f'Saving power spectrum to {save_fn}')
            self.spectrum.write(save_fn)
        return self.spectrum
    
    def get_multipoles(self, kmin: Optional[float] = None, kmax: Optional[float] = None, rebin: int = 1):
        spectrum = self.spectrum.select(k=slice(0, None, rebin))
        poles = [spectrum.get(ell) for ell in spectrum.ells]
        k = poles[0].coords('k')
        poles = [pole.value() for pole in poles]
        return k, poles