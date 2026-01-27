from jaxpower import (
    MeshAttrs, ParticleField, FKPField,
    BinMesh2SpectrumPoles, get_mesh_attrs,
    compute_mesh2_spectrum, compute_fkp2_shotnoise,
    compute_box2_normalization
)
import jax
import logging
from .base import BaseDensityMeshEstimator


class PowerSpectrumMultipoles(BaseDensityMeshEstimator):
    """
    Calculate the power spectrum multipoles using jaxpower.
    https://github.com/adematti/jax-power/
    """
    
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        super().__init__(**kwargs)

        self.jitted_compute_mesh2_spectrum = jax.jit(
            compute_mesh2_spectrum,
            static_argnames=['los'],
            donate_argnums=[0]
        )

    def compute_spectrum(self, edges={'step': 0.001}, ells=(0, 2, 4), los='z', save_fn=None):
        self.bin = BinMesh2SpectrumPoles(
            self.mattrs,
            edges=edges,
            ells=ells,
        )

        norm = compute_box2_normalization(self.data_mesh, bin=self.bin)
        num_shotnoise = compute_fkp2_shotnoise(self.data_mesh, bin=self.bin)
        spectrum = self.jitted_compute_mesh2_spectrum(self.delta_mesh, bin=self.bin, los=los)
        self.spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

        if save_fn and jax.process_index() == 0:
            self.logger.info(f'Saving power spectrum to {save_fn}')
            self.spectrum.write(save_fn)
        return self.spectrum