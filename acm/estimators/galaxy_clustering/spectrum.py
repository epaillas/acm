from typing import Optional

import jax
from jaxpower import FKPField, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum, compute_fkp2_shotnoise, compute_fkp2_normalization, compute_box2_normalization

from .base import BaseEstimator


class PowerSpectrumMultipoles(BaseEstimator):
    """
    Calculate the power spectrum multipoles using jaxpower.
    https://github.com/adematti/jax-power/
    """
    
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault('backend', 'jaxpower')
        super().__init__(**kwargs)
        
        if self.backend.name != 'jaxpower':
            raise ValueError("PowerSpectrumMultipoles only supports the 'jaxpower' backend.")

        self.jitted_compute_mesh2_spectrum = jax.jit(
            compute_mesh2_spectrum,
            static_argnames=['los'],
            donate_argnums=[0]
        )

    def compute_spectrum(self, edges: dict = {'step': 0.001}, ells: tuple = (0, 2, 4), los: str = 'z',
            interlacing: int = 3, compensate: bool = True, resampler='tsc', save_fn: Optional[str] = None):
        """
        Calculate the power spectrum multipoles.
        
        If randoms are provided (self.has_randoms=True), uses FKP estimator with 'firstpoint' line-of-sight.
        Otherwise, uses box normalization with specified line-of-sight.
        
        Parameters
        ----------
        edges : dict, optional
            Binning specification for the spectrum. Default is {'step': 0.001}.
        ells : tuple, optional
            Multipole orders to compute. Default is (0, 2, 4).
        los : str, optional
            Line-of-sight direction. Default is 'z'. Used only when has_randoms=False.
            When has_randoms=True, 'firstpoint' is always used.
        interlacing : int, optional
            Interlacing factor to reduce aliasing. Default is 3.
        compensate : bool, optional
            Whether to compensate for the mass assignment scheme. Default is True.
        resampler : str, optional
            Resampling scheme to use for mesh painting. Default is 'tsc'.
        save_fn : str, optional
            Path to save the computed spectrum. Saves only on process 0.
            
        Returns
        -------
        spectrum
            Computed power spectrum multipoles.
        """
        self.bin = BinMesh2SpectrumPoles(
            self.mattrs,
            edges=edges,
            ells=ells,
        )
        
        if self.has_randoms:
            self.logger.info("Computing power spectrum using FKP estimator with randoms.")
            fkp = FKPField(self.data_mesh, self.randoms_mesh)
            norm = compute_fkp2_normalization(fkp, bin=self.bin)
            num_shotnoise = compute_fkp2_shotnoise(fkp, bin=self.bin)
            delta_mesh = fkp.paint(resampler=resampler, interlacing=interlacing, compensate=compensate, out='real')
            spectrum = self.jitted_compute_mesh2_spectrum(delta_mesh, bin=self.bin, los='firstpoint')
        else:
            self.logger.info("Computing power spectrum using box normalization without randoms.")
            norm = compute_box2_normalization(self.data_mesh, bin=self.bin)
            num_shotnoise = compute_fkp2_shotnoise(self.data_mesh, bin=self.bin)
            delta_mesh = self.data_mesh.paint(resampler=resampler, interlacing=interlacing, compensate=compensate, out='real')
            delta_mesh -= delta_mesh.mean()
            spectrum = self.jitted_compute_mesh2_spectrum(delta_mesh, bin=self.bin, los=los)
        
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