import logging
import time
from pathlib import Path
from typing import Optional

import jax
from jaxpower import (
    BinMesh2SpectrumPoles,
    FKPField,
    compute_box2_normalization,
    compute_fkp2_normalization,
    compute_fkp2_shotnoise,
    compute_mesh2_spectrum,
)

from .base import BaseEstimator

logger = logging.getLogger(__name__)


class PowerSpectrumMultipoles(BaseEstimator):
    """
    Calculate the power spectrum multipoles using jaxpower.
    https://github.com/adematti/jax-power/
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("backend", "jaxpower")
        super().__init__(**kwargs)

        if self.backend.name != "jaxpower":
            raise ValueError(
                "PowerSpectrumMultipoles only supports the 'jaxpower' backend."
            )

        self.jitted_compute_mesh2_spectrum = jax.jit(
            compute_mesh2_spectrum, static_argnames=["los"], donate_argnums=[0]
        )

    def compute_spectrum(
        self,
        edges: dict = {"step": 0.001},
        ells: tuple = (0, 2, 4),
        los: str = "z",
        interlacing: int = 3,
        compensate: bool = True,
        resampler="tsc",
        save_fn: Optional[str] = None,
    ):
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
        t0 = time.time()
        self.bin = BinMesh2SpectrumPoles(
            self.mattrs,
            edges=edges,
            ells=ells,
        )

        if self.has_randoms:
            logger.info("Computing power spectrum using FKP estimator with randoms.")
            fkp = FKPField(self.data_mesh, self.randoms_mesh)
            norm = compute_fkp2_normalization(fkp, bin=self.bin)
            num_shotnoise = compute_fkp2_shotnoise(fkp, bin=self.bin)
            delta_mesh = fkp.paint(
                resampler=resampler,
                interlacing=interlacing,
                compensate=compensate,
                out="real",
            )
            spectrum = self.jitted_compute_mesh2_spectrum(
                delta_mesh, bin=self.bin, los="firstpoint"
            )
        else:
            logger.info(
                "Computing power spectrum using box normalization without randoms."
            )
            norm = compute_box2_normalization(self.data_mesh, bin=self.bin)
            num_shotnoise = compute_fkp2_shotnoise(self.data_mesh, bin=self.bin)
            delta_mesh = self.data_mesh.paint(
                resampler=resampler,
                interlacing=interlacing,
                compensate=compensate,
                out="real",
            )
            delta_mesh -= delta_mesh.mean()
            spectrum = self.jitted_compute_mesh2_spectrum(
                delta_mesh, bin=self.bin, los=los
            )

        self.spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

        if save_fn:
            self.save(save_fn)

        logger.info(f"Power spectrum computed in {time.time() - t0:.2f} s.")
        return self.spectrum

    def get_multipoles(
        self,
        kmin: Optional[float] = None,
        kmax: Optional[float] = None,
        rebin: int = 1,
        return_k: bool = False,
    ):
        """
        Get the power spectrum multipoles, optionally rebinned and with k-range selection.

        Parameters
        ----------
        kmin : float, optional
            Minimum k value to include. Default is None (no minimum).
        kmax : float, optional
            Maximum k value to include. Default is None (no maximum).
        rebin : int, optional
            Factor by which to rebin the k-bins. Default is 1 (no rebinning).
        return_k : bool, optional
            Whether to return the k values along with the multipoles. Default is False.
        """
        spectrum = self.spectrum.select(k=slice(0, None, rebin))
        if kmin is not None and kmax is not None:
            spectrum = spectrum.select(k=(kmin, kmax))

        poles = [spectrum.get(ell) for ell in spectrum.ells]
        k = poles[0].coords("k")
        poles = [pole.value() for pole in poles]

        if return_k:
            return k, poles
        return poles

    def save(self, fn: str | Path) -> None:
        """Save the computed power spectrum to a file. Only process 0 will write to disk.

        Parameters
        ----------
        fn : str | Path
            Path to save the power spectrum.
        """
        if jax.process_index() != 0:  # Only process 0 saves to disk
            return  # Exit early for non-zero processes

        fn = Path(fn)  # Ensure fn is a Path object
        tmp_fn = fn.with_name(fn.stem + ".tmp" + fn.suffix)
        self.spectrum.write(tmp_fn)
        logger.info(f"Saving power spectrum to {fn}")
        tmp_fn.replace(fn)  # Atomic move to avoid partial writes
