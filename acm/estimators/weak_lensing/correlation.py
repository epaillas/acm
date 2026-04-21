import numpy as np
import treecorr


class LensingCorrelation:
    """Class to compute lensing correlation functions using TreeCorr."""

    def __init__(self) -> None:
        pass

    def kappa_correlation(
        self, cat1: "AbacusConvergenceMap", cat2: "AbacusConvergenceMap | None" = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the convergence (auto/cross) correlation function using TreeCorr.

        Parameters
        ----------
        cat1 : AbacusConvergenceMap
            The first convergence map catalog.
        cat2 : AbacusConvergenceMap, optional
            The second convergence map catalog. If None, compute the auto-correlation of cat1.

        Returns
        -------
        r : array
            The separation bins (in degrees).
        xi : array
            The correlation function values.
        """
        # Compute the kappa correlation
        kk = treecorr.KKCorrelation(
            min_sep=0.1, max_sep=5, nbins=10, sep_units="degrees", bin_type="Linear"
        )
        cat1 = cat1.treecorr
        cat2 = cat2.treecorr if cat2 else None
        kk.process(cat1=cat1, cat2=cat2)
        r = kk.meanr
        xi = kk.xi
        return r, xi


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    from acm.lensing import AbacusConvergenceMap

    # Initialize convergence map
    kappa_map = AbacusConvergenceMap(
        snap_idx=47, cosmo_idx=0, phase_idx=0, sim_type="base"
    )
    kappa_map.read_map()
    kappa_map.sample_mask()
    kappa_map.to_treecorr()

    # Create lensing correlation object
    lens_corr = LensingCorrelation()

    # Compute kappa correlation
    r, xi = lens_corr.kappa_correlation(kappa_map)

    # plot results
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(r, xi)
    ax.set_xlabel("separation (degrees)")
    ax.set_ylabel(r"correlation function $\xi$")
    plt.tight_layout()
    plt.savefig("kappa_correlation.png", dpi=300, bbox_inches="tight")
