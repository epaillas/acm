from pathlib import Path  # noqa: INP001

import numpy as np
import yaml
from cosmoprimo.fiducial import DESI
from mockfactory import LagrangianLinearMock

from acm.utils.scripts import NumpyLoader


def make_lagrangian_mock(
    boxsize: float = 500.0,
    nbar: float = 8e-4,
    nmesh: int = 256,
    bias: float = 2.0,
    redshift: float = 0.5,
    seed: int = 42,
    rsd: bool = True,
    los: str = "z",
) -> tuple[np.ndarray, float]:
    """Return cartesian redshift-space positions from a small mockfactory Lagrangian mock."""
    cosmo = DESI()
    power = (
        cosmo.get_fourier(engine="eisenstein_hu", set_engine=False)
        .pk_interpolator()
        .to_1d(z=redshift)
    )
    mock = LagrangianLinearMock(
        power,
        nmesh=nmesh,
        boxsize=boxsize,
        boxcenter=0.0,
        seed=seed,
        unitary_amplitude=False,
    )
    mock.set_real_delta_field(bias=bias - 1.0)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=seed + 1)
    if rsd:
        f = cosmo.growth_rate(redshift)
        mock.set_rsd(f=f, los=los)
    return np.asarray(mock.to_catalog()["Position"]), boxsize

def load_estimator_parameters(name: str | None = None) -> dict:
    """Load estimator parameters from a YAML file."""
    params_file = Path(__file__).parent / "estimator_parameters.yaml"
    with params_file.open("r") as f:
        params = yaml.load(f, Loader=NumpyLoader)  # noqa: S506
    if name is not None:
        if name not in params:
            raise ValueError(f"No default parameters found for estimator: {name}")
        return params[name]
    return params
