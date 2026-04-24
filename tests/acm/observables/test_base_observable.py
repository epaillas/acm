"""
Observable(stat_name='tpcf', ...)


Two important attributes of the Observable class are :
 * _dataset : xarray.Dataset containing the data and metadata of the observable.
 * model    : an instance of a MLP neural networkel that can be used to make predictions based on the data in the dataset.

From npy file (in fact it's a pickle file with xarray Dataset) in 'data_dir' parameter
* https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
* https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataset

'_dataset': <xarray.Dataset> Size: 9MB
Dimensions:       (cosmo_idx: 85, hod_idx: 100, parameters: 20, ells: 2, s: 50,
                   phase_idx: 1643)
Coordinates:
  * cosmo_idx     (cosmo_idx) int64 680B 0 1 2 3 4 13 ... 177 178 179 180 181
  * hod_idx       (hod_idx) int64 800B 0 1 2 3 4 5 6 7 ... 93 94 95 96 97 98 99
  * parameters    (parameters) <U9 720B 'omega_b' 'omega_cdm' ... 'B_sat'
  * ells          (ells) int64 16B 0 2
  * s             (s) float64 400B 1.5 4.5 7.5 10.5 ... 139.5 142.5 145.5 148.5
  * phase_idx     (phase_idx) int64 13kB 3000 3001 3002 3003 ... 4997 4998 4999
  
Data variables: (data_vars attribute of xarray.Dataset) 
https://docs.xarray.dev/en/stable/api/dataset.html#attributes
    x             (cosmo_idx, hod_idx, parameters) float64 1MB 0.02237 ... 0....
    y             (cosmo_idx, hod_idx, ells, s) float64 7MB 2.486 ... -0.001705
    covariance_y  (phase_idx, ells, s) float64 1MB 3.873 1.615 ... -0.00172,

From ckpt file in 'model_dir' parameter

 'model': FCN(
  (mlp): Sequential(
    (mlp0): Linear(in_features=20, out_features=549, bias=True)
    (act0): LearnedSigmoid()
    (dropout0): Dropout(p=0.00019182624558841687, inplace=False)
    (mlp1): Linear(in_features=549, out_features=549, bias=True)
    (act1): LearnedSigmoid()
    (dropout1): Dropout(p=0.00019182624558841687, inplace=False)
    (mlp2): Linear(in_features=549, out_features=549, bias=True)
    (act2): LearnedSigmoid()
    (dropout2): Dropout(p=0.00019182624558841687, inplace=False)
    (mlp3): Linear(in_features=549, out_features=549, bias=True)
    (act3): LearnedSigmoid()
    (dropout3): Dropout(p=0.00019182624558841687, inplace=False)
    (mlp4): Linear(in_features=549, out_features=549, bias=True)
    (act4): LearnedSigmoid()
    (dropout4): Dropout(p=0.00019182624558841687, inplace=False)
    (mlp5): Linear(in_features=549, out_features=100, bias=True)
  )
  (loss_fct): L1Loss()
  
  
__getattr__(name) method of Observable class is used to access the attributes of the dataset.
apply also filter if  name is in data_vars of _dataset attibut, ie in this example:
* x,y, covariance_y.

"""
import os
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import xarray
import pytest

from acm.observables.base import Observable

DIR_TEST = os.getenv("ACM_TEST_DATA")


class FakeLoadedModel:
    pass


class FakeModelClass:
    pass


def test_load_model_delegates_to_sunbird_loader(monkeypatch, tmp_path):
    checkpoint_fn = tmp_path / "model.ckpt"
    calls = []

    def fake_load_model_from_checkpoint(checkpoint_fn, model_cls=None):
        calls.append(
            {
                "checkpoint_fn": Path(checkpoint_fn),
                "model_cls": model_cls,
            }
        )
        return FakeLoadedModel()

    monkeypatch.setattr(
        "acm.observables.base.load_model_from_checkpoint",
        fake_load_model_from_checkpoint,
    )

    model = Observable.load_model(checkpoint_fn, model_cls=FakeModelClass)

    assert isinstance(model, FakeLoadedModel)
    assert calls == [
        {
            "checkpoint_fn": checkpoint_fn,
            "model_cls": FakeModelClass,
        }
    ]


def test_init_loads_model_from_paths_with_model_cls(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dataset = xarray.Dataset()
    calls = []

    def fake_load_model_from_checkpoint(checkpoint_fn, model_cls=None):
        calls.append(
            {
                "checkpoint_fn": Path(checkpoint_fn),
                "model_cls": model_cls,
            }
        )
        return FakeLoadedModel()

    monkeypatch.setattr(
        "acm.observables.base.load_model_from_checkpoint",
        fake_load_model_from_checkpoint,
    )

    observable = Observable(
        stat_name="statistic",
        dataset=dataset,
        paths={"model_dir": model_dir},
        model_cls=FakeModelClass,
    )

    assert isinstance(observable.model, FakeLoadedModel)
    assert calls == [
        {
            "checkpoint_fn": model_dir / "statistic.ckpt",
            "model_cls": FakeModelClass,
        }
    ]


def test_init_loads_legacy_checkpoint_with_model_cls(monkeypatch, tmp_path):
    checkpoint_fn = tmp_path / "legacy.ckpt"
    dataset = xarray.Dataset()
    calls = []

    def fake_load_model_from_checkpoint(checkpoint_fn, model_cls=None):
        calls.append(
            {
                "checkpoint_fn": Path(checkpoint_fn),
                "model_cls": model_cls,
            }
        )
        return FakeLoadedModel()

    monkeypatch.setattr(
        "acm.observables.base.load_model_from_checkpoint",
        fake_load_model_from_checkpoint,
    )

    observable = Observable(
        stat_name="statistic",
        dataset=dataset,
        checkpoint_fn=checkpoint_fn,
        model_cls=FakeModelClass,
    )

    assert isinstance(observable.model, FakeLoadedModel)
    assert calls == [
        {
            "checkpoint_fn": checkpoint_fn,
            "model_cls": FakeModelClass,
        }
    ]


def test_init_uses_explicit_model_without_loading_checkpoint(monkeypatch, tmp_path):
    checkpoint_fn = tmp_path / "model.ckpt"
    dataset = xarray.Dataset()
    explicit_model = FakeLoadedModel()

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Explicit model should bypass checkpoint loading.")

    monkeypatch.setattr(
        "acm.observables.base.load_model_from_checkpoint",
        unexpected_load,
    )

    observable = Observable(
        stat_name="statistic",
        dataset=dataset,
        model=explicit_model,
        checkpoint_fn=checkpoint_fn,
        model_cls=FakeModelClass,
    )

    assert observable.model is explicit_model


def test_init():
    OBS_TEST = Observable(
        stat_name="tpcf", paths=dict(data_dir=DIR_TEST, model_dir=DIR_TEST)
    )
    assert isinstance(OBS_TEST.x, xarray.DataArray)
    assert isinstance(OBS_TEST.y, xarray.DataArray)
    assert isinstance(OBS_TEST.covariance_y, xarray.DataArray)


def test_copy():
    common_list = [0, 1]
    OBS_TEST = Observable(
        stat_name="tpcf",
        paths=dict(
            data_dir=DIR_TEST,
            model_dir=DIR_TEST,
        ),
        select_filters={
            "cosmo_idx": common_list,
        },
    )
    obst_copy = copy(OBS_TEST)
    assert obst_copy.paths == OBS_TEST.paths
    xarray.testing.assert_equal(obst_copy.x, OBS_TEST.x)
    xarray.testing.assert_equal(obst_copy.y, OBS_TEST.y)
    xarray.testing.assert_equal(obst_copy.covariance_y, OBS_TEST.covariance_y)
    # test to see if the copy is shallow or deep
    OBS_TEST.select_filters["cosmo_idx"].append(2)
    assert obst_copy.select_filters["cosmo_idx"] == OBS_TEST.select_filters["cosmo_idx"]


def test_deepcopy():
    common_list = [0, 1]
    OBS_TEST = Observable(
        stat_name="tpcf",
        paths=dict(
            data_dir=DIR_TEST,
            model_dir=DIR_TEST,
        ),
        select_filters={
            "cosmo_idx": common_list,
        },
    )
    obst_copy = deepcopy(OBS_TEST)
    assert obst_copy.paths == OBS_TEST.paths
    xarray.testing.assert_equal(obst_copy.x, OBS_TEST.x)
    xarray.testing.assert_equal(obst_copy.y, OBS_TEST.y)
    xarray.testing.assert_equal(obst_copy.covariance_y, OBS_TEST.covariance_y)
    # test to see if the copy is shallow or deep
    OBS_TEST.select_filters["cosmo_idx"].append(2)
    assert obst_copy.select_filters["cosmo_idx"] != OBS_TEST.select_filters["cosmo_idx"]


@pytest.mark.skip(reason="Test temporarily skipped")
def test_copy_method():
    """
    Certainly __getattr__() use method of dataset from xarray ?
    This can cause confusion

    AttributeError: 'Dataset' object has no attribute 'paths'

    """
    
    OBS_TEST = Observable(
        stat_name="tpcf", paths=dict(data_dir=DIR_TEST, model_dir=DIR_TEST)
    )
    obst_copy = OBS_TEST.copy()
    assert obst_copy.paths == OBS_TEST.paths


class TestObservable:
    def setup_method(self):
        self.obst = Observable(
            stat_name="tpcf", paths=dict(data_dir=DIR_TEST, model_dir=DIR_TEST)
        )

    def test_get_coordinate_list(self):
        assert self.obst.get_coordinate_list("hod_idx") == list(range(100))

    def test_get_model_prediction(self):
        """
        model.mlp[-1].out_features is the number of sample in output of the model,
        which should match the size of the model prediction for a given input.
        """
        # y_est = model(x)
        y_est = self.obst.get_model_prediction(self.obst.x[0, 0])
        assert y_est.size == self.obst.model.mlp[-1].out_features

    def test_select_filters(self):
        nb_x_cosmo = self.obst.x.shape[0]
        self.obst.select_filters = {
            "cosmo_idx": [0],
        }
        assert self.obst.x.shape[0] == 1
        assert self.obst.x.shape[0] != nb_x_cosmo
        self.obst.select_filters = {}
        assert self.obst.x.shape[0] == nb_x_cosmo
        self.obst.select_filters = {
            "cosmo_idx": [0, 2],
        }
        assert self.obst.x.shape[0] == 2


def test_drop_nan_dimensions():
    """
    Test that the method drop_nan_dimensions correctly drops dimensions with NaN values in the dataset.
    """
    temperature = [
        [np.nan,0, 2, 9],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 4, 2, 0],
        [np.nan, 1, 0, 0],
    ]
    daa = xarray.DataArray(
        data=temperature,
        dims=["Y", "X"],
        coords=dict(
            lat=("Y", np.array([-20.0, -20.25, -20.50, -20.75])),
            lon=("X", np.array([10.0, 10.25, 10.5, 10.75])),
        ),
    )
    obst = Observable(stat_name="tpcf", paths=dict(data_dir=DIR_TEST, model_dir=DIR_TEST))
    out_daa = obst.drop_nan_dimensions(daa)
    xarray.testing.assert_equal(daa, out_daa)
    daa.attrs["nan_dims"] = ["Y"]
    out_daa = obst.drop_nan_dimensions(daa)
    assert daa.values.shape == (4, 4)
    assert out_daa.values.shape == (3,4)
    daa.attrs["nan_dims"] = ["X"]
    out_daa = obst.drop_nan_dimensions(daa)
    assert out_daa.values.shape == (4,3)
    daa.attrs["nan_dims"] = ["X", "Y"]
    out_daa = obst.drop_nan_dimensions(daa)
    assert out_daa.values.shape == (3,3)

@pytest.mark.skip(reason="Test temporarily skipped")
def test_get_covariance_matrix():
    """
    Test that the method get_covariance_matrix correctly returns the covariance matrix from the dataset.
    
CODE of method get_covariance_matrix()

    cov_y = self.covariance_y  # Filtered and flattened DataArray
    cov_y = self.flatten_output(cov_y, flat_output_dims=2, unstack=False)  # No unstacking to avoid NaN
    cov_y = cov_y.values
    prefactor = prefactor / volume_factor
    cov = prefactor * np.cov(cov_y, rowvar=False)


Comments about method:
    * use self._dataset.covariance_y to get the covariance matrix from the dataset
    * Observe apply filter : cov_y = self.covariance_y 


    1) covariance_y shape is (phase_idx, ells, s) = (1643, 2, 50) 
    covariance matrix must be square.
    
In [2]: np.sqrt(1643)
Out[2]: 40.53393639902249

maybe triangle plus diagonal: n(n-1)/2 = 1643

np.roots([1,-1,-1643*2])
Out[6]: array([ 57.82582315, -56.82582315])

neither , so ?


    2) cov = prefactor * np.cov(cov_y, rowvar=False)
    covariance is defined by covariance method of numpy with covariance  ...?
    """
    obst = Observable(stat_name="tpcf", paths=dict(data_dir=DIR_TEST, model_dir=DIR_TEST))
    cov_matrix = obst.get_covariance_matrix()
    assert False
    

def test_flatten_output():
    arr = xarray.DataArray(
        np.arange(24).reshape(2, 3,4),
        coords=[("x", ["a", "b"]), ("features", [0, 1, 2]), ("sample", [10, 20, 30, 40])],
    )
    res = Observable.flatten_output(arr, flat_output_dims=1)
    print(res)
