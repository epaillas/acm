"""
Docstring for tests.acm.observables.test_base_observable

Observable(stat_name='tpcf', ...)

tpcf.npy is a pickle file with xarray DataSet
https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html

"""

import os
from copy import copy

from acm.observables.base import *
from acm.observables.base import Observable

DIR_TEST = os.getenv("ACM_TEST_DATA")

OBS_TEST = Observable(
    stat_name="tpcf", paths=dict(data_dir=DIR_TEST, model_dir=DIR_TEST)
)


def test_init():
    assert isinstance(OBS_TEST.x, xarray.DataArray)
    assert isinstance(OBS_TEST.y, xarray.DataArray)
    assert isinstance(OBS_TEST.covariance_y, xarray.DataArray)


def test_copy():
    try:
        obst_copy = copy(OBS_TEST)
        assert obst_copy.x == OBS_TEST.x
    except Exception as e:
        assert False


class TestObservable:
    def setup_method(self):
        self.obst = Observable(
            stat_name="tpcf", paths=dict(data_dir=DIR_TEST, model_dir=DIR_TEST)
        )

    def test_get_coordinate_list(self):
        assert self.obst.get_coordinate_list("hod_idx") == list(range(100))

    def test_get_model_prediction(self):
        model = self.obst.get_model_prediction(self.obst.x[0,0])
        assert model.size == self.obst.model.mlp[-1].out_features