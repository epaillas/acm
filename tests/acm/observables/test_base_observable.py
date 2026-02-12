"""
Docstring for tests.acm.observables.test_base_observable

Observable(stat_name='tpcf', ...)

tpcf.npy is a pickle file with xarray DataSet
https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
"""

import os
from copy import copy

import xarray

from acm.observables.base import *
from acm.observables.base import Observable

DIR_TEST = os.getenv("ACM_TEST_DATA")


def test_init():
    OBS_TEST = Observable(
        stat_name="tpcf", paths=dict(data_dir=DIR_TEST, model_dir=DIR_TEST)
    )
    assert isinstance(OBS_TEST.x, xarray.DataArray)
    assert isinstance(OBS_TEST.y, xarray.DataArray)
    assert isinstance(OBS_TEST.covariance_y, xarray.DataArray)


def test_copy():
    OBS_TEST = Observable(
        stat_name="tpcf", paths=dict(data_dir=DIR_TEST, model_dir=DIR_TEST)
    )
    obst_copy = copy(OBS_TEST)
    assert obst_copy.paths == OBS_TEST.paths
    xarray.testing.assert_equal(obst_copy.x, OBS_TEST.x)
    xarray.testing.assert_equal(obst_copy.y, OBS_TEST.y)
    xarray.testing.assert_equal(obst_copy.covariance_y, OBS_TEST.covariance_y)


def test_copy_method():
    """
    Certainly __getattr__() use method of dataset from xarray ?
    This can cause confusion
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
        model = self.obst.get_model_prediction(self.obst.x[0, 0])
        assert model.size == self.obst.model.mlp[-1].out_features
        
    def test_select_filters(self):
        nb_x_cosmo = self.obst.x.shape[0] 
        self.obst.select_filters = {'cosmo_idx': [0],}
        assert self.obst.x.shape[0] == 1
        assert self.obst.x.shape[0] != nb_x_cosmo
        self.obst.select_filters = {}
        assert self.obst.x.shape[0] == nb_x_cosmo
        self.obst.select_filters = {'cosmo_idx': [0,2],}
        assert self.obst.x.shape[0] == 2

