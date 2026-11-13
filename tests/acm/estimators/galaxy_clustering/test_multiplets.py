from unittest.mock import MagicMock, patch

import lsstypes
import matplotlib  # noqa: ICN001
import matplotlib.pyplot as plt
import numpy as np
import pytest

from acm.estimators.galaxy_clustering.multiplets import (
    GalaxyMultiplets,
    identify_multiplets,
)

matplotlib.use("Agg")  # Use a non-interactive backend for testing

MODULE = "acm.estimators.galaxy_clustering.multiplets"
EDGES = (np.linspace(0.1, 20, 5), np.linspace(-20, 20, 9))

# ruff: noqa: ANN001, ANN201, ARG002, D101, D102, D103, INP001, S101


def pycorr_like(name):
    """Mock pycorr correlation (``name`` is an attribute, not MagicMock's repr name)."""
    corr = MagicMock()
    corr.name = name
    return corr


def make_box(positions, **kwargs):
    """Box mode: no randoms (DummyBackend boxsize = 100)."""
    return GalaxyMultiplets(
        backend="dummyestimatorbackend",
        data_positions=np.asarray(positions, dtype=float),
        **kwargs,
    )


@pytest.fixture
def box_estimator(data_positions):
    return make_box(data_positions)


@pytest.fixture
def cutsky_estimator(make_estimator):
    return make_estimator(GalaxyMultiplets)


class TestInit:
    def test_box_mode_without_randoms(self, box_estimator):
        assert not box_estimator.is_cutsky

    def test_cutsky_mode_with_randoms(self, cutsky_estimator):
        assert cutsky_estimator.is_cutsky


class TestIdentifyBox:
    def test_singlet_pair_triplet(self):
        pos = [
            [10, 10, 10],
            [10.5, 10, 10],  # pair
            [50, 50, 50],
            [50.5, 50, 50],
            [51.0, 50, 50],  # triplet (chain)
            [80, 80, 80],  # singlet
        ]
        box = make_box(pos)
        groups = identify_multiplets(box.data_positions, box.backend.boxsize)
        assert [len(groups[n]) for n in (1, 2, 3, 4)] == [1, 1, 1, 0]
        np.testing.assert_allclose(groups[2][0], [10.25, 10, 10])
        np.testing.assert_allclose(groups[1][0], [80, 80, 80])

    def test_rperp_cut_along_los(self):
        # 5 Mpc/h apart along the LOS (r_perp = 0) -> linked;
        # 5 Mpc/h apart perpendicular to the LOS -> not linked
        box = make_box([[10, 10, 10], [10, 10, 15]])
        mult_x = identify_multiplets(box.data_positions, box.backend.boxsize, los="x")
        mult_z = identify_multiplets(box.data_positions, box.backend.boxsize, los="z")
        assert len(mult_z[2]) == 1
        assert len(mult_x[2]) == 0

    def test_rmax_cut(self):
        # r_perp = 0 but 3D separation above r_max
        box = make_box([[10, 10, 10], [10, 10, 18]])
        assert len(identify_multiplets(box.data_positions, box.backend.boxsize)[2]) == 0

    def test_pair_across_periodic_boundary(self):
        box = make_box([[0.2, 10, 10], [99.9, 10, 10], [50, 50, 50]])
        groups = identify_multiplets(box.data_positions, box.backend.boxsize)
        assert len(groups[2]) == 1
        center_x = groups[2][0, 0]
        assert min(center_x, 100 - center_x) < 0.5  # wrapped mean, not the box middle

    def test_positions_centered_on_zero_are_wrapped(self):
        box = make_box([[-10, -10, -10], [-9.5, -10, -10]])
        groups = identify_multiplets(box.data_positions, box.backend.boxsize)
        np.testing.assert_allclose(groups[2][0], [90.25, 90, 90])

    def test_max_multiplicity(self):
        pos = [[10 + 0.5 * i, 10, 10] for i in range(5)]  # chain of 5
        box = make_box(pos)
        groups = identify_multiplets(
            box.data_positions, box.backend.boxsize, max_multiplicity=5
        )
        assert len(groups[5]) == 1
        assert all(len(groups[n]) == 0 for n in (1, 2, 3, 4))


class TestIdentifyCutsky:
    def make(self, positions):
        positions = np.asarray(positions, dtype=float)
        return GalaxyMultiplets(
            backend="dummyestimatorbackend",
            data_positions=positions,
            randoms_positions=positions.copy(),
        )

    def test_midpoint_los(self):
        # Two galaxies 5 Mpc/h apart along the radial direction (LOS): linked.
        # Same separation transverse to the LOS: not linked.
        box_r = self.make([[1000, 0, 0], [1005, 0, 0]])
        box_t = self.make([[1000, 0, 0], [1000, 5, 0]])
        radial = identify_multiplets(box_r.data_positions, None)
        transverse = identify_multiplets(box_t.data_positions, None)
        assert len(radial[2]) == 1
        assert len(transverse[2]) == 0

    def test_no_periodic_wrapping(self):
        # in cutsky, galaxies on opposite sides of the (dummy) box are never linked
        box = self.make([[0.2, 10, 10], [99.9, 10, 10]])
        groups = identify_multiplets(box.data_positions, None)
        assert len(groups[2]) == 0


class TestCompute:
    @patch(f"{MODULE}.ObservableTree")
    @patch(f"{MODULE}.from_pycorr")
    @patch(f"{MODULE}.TwoPointCorrelationFunction")
    def test_box_arguments(self, mock_tpcf, mock_from_pycorr, mock_tree, box_estimator):
        box_estimator.compute(edges=EDGES, nthreads=2)
        args, kwargs = mock_tpcf.call_args
        assert args == ("rppi",)
        assert kwargs["edges"] == EDGES
        np.testing.assert_array_equal(kwargs["boxsize"], box_estimator.backend.boxsize)
        assert kwargs["los"] == "z"
        assert kwargs["position_type"] == "pos"
        assert kwargs["nthreads"] == 2
        assert "randoms_positions2" not in kwargs
        assert "estimator" not in kwargs

    @patch(f"{MODULE}.ObservableTree")
    @patch(f"{MODULE}.from_pycorr")
    @patch(f"{MODULE}.TwoPointCorrelationFunction")
    def test_cutsky_arguments(
        self, mock_tpcf, mock_from_pycorr, mock_tree, cutsky_estimator
    ):
        cutsky_estimator.compute(edges=EDGES)
        kwargs = mock_tpcf.call_args.kwargs
        assert kwargs["estimator"] == "davispeebles"
        assert kwargs["los"] == "midpoint"
        assert kwargs["randoms_positions2"] is cutsky_estimator.randoms_positions
        assert "boxsize" not in kwargs


class TestToLsstypes:
    @patch(
        f"{MODULE}.from_pycorr",
        side_effect=AssertionError("count names in davispeebles"),
    )
    def test_davispeebles_unsupported_gives_clear_error(self, mock_from_pycorr):
        with pytest.raises(NotImplementedError, match="lsstypes/issues/6"):
            GalaxyMultiplets._to_lsstypes(pycorr_like("davispeebles"))

    @patch(f"{MODULE}.from_pycorr", side_effect=AssertionError("other"))
    def test_other_errors_propagate(self, mock_from_pycorr):
        with pytest.raises(AssertionError, match="other"):
            GalaxyMultiplets._to_lsstypes(pycorr_like("natural"))

    @patch(f"{MODULE}.from_pycorr")
    def test_passthrough(self, mock_from_pycorr):
        corr = MagicMock()
        assert GalaxyMultiplets._to_lsstypes(corr) is mock_from_pycorr.return_value
        mock_from_pycorr.assert_called_once_with(corr)


class TestLoad:
    @patch(f"{MODULE}.lsstypes.read")
    def test_load_without_project(self, mock_read, tmp_path):
        obj = MagicMock()
        mock_read.return_value = obj
        assert GalaxyMultiplets.load(tmp_path / "m.h5") is obj
        obj.flatten.assert_not_called()

    @patch(f"{MODULE}.ObservableTree")
    @patch(f"{MODULE}.lsstypes.read")
    def test_load_with_project(self, mock_read, mock_tree, tmp_path):
        leaf = MagicMock()
        obj = MagicMock(multiplets=["pair"], attrs={})
        obj.flatten.return_value = [leaf]
        mock_read.return_value = obj
        GalaxyMultiplets.load(tmp_path / "m.h5", project=True, ignore_nan=True)
        leaf.project.assert_called_once_with(ignore_nan=True)
        mock_tree.assert_called_once_with(
            [leaf.project.return_value], multiplets=["pair"], attrs={}
        )


class TestPlot:
    @pytest.fixture
    def projected(self):
        leaf = MagicMock()  # already projected (not a Count2Correlation)
        leaf.coords.return_value = np.linspace(1, 20, 5)
        leaf.value.return_value = np.ones(5)
        obj = MagicMock(multiplets=["singlet", "pair"])
        obj.get.return_value = leaf
        return obj

    def test_one_line_per_class(self, projected):
        fig, ax = GalaxyMultiplets.plot(projected)
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_uses_provided_axes(self, projected):
        fig_in, ax_in = plt.subplots()
        fig, ax = GalaxyMultiplets.plot(projected, fig=fig_in, ax=ax_in)
        assert fig is fig_in
        assert ax is ax_in
        plt.close(fig)

    def test_projects_unprojected_leaves(self, projected):
        leaf = MagicMock(spec=lsstypes.Count2Correlation)
        leaf.project.return_value = projected.get.return_value
        projected.get.return_value = leaf
        fig, _ = GalaxyMultiplets.plot(projected)
        assert leaf.project.call_count == 2
        plt.close(fig)
