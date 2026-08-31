"""Tests for acm.observables.factory."""
import pytest

from acm.observables.base import BaseObservable
from acm.observables.factory import Observable, ObservableFactory, factory
from acm.observables.lsstypes import LsstypesObservable
from acm.observables.xarray import XarrayObservable

# ruff: noqa: ANN001, ANN201, ANN206, D102, S101

class DummyObservable:
    """Minimal stand-in backend class, decoupled from any real Observable implementation."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def can_load(cls, filename) -> bool:
        return str(filename).endswith(".foo")

    @classmethod
    def load(cls, filename, **kwargs):
        return cls(loaded_from=filename, **kwargs)


class OtherDummyObservable(DummyObservable):
    """A second dummy backend, distinct from DummyObservable, for multi-registration tests."""

    @classmethod
    def can_load(cls, filename) -> bool:
        return str(filename).endswith(".bar")


@pytest.fixture
def dummy_factory() -> ObservableFactory:
    """Fresh ObservableFactory registered with two unrelated dummy backends."""
    f = ObservableFactory()
    f.register_observable("foo", DummyObservable)
    f.register_observable("bar", OtherDummyObservable)
    return f


class TestObservableFactory:
    """Tests for the ObservableFactory."""

    def test_register_and_get_observable(self, dummy_factory):
        assert dummy_factory.get_observable("foo") is DummyObservable
        assert dummy_factory.get_observable("bar") is OtherDummyObservable

    def test_get_observable_unsupported_backend_raises(self, dummy_factory):
        with pytest.raises(ValueError, match="Unsupported backend"):
            dummy_factory.get_observable("baz")

    def test_get_loader_returns_matching_creator(self, dummy_factory):
        """Checks get_loader picks the one registered creator whose can_load returns True."""
        assert dummy_factory.get_loader("data.foo") is DummyObservable
        assert dummy_factory.get_loader("data.bar") is OtherDummyObservable

    def test_get_loader_no_match_raises(self, dummy_factory):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            dummy_factory.get_loader("data.baz")


class TestObservable:
    """Tests for the Observable class."""

    def test_new_dispatches_to_backend_class(self, dummy_factory, monkeypatch):
        """Checks Observable instantiates the registered class with forwarded args."""
        monkeypatch.setattr("acm.observables.factory.factory", dummy_factory)
        obs = Observable(1, x=2, backend="foo")
        assert isinstance(obs, DummyObservable)
        assert obs.args == (1,)
        assert obs.kwargs == {"x": 2}

    def test_load_dispatches_via_get_loader(self, dummy_factory, monkeypatch):
        monkeypatch.setattr("acm.observables.factory.factory", dummy_factory)
        obs = Observable.load("data.bar", extra=3)
        assert isinstance(obs, OtherDummyObservable)
        assert obs.kwargs == {"loaded_from": "data.bar", "extra": 3}


class TestModuleFactorySingleton:
    """Tests for the module-level factory singleton."""

    def test_default_registrations(self):
        """Checks the real module-level factory has the two real backends registered."""
        assert factory.get_observable("xarray") is XarrayObservable
        assert factory.get_observable("lsstypes") is LsstypesObservable
        assert issubclass(XarrayObservable, BaseObservable)
        assert issubclass(LsstypesObservable, BaseObservable)
