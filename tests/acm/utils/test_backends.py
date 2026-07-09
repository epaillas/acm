import logging

import pytest

from acm.utils.backends import BackendRegistry

# ruff: noqa: ANN001, ANN201, ANN204, ARG001, D101, D103, INP001, S101


# Dummy base and concrete classes
class DummyBase:
    def __init__(self, value=0):
        self.value = value

class DummyBackend(DummyBase):
    pass

class AnotherDummyBackend(DummyBase):
    pass

class NotADummyBackend:
    """Does not inherit from DummyBase."""


@pytest.fixture
def registry():
    return BackendRegistry(DummyBase)

@pytest.fixture
def populated_registry(registry):
    registry.register("dummy")(DummyBackend)
    return registry


# --- Registration ---

def test_register_valid_backend(registry):
    """A valid subclass should be registered without error."""
    registry.register("dummy")(DummyBackend)
    assert "dummy" in registry.available

def test_register_returns_class_unchanged(registry):
    """The decorator should return the class itself."""
    result = registry.register("dummy")(DummyBackend)
    assert result is DummyBackend

def test_register_invalid_backend_raises(registry):
    """A class not inheriting from the base class should raise TypeError."""
    with pytest.raises(TypeError, match="DummyBase"):
        registry.register("invalid")(NotADummyBackend)

def test_register_overwrite_warns(populated_registry, caplog):
    with caplog.at_level(logging.WARNING):
        populated_registry.register("dummy")(AnotherDummyBackend)
    assert "dummy" in caplog.text

def test_register_overwrite_replaces(populated_registry):
    """Overwriting a registration should replace the previous class."""
    populated_registry.register("dummy")(AnotherDummyBackend)
    instance = populated_registry.load("dummy")
    assert isinstance(instance, AnotherDummyBackend)


# --- Loading by name ---

def test_load_by_name(populated_registry):
    """Loading by registered name should return an instance of the backend."""
    instance = populated_registry.load("dummy")
    assert isinstance(instance, DummyBackend)

def test_load_by_name_forwards_args(populated_registry):
    """Constructor args should be forwarded when loading by name."""
    instance = populated_registry.load("dummy", value=42)
    assert instance.value == 42

def test_load_unknown_name_raises(populated_registry):
    """Loading an unregistered name should raise KeyError."""
    with pytest.raises(KeyError, match="unknown"):
        populated_registry.load("unknown")


# --- Loading by instance ---

def test_load_existing_instance_passthrough(populated_registry):
    """Passing an existing instance should return it unchanged."""
    existing = DummyBackend()
    result = populated_registry.load(existing)
    assert result is existing

def test_load_invalid_type_raises(populated_registry):
    """Passing an invalid type should raise TypeError."""
    with pytest.raises(TypeError):
        populated_registry.load(123)


# --- Available ---

def test_available_empty(registry):
    """An empty registry should return an empty list."""
    assert registry.available == []

def test_available_lists_registered(registry):
    """Available should list all registered backend names."""
    registry.register("foo")(DummyBackend)
    registry.register("bar")(AnotherDummyBackend)
    assert set(registry.available) == {"foo", "bar"}


# --- Registry isolation ---

@pytest.fixture
def second_registry():
    return BackendRegistry(DummyBase)

def test_registries_do_not_share_backends(populated_registry, second_registry):
    """A backend registered in one registry should not appear in another."""
    assert "dummy" not in second_registry.available

def test_registries_do_not_interfere_on_load(populated_registry, second_registry):
    """Loading from an empty registry should raise even if another registry has the name."""
    with pytest.raises(KeyError):
        second_registry.load("dummy")

def test_registries_can_hold_same_name_independently(populated_registry, second_registry):
    """Two registries can register different classes under the same name without conflict."""
    second_registry.register("dummy")(AnotherDummyBackend)
    assert isinstance(populated_registry.load("dummy"), DummyBackend)
    assert isinstance(second_registry.load("dummy"), AnotherDummyBackend)
