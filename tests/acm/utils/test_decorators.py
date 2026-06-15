import os
import pytest
from unittest.mock import patch

from acm.utils.decorators import temporary_class_state, require_nersc, kwargs_alias

#%% Tests for temporary_class_state decorator

class Dummy:
    """A simple class with attributes to test temporary_class_state."""
    def __init__(self, x=1, y=2):
        self.x = x
        self.y = y

    @temporary_class_state(x=99)
    def get_x(self):
        return self.x

    @temporary_class_state(x=99, y=88)
    def get_xy(self):
        return self.x, self.y

    @temporary_class_state(x=99)
    def raise_error(self):
        raise RuntimeError("oops")

class TestTemporaryClassState:
    """General test suite for the temporary_class_state decorator."""

    # --- Temporary modification ---

    def test_attr_is_modified_during_call(self):
        """Attribute should be temporarily set to the decorator value during the call."""
        d = Dummy(x=1)
        assert d.get_x() == 99

    def test_attr_is_restored_after_call(self):
        """Attribute should be restored to its original value after the call."""
        d = Dummy(x=1)
        d.get_x()
        assert d.x == 1

    def test_multiple_attrs_modified_during_call(self):
        """Multiple attributes should all be temporarily modified."""
        d = Dummy(x=1, y=2)
        assert d.get_xy() == (99, 88)

    def test_multiple_attrs_restored_after_call(self):
        """All modified attributes should be restored after the call."""
        d = Dummy(x=1, y=2)
        d.get_xy()
        assert d.x == 1
        assert d.y == 2

    # --- Restoration on exception ---

    def test_attr_restored_after_exception(self):
        """Attribute should be restored even if the method raises an exception."""
        d = Dummy(x=1)
        with pytest.raises(RuntimeError):
            d.raise_error()
        assert d.x == 1

    # --- Independence between instances ---

    def test_state_is_instance_specific(self):
        """Temporary modification on one instance should not affect another."""
        d1 = Dummy(x=1)
        d2 = Dummy(x=5)
        d1.get_x()
        assert d2.x == 5

##%% Tests for require_nersc decorator
@require_nersc(enabled=True)
def nersc_dummy():
    return "ok"

@require_nersc(enabled=False)
def nersc_dummy_disabled():
    return "ok"

class TestRequireNersc:
    """General test suite for the require_nersc decorator."""

    def test_raises_outside_nersc_when_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(OSError, match="nersc_dummy"):
                nersc_dummy()

    def test_runs_outside_nersc_when_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            assert nersc_dummy_disabled() == "ok"

    def test_runs_on_nersc_when_enabled(self):
        with patch.dict(os.environ, {"NERSC_HOST": "perlmutter"}):
            assert nersc_dummy() == "ok"

    def test_preserves_function_name(self):
        assert nersc_dummy.__name__ == "nersc_dummy"

#%% Tests for kwargs_alias decorator

# Dummy function to test the decorator
@kwargs_alias(foo="bar")
def dummy(foo=None, **kwargs):
    return foo, kwargs

@kwargs_alias(foo="bar", baz="qux")
def dummy_multi(foo=None, baz=None, **kwargs):
    return foo, baz, kwargs

class TestKwargsAlias:
    """General test suite for the kwargs_alias decorator."""

    # --- Normal resolution ---

    def test_alias_is_resolved(self):
        """Alias 'bar' should be passed as canonical 'foo'."""
        assert dummy(bar=42) == (42, {})

    def test_canonical_passthrough(self):
        """Canonical name 'foo' should pass through unchanged."""
        assert dummy(foo=42) == (42, {})

    def test_unrelated_kwargs_are_untouched(self):
        """Kwargs unrelated to any alias should be forwarded as-is."""
        foo, kwargs = dummy(foo=1, extra=99)
        assert foo == 1
        assert kwargs == {"extra": 99}

    def test_alias_with_unrelated_kwargs(self):
        """Alias resolution should not affect unrelated kwargs."""
        foo, kwargs = dummy(bar=1, extra=99)
        assert foo == 1
        assert kwargs == {"extra": 99}

    def test_no_args_uses_default(self):
        """When neither canonical nor alias is passed, default should apply."""
        assert dummy() == (None, {})

    # --- Conflict detection ---

    def test_both_canonical_and_alias_raises(self):
        """Passing both canonical and alias at the same time should raise ValueError."""
        with pytest.raises(ValueError, match="foo"):
            dummy(foo=1, bar=2)

    # --- Multiple aliases ---

    def test_multiple_aliases_resolved_independently(self):
        assert dummy_multi(bar=1, qux=2) == (1, 2, {})

    def test_multiple_aliases_conflict_raises(self):
        with pytest.raises(ValueError, match="foo"):
            dummy_multi(foo=1, bar=2)