import pytest
import pandas as pd
from acm.catalogs.dataclasses import Tracer, Transform


#%% Fixtures

@pytest.fixture
def dummy_dataframe():
    return pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

@pytest.fixture
def dummy_tracer():
    return Tracer(name="FOO", params={"a": 1, "b": 2})

@pytest.fixture
def dummy_transform(dummy_dataframe):
    def scale(data, factor):
        return data * factor
    return Transform(
        name="scale",
        func=scale,
        kwargs={"factor": 2.0},
    )

#%% Test classes (for organizational purposes only, not required by pytest)

class TestTracer:
    def test_tracer_name(self, dummy_tracer):
        assert dummy_tracer.name == "FOO"

    def test_tracer_params(self, dummy_tracer):
        assert dummy_tracer.params == {"a": 1, "b": 2}

    def test_tracer_default_params(self, ):
        """Tracer should default to an empty params dict."""
        tracer = Tracer(name="BAR")
        assert tracer.params == {}

    def test_tracer_default_params_are_independent(self, ):
        """Each Tracer instance should have its own params dict."""
        t1 = Tracer(name="A")
        t2 = Tracer(name="B")
        t1.params["x"] = 1
        assert "x" not in t2.params

    def test_tracer_repr(self, dummy_tracer):
        assert "FOO" in repr(dummy_tracer)
        assert "params" in repr(dummy_tracer)


class TestTransform:
    def test_transform_name(self, dummy_transform):
        assert dummy_transform.name == "scale"

    def test_transform_apply(self, dummy_dataframe, dummy_transform):
        """Apply should call func with data and stored kwargs."""
        result = dummy_transform.apply(dummy_dataframe)
        pd.testing.assert_frame_equal(result, dummy_dataframe * 2.0)

    def test_transform_apply_does_not_mutate_input(self, dummy_dataframe, dummy_transform):
        """Apply should not modify the original DataFrame."""
        original = dummy_dataframe.copy()
        dummy_transform.apply(dummy_dataframe)
        pd.testing.assert_frame_equal(dummy_dataframe, original)

    def test_transform_default_kwargs(self, ):
        """Transform should default to empty kwargs and None tracer."""
        t = Transform(name="identity", func=lambda data: data)
        assert t.kwargs == {}
        assert t.tracer is None

    def test_transform_kwargs_forwarded(self, ):
        """All stored kwargs should be forwarded to the function."""
        received = {}
        def capture(data, **kwargs):
            received.update(kwargs)
            return data

        t = Transform(name="capture", func=capture, kwargs={"foo": 1, "bar": 2})
        t.apply(pd.DataFrame({"x": [1]})) # We just need to call apply to trigger the function and capture the kwargs
        assert received == {"foo": 1, "bar": 2}