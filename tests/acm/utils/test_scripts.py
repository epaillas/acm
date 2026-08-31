import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from acm.utils.scripts import (
    NumpyLoader,
    apply_parser_default,
    detect_gpu,
    dump_config,
    get_nthreads,
    load_parser_default,
    retry,
)

# ruff: noqa: ANN001, ANN201, ARG002, D101, D102, D103, S101

#%% Fixtures
@pytest.fixture
def config_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("alpha: 1\nbeta: hello\n")
    return p


def make_parser_with_config(config_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_path)
    return parser

#%% Test classes
class TestDetectGpu:
    def test_detected_gpu(self):
        with patch("acm.utils.scripts.check_output", return_value=b"GPU info"):
            assert detect_gpu() is True

    def test_gpu_not_found(self):
        with patch("acm.utils.scripts.check_output", side_effect=Exception("no gpu")):
            assert detect_gpu() is False


class TestGetNThreads:
    def test_get_count(self):
        with patch("acm.utils.scripts.cpu_count", return_value=4):
            assert get_nthreads() == 4

    def test_multiplier(self):
        with patch("acm.utils.scripts.cpu_count", return_value=4):
            assert get_nthreads(nthread_per_cpu=2) == 8

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="must be bigger than 1"):
            get_nthreads(nthread_per_cpu=0)


class TestLoadParserDefault:
    def test_load_yaml(self, config_file):
        parser = make_parser_with_config(str(config_file))
        with patch("sys.argv", ["prog"]):
            result = load_parser_default(parser)
        assert result == {"alpha": 1, "beta": "hello"}

    def test_no_config_returns_empty(self):
        """A parser without a provided config files returns no default values."""
        parser = make_parser_with_config(None)
        with patch("sys.argv", ["prog"]):
            result = load_parser_default(parser)
        assert result == {}

    def test_missing_config_arg_raises(self):
        """A parser without a --config argument should raise before any file I/O."""
        parser = argparse.ArgumentParser()
        with patch("sys.argv", ["prog"]), pytest.raises(ValueError, match="config"):
            load_parser_default(parser)

    def test_extra_args_ignored(self, config_file):
        """Extra arguments added after load_parser_default should not cause an error."""
        parser = make_parser_with_config(str(config_file))
        with patch("sys.argv", ["prog", "--some_arg", "0"]):
            load_parser_default(parser)

class TestApplyParserDefault:
    def test_sets_defaults(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--alpha", type=float)
        apply_parser_default(parser, {"alpha": 3.14})
        with patch("sys.argv", ["prog"]):
            args = parser.parse_args()
        assert args.alpha == pytest.approx(3.14)

    def test_clears_required(self):
        """Arguments supplied via config should no longer be marked required, so that the parser does not fail when they are absent from sys.argv."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--alpha", type=float, required=True)
        apply_parser_default(parser, {"alpha": 1.0})
        action = next(a for a in parser._actions if a.dest == "alpha")
        assert action.required is False


class TestDumpConfig:
    def test_missing_dump_arg_raises(self):
        parser = argparse.ArgumentParser()
        with patch("sys.argv", ["prog"]), pytest.raises(ValueError, match="dump_config"):
                dump_config(parser)

    def test_exits_when_true(self, capsys):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dump_config", action="store_true")
        parser.add_argument("--config", default=None)
        with patch("sys.argv", ["prog", "--dump_config"]), pytest.raises(SystemExit):
            dump_config(parser)

    def test_does_nothing_when_false(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dump_config", action="store_true")
        with patch("sys.argv", ["prog"]):
            dump_config(parser)  # should not raise or exit

    def test_prints_each_arg(self, capsys):
        """Each non-meta argument (excluding config / dump_config) should appear on stdout as 'key: value' when --dump_config is set."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dump_config", action="store_true")
        parser.add_argument("--config", default=None)
        parser.add_argument("--alpha", type=float, default=3.14)
        parser.add_argument("--label", default="test")
        with patch("sys.argv", ["prog", "--dump_config"]), pytest.raises(SystemExit):
            dump_config(parser)
        captured = capsys.readouterr().out
        assert "alpha: 3.14" in captured
        assert "label: test" in captured


class TestRetry:
    def test_succeeds_on_first_attempt(self):
        """Operation succeeding immediately should return its value after one call."""
        op = MagicMock(return_value=42)
        assert retry(3, op) == 42
        op.assert_called_once()

    def test_succeeds_after_transient_failure(self):
        """Operation failing once then succeeding should return the successful value."""
        op = MagicMock(side_effect=[Exception("fail"), 99])
        result = retry(3, op)
        assert result == 99
        assert op.call_count == 2

    def test_returns_none_after_all_failures(self):
        """When all attempts fail the return value should be None."""
        op = MagicMock(side_effect=Exception("always fails"))
        result = retry(3, op)
        assert result is None
        assert op.call_count == 3

    def test_forwards_args_and_kwargs(self):
        """Positional and keyword arguments must be forwarded to the operation unchanged."""
        op = MagicMock(return_value="ok")
        retry(2, op, "a", "b", key="val")
        op.assert_called_once_with("a", "b", key="val")

    @patch("jax.clear_caches")
    @patch("gc.collect")
    def test_cache_cleared_on_failure(self, mock_gc, mock_jax_clear):
        """jax.clear_caches and gc.collect should each be called once per failure."""
        op = MagicMock(side_effect=[Exception("fail"), Exception("fail"), None])
        ntries = 3
        failures = 2
        retry(ntries, op)
        assert mock_jax_clear.call_count == failures
        assert mock_gc.call_count == failures

    def test_times_one_no_retry(self):
        """With times=1 a failing operation should be attempted exactly once with no retry."""
        op = MagicMock(side_effect=Exception("fail"))
        result = retry(1, op)
        assert result is None
        op.assert_called_once()

    def test_times_less_than_one_raises(self):
        """times<1 should raise an error."""
        op = MagicMock(return_value="ok")
        with pytest.raises(ValueError, match='got 0'):
            retry(0, op)


class TestNumpyLoader:
    def test_arange(self):
        data = yaml.load("values: !np.arange [0, 5, 1]", Loader=NumpyLoader)  # noqa: S506
        np.testing.assert_array_equal(data["values"], np.arange(0, 5, 1))

    def test_linspace(self):
        data = yaml.load("values: !np.linspace [0, 1, 5]", Loader=NumpyLoader)  # noqa: S506
        np.testing.assert_array_almost_equal(data["values"], np.linspace(0, 1, 5))
