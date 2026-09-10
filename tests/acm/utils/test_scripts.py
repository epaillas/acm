import argparse
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from acm.utils.scripts import (
    BenchmarkTimer,
    NumpyLoader,
    apply_parser_default,
    detect_gpu,
    dump_config,
    get_nthreads,
    load_parser_default,
    memory_cleanup,
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


class TestMemoryCleanup:
    @patch("acm.utils.scripts.gc.collect")
    @patch("acm.utils.scripts.clear_caches")
    def test_memory_cleanup_calls_gc_and_jax(self, mock_jax_clear, mock_gc):
        memory_cleanup(use_jax=True)
        mock_gc.assert_called_once()
        mock_jax_clear.assert_called_once()

    @patch("acm.utils.scripts.gc.collect")
    @patch("acm.utils.scripts.clear_caches")
    def test_memory_cleanup_without_jax(self, mock_jax_clear, mock_gc):
        memory_cleanup(use_jax=False)
        mock_gc.assert_called_once()
        mock_jax_clear.assert_not_called()


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

    @patch("acm.utils.scripts.clear_caches")
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


class TestBenchmarkTimer:

    @pytest.fixture
    def timer(self):
        return BenchmarkTimer(keys=["foo", "bar"])

    def test_init_empty_times(self, timer):
        assert timer.times == {"foo": [], "bar": []}

    def test_init_empty_t0(self, timer):
        assert timer.t0 == {}

    def test_start_unknown_key_raises(self, timer):
        with pytest.raises(ValueError, match="Unknown key"):
            timer.start("nonexistent")

    def test_register_unknown_key_raises(self, timer):
        with pytest.raises(ValueError, match="Unknown key"):
            timer.register("nonexistent")

    def test_register_without_start_raises(self, timer):
        """Registering a key that was never started should raise."""
        with pytest.raises(ValueError, match="was not started"):
            timer.register("foo")

    def test_register_returns_elapsed(self, timer):
        timer.start("foo")
        elapsed = timer.register("foo")
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    def test_register_appends_to_times(self, timer):
        timer.start("foo")
        timer.register("foo")
        timer.start("foo")
        timer.register("foo")
        assert len(timer.times["foo"]) == 2

    def test_register_multiple_keys(self, timer):
        """Registering multiple keys should store the same initial time for each key."""
        timer.start("foo", "bar")
        assert timer.t0["foo"] == timer.t0["bar"]

    def test_register_logs_when_log_true(self, timer, caplog):
        """Register should log elapsed time when log=True."""
        timer.start("foo")
        with caplog.at_level(logging.DEBUG):
            timer.register("foo", log=True)
        assert any("foo" in r.message and "Elapsed time" in r.message for r in caplog.records)

    def test_multiple_keys_independent(self, timer):
        timer.start("foo", "bar")
        timer.register("foo")
        timer.register("bar")
        assert len(timer.times["foo"]) == 1
        assert len(timer.times["bar"]) == 1

    def test_report_logs_average(self, timer, caplog):
        """Report should log average time for keys with recorded times."""
        timer.start("foo")
        timer.register("foo")
        with caplog.at_level(logging.INFO):
            timer.report()
        assert any("foo" in r.message and "Average" in r.message for r in caplog.records)

    def test_report_logs_no_recorded_times(self, timer, caplog):
        """Report should log a message for keys with no recorded times."""
        with caplog.at_level(logging.INFO):
            timer.report()
        assert any("No recorded times" in r.message for r in caplog.records)

    def test_register_twice_without_restart_records_stale_time(self, timer):
        """Registering twice without restarting silently records a larger elapsed time."""
        timer.start("foo")
        timer.register("foo")
        elapsed_stale = timer.register("foo")  # no start in between
        assert elapsed_stale > timer.times["foo"][0]  # stale: wall time keeps growing


class TestNumpyLoader:
    def test_arange(self):
        data = yaml.load("values: !np.arange [0, 5, 1]", Loader=NumpyLoader)  # noqa: S506
        np.testing.assert_array_equal(data["values"], np.arange(0, 5, 1))

    def test_linspace(self):
        data = yaml.load("values: !np.linspace [0, 1, 5]", Loader=NumpyLoader)  # noqa: S506
        np.testing.assert_array_almost_equal(data["values"], np.linspace(0, 1, 5))
