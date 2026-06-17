import argparse

import pytest
import yaml
import numpy as np
from unittest.mock import patch

from acm.utils.scripts import (
    detect_gpu,
    get_nthreads,
    load_parser_default,
    apply_parser_default,
    dump_config,
    NumpyLoader,
)
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
        with pytest.raises(ValueError):
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
        with patch("sys.argv", ["prog"]):
            with pytest.raises(ValueError, match="config"):
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
        with patch("sys.argv", ["prog"]):
            with pytest.raises(ValueError, match="dump_config"):
                dump_config(parser)
    
    def test_exits_when_true(self, capsys):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dump_config", action="store_true")
        parser.add_argument("--config", default=None)
        with patch("sys.argv", ["prog", "--dump_config"]):
            with pytest.raises(SystemExit):
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
        with patch("sys.argv", ["prog", "--dump_config"]):
            with pytest.raises(SystemExit):
                dump_config(parser)
        captured = capsys.readouterr().out
        assert "alpha: 3.14" in captured
        assert "label: test" in captured


class TestNumpyLoader:
    def test_arange(self):
        data = yaml.load("values: !np.arange [0, 5, 1]", Loader=NumpyLoader)
        np.testing.assert_array_equal(data["values"], np.arange(0, 5, 1))

    def test_linspace(self):
        data = yaml.load("values: !np.linspace [0, 1, 5]", Loader=NumpyLoader)
        np.testing.assert_array_almost_equal(data["values"], np.linspace(0, 1, 5))