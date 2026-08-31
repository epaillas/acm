import json
import os
from unittest.mock import patch

import pytest

from acm.utils.paths.reader import list_registry_files, lookup_registry_path

# ruff: noqa: ANN001, ANN201, ARG002, D101, D102, D103, INP001, S101

#%% Fixtrures
MODULE_NAME = "acm.utils.paths.reader"

YAML_CONTENT = """\
top:
  mid:
    leaf: 42
flat_key: hello
"""

@pytest.fixture
def nersc_env():
    """Enable require_nersc by setting NERSC_HOST=perlmutter."""
    with patch.dict(os.environ, {"NERSC_HOST": "perlmutter"}):
        yield

@pytest.fixture
def yaml_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(YAML_CONTENT)
    return p

#%% Test classes

class TestLookupRegistry:

    def test_single_key(self, nersc_env, yaml_file):
        result = lookup_registry_path(str(yaml_file), "flat_key")
        assert result == "hello"

    def test_nested_keys(self, nersc_env, yaml_file):
        result = lookup_registry_path(str(yaml_file), "top", "mid", "leaf")
        assert result == 42

    def test_no_keys_returns_full_data(self, nersc_env, yaml_file):
        result = lookup_registry_path(str(yaml_file))
        assert result == {"top": {"mid": {"leaf": 42}}, "flat_key": "hello"}

    def test_invalid_key_raises(self, nersc_env, yaml_file):
        with pytest.raises(KeyError, match="Invalid key path"):
            lookup_registry_path(str(yaml_file), "nonexistent")

    def test_partial_invalid_path_raises(self, nersc_env, yaml_file):
        with pytest.raises(KeyError, match="Invalid key path"):
            lookup_registry_path(str(yaml_file), "top", "mid", "leaf", "too_deep")

    def test_relative_to_cwd(self, nersc_env, tmp_path, monkeypatch, yaml_file):
        monkeypatch.chdir(tmp_path)
        result = lookup_registry_path("config.yaml", "flat_key")
        assert result == "hello"

    def test_missing_file_raises(self, nersc_env):
        with pytest.raises(FileNotFoundError):
            lookup_registry_path("definitely_does_not_exist.yaml", "key")

    def test_custom_loader(self, nersc_env, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"a": {"b": 1}}')

        result = lookup_registry_path(str(p), "a", "b", loader=json.load)
        assert result == 1

    def test_raises_outside_nersc(self, yaml_file):
        with patch.dict(os.environ, {}, clear=True), pytest.raises(OSError):  # noqa: PT011
            lookup_registry_path(str(yaml_file), "flat_key")

class TestListRegistry:

    def test_default_ext(self):
        """Real package directory should contain at least the yaml files used by lookup_registry_path."""
        files = list_registry_files()
        assert isinstance(files, list)
        assert all(f.endswith((".yaml", ".yml")) for f in files)

    def test_custom_ext(self, tmp_path):
        """Use a mocked __file__ location to control the directory contents."""
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.csv").write_text("")
        (tmp_path / "c.yaml").write_text("")

        with patch(f"{MODULE_NAME}.__file__", str(tmp_path / "module.py")):
            files = list_registry_files(ext=(".txt", ".csv"))

        assert sorted(files) == ["a.txt", "b.csv"]

    def test_recursive_files(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "top.yaml").write_text("")
        (tmp_path / "sub" / "nested.yaml").write_text("")

        with patch(f"{MODULE_NAME}.__file__", str(tmp_path / "module.py")):
            non_recursive = list_registry_files(recursive=False)
            recursive = list_registry_files(recursive=True)

        assert non_recursive == ["top.yaml"]
        assert sorted(recursive) == ["nested.yaml", "top.yaml"]

    def test_no_matched_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("")

        with patch(f"{MODULE_NAME}.__file__", str(tmp_path / "module.py")):
            files = list_registry_files(ext=(".yaml",))

        assert files == []
