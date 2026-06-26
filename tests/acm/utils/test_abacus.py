import logging
from pathlib import Path

import pytest

from acm.utils.abacus import (
    get_abacus_phases,
    get_abacus_simname,
    load_abacus_cosmologies,
    map_params,
)

# ruff: noqa: ANN001, ANN201, D101, D102, D103, INP001, S101

#%% Fixtures
params = ["param1", "param2", "param3"]
dummy_csv = """\
root,param1,param2,param3
abacus_cosm000,1.0,2.0,3.0
abacus_cosm001,4.0,5.0,6.0
abacus_cosm002,7.0,8.0,9.0
abacus_cosm003,10.0,11.0,12.0
"""

@pytest.fixture
def csv_file(tmp_path):
    """Write the dummy CSV to a temp file and return its path."""
    p = tmp_path / "cosmologies.csv"
    p.write_text(dummy_csv)
    return str(p)

def _make_phase_dir(base: Path, cosmo: int, phase: int, z: float, subdir: str = "data") -> Path:
    """Create a dummy phase directory tree and return the deepest path."""
    leaf = base / f"AbacusSummit_small_c{cosmo:03d}_ph{phase:03d}" / subdir / f"z{z:.3f}"
    leaf.mkdir(parents=True)
    return leaf

test_data = [
    ("base", 0, 0, "AbacusSummit_base_c000_ph000"),
    ("small", 1, 2, "AbacusSummit_small_c001_ph002"),
    ("base", 5, 1000, "AbacusSummit_base_c005_ph1000"),
    ("png", 1, 2, "Abacus_pngbase_c001_ph002"),
]

@pytest.mark.parametrize(("simtype", "cosmo", "phase", "expected"), test_data)
def test_get_abacus_simname(simtype, cosmo, phase, expected):
    assert get_abacus_simname(simtype, cosmo, phase) == expected

class TestMapParams:

    def test_dict_default_mapping(self):
        params = {"logM_1": 13.5, "A_cen": 0.1}
        expected = {"logM1": 13.5, "Acent": 0.1}
        assert map_params(params) == expected

    def test_list_default_mapping(self):
        params = ["logM_1", "A_cen"]
        expected = ["logM1", "Acent"]
        assert map_params(params) == expected

    def test_dict_mutates(self):
        """Dicts are mutated in-place."""
        params = {"logM_1": 13.5, "A_cen": 0.1}
        result = map_params(params)
        assert params == result

    def test_extra_unchanged(self):
        "Unmapped keys should stay inchanged."
        params = {"unrelated_key": 42}
        expected = {"unrelated_key": 42}
        assert map_params(params) == expected

    def test_custom_mapping(self):
        custom_mapping = {"foo": ["bar", "baz"]}
        params = {"baz": 1}
        result = map_params(params, mapping=custom_mapping)
        assert result == {"foo": 1}

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid type"):
            map_params("not_a_dict_or_list")  # ty:ignore[no-matching-overload]

    def test_empty_dict(self):
        assert map_params({}) == {}

    def test_empty_list(self):
        assert map_params([]) == []

class TestLoadAbacusCosmologies:
    """Tests for the load_abacus_cosmologies function."""

    def test_returns_selected_cosmologies(self, csv_file):
        """Test that the function returns only the selected cosmologies with the correct keys."""
        result = load_abacus_cosmologies(
            filename=csv_file,
            cosmologies=[0, 2],
            parameters=params,
        )
        assert len(result) == 2
        assert set(result.keys()) == {"c000", "c002"}

    def test_exclude_unselected_cosmologies(self, csv_file):
        """Test that the function excludes unselected cosmologies."""
        result = load_abacus_cosmologies(
            filename=csv_file,
            cosmologies=[0, 2],
            parameters=params,
        )
        assert "c001" not in result
        assert "c003" not in result

    def test_correct_values(self, csv_file):
        """Test that the function returns the correct parameter values for the selected cosmologies."""
        result = load_abacus_cosmologies(
            filename=csv_file,
            cosmologies=[0],
            parameters=params,
        )
        assert result["c000"]["param1"] == 1.0
        assert result["c000"]["param2"] == 2.0
        assert result["c000"]["param3"] == 3.0

    def test_mapping(self, csv_file):
        """Test that the function correctly renames parameters according to the mapping."""
        mapping = {"param1": "p1"}
        result = load_abacus_cosmologies(
            filename=csv_file,
            cosmologies=[0],
            parameters=params,
            mapping=mapping,
        )
        assert "p1" in result["c000"]
        assert "param1" not in result["c000"]
        assert result["c000"]["p1"] == 1.0

    def test_nonexistent_cosmology(self, csv_file):
        """Test that the function returns an empty dict for non-existent cosmologies."""
        mapping = {"param1": "p1"}
        result = load_abacus_cosmologies(
            filename=csv_file,
            cosmologies=[999],
            parameters=params,
            mapping=mapping, # Add mapping here to ensure no crashes on empty dict
        )
        assert result == {}

    def test_raises_on_missing_parameter(self, csv_file):
        """Test that the function raises an error if a requested parameter does not exist."""
        with pytest.raises(ValueError):
            load_abacus_cosmologies(
                filename=csv_file,
                cosmologies=[0],
                parameters=["nonexistent_param"],
            )

    def test_raises_on_missing_file(self):
        """Test that the function raises an error if the file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_abacus_cosmologies(
                filename="nonexistent_file.csv",
                cosmologies=[0],
                parameters=params,
            )

class TestGetAbacusPhases:
    """Tests for the get_abacus_phases function."""

    def test_single_phase(self, tmp_path):
        """Test that the function correctly identifies a single phase."""
        _make_phase_dir(tmp_path, cosmo=0, phase=1, z=0.500)
        fns, phases = get_abacus_phases(tmp_path, z=0.5, cosmo=0)
        assert len(fns) == 1
        assert phases == [1]

    def test_multiple_phases(self, tmp_path):
        """Test that the function correctly identifies multiple phases."""
        for phase in [1, 2, 3]:
            _make_phase_dir(tmp_path, cosmo=0, phase=phase, z=0.500)
        fns, phases = get_abacus_phases(tmp_path, z=0.5, cosmo=0)
        assert sorted(phases) == [1, 2, 3]
        assert len(fns) == 3

    def test_sorted_phases(self, tmp_path):
        """Test that the function returns phases in sorted order."""
        for phase in [3, 1, 2]:
            _make_phase_dir(tmp_path, cosmo=0, phase=phase, z=0.500)
        _, phases = get_abacus_phases(tmp_path, z=0.5, cosmo=0)
        assert phases == [1, 2, 3]

    def test_empty_for_wrong_redshift(self, tmp_path):
        """Test that the function returns empty lists if no phases match the redshift."""
        _make_phase_dir(tmp_path, cosmo=0, phase=1, z=0.500)
        fns, phases = get_abacus_phases(tmp_path, z=1.0, cosmo=0)
        assert fns == []
        assert phases == []

    def test_empty_for_wrong_cosmology(self, tmp_path):
        """Test that the function returns empty lists if no phases match the cosmology."""
        _make_phase_dir(tmp_path, cosmo=0, phase=1, z=0.500)
        fns, phases = get_abacus_phases(tmp_path, z=0.5, cosmo=1)
        assert fns == []
        assert phases == []

    def test_ignores_non_matching_files(self, tmp_path):
        """Test that the function ignores files that do not match the expected pattern."""
        # Create a valid phase directory
        _make_phase_dir(tmp_path, cosmo=0, phase=1, z=0.500)
        # Create some non-matching files
        (tmp_path / "random_file.txt").write_text("This should be ignored.")
        (tmp_path / "AbacusSummit_small_c000_ph001/data/z0.500/extra_file.txt").write_text("This should also be ignored.")

        fns, phases = get_abacus_phases(tmp_path, z=0.5, cosmo=0)
        assert len(fns) == 1
        assert phases == [1]

    def test_raises_on_nonexistent_directory(self):
        """Test that the function raises an error if the phase directory does not exist."""
        with pytest.raises(ValueError, match="not a valid directory"):
            get_abacus_phases("nonexistent_directory", z=0.5, cosmo=0)

    def test_absolute_paths(self, tmp_path):
        """Test that the function returns absolute paths."""
        _make_phase_dir(tmp_path, cosmo=0, phase=1, z=0.500)
        fns, _ = get_abacus_phases(tmp_path, z=0.5, cosmo=0)
        assert all(fn.is_absolute() for fn in fns)

    def test_match_glob_and_no_re(self, tmp_path, caplog):
        """Test that a file matching the glob pattern but not the regex is ignored with a warning level log."""
        _make_phase_dir(tmp_path, cosmo=0, phase=1, z=0.500)
        extra_dir = tmp_path / "AbacusSummit_small_c000_ph001_some_other_stuff/data/z0.500"
        extra_dir.mkdir(parents=True)

        with caplog.at_level(logging.WARNING):
            fns, phases = get_abacus_phases(tmp_path, z=0.5, cosmo=0)
        assert "will be skipped" in caplog.text
        assert len(fns) == 1
        assert phases == [1]
