import logging
from pathlib import Path

import pytest

from acm.utils.abacus import load_abacus_cosmologies, get_abacus_phases

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
        with pytest.raises(ValueError):
            get_abacus_phases("nonexistent_directory", z=0.5, cosmo=0)
            
    def test_absolute_paths(self, tmp_path):
        """test that the function returns absolute paths."""
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