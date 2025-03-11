from abc import ABC, abstractmethod
from pathlib import Path


DEFAULT_DIRS = get_default_dirs()

def get_default_dirs():
    with open('paths.yaml', 'r') as f:
        dirs = yaml.safe_load(f)
    return dirs


class BaseSummary(ABC):
    """
    Base class for summary statistics.
    """
    def __init__(self,
                 training_set_dir: str = DEFAULT_DIRS['emc']['training_sets'],
                 coordinates_dir: str = DEFAULT_DIRS['emc']['coordinates'],
                 covariance_set_dir: str = DEFAULT_DIRS['emc']['covariance_sets'],
                 emulator_error_dir: str = DEFAULT_DIRS['emc']['emulator_error'],
                 chains_dir: str = DEFAULT_DIRS['emc']['chains'],
                 diffsky_dir: str = DEFAULT_DIRS['diffsky']['data_vectors']):
                 
        self.training_set_fn = Path(training_set_dir) / self.summary_str / '.npy'
        self.coordinates_fn = Path(coordinates_dir) / self.summary_str / '.npy'
        self.covariance_set_fn = Path(covariance_set_dir) / self.summary_str / '.npy'
        self.emulator_error_fn = Path(emulator_error_dir) / self.summary_str / '.npy'
        self.chains_fn = Path(chains_dir) / self.summary_str / '.npy'
        self.diffsky_fn= Path(diffsky_dir) / self.summary_str / '.npy'

    @abstractmethod
    def coords_lhc_y(self):
        """Coordinates for the LHC y-values (measurements)."""
        pass

    @abstractmethod
    def coords_lhc_x(self):
        """Coordinates for the LHC x-values (parameters)."""
        pass

    @abstractmethod
    def coords_diffsky(self):
        """Coordinates for the Diffsky data vectors."""
        pass

    def get_abscissa(self):
        """Get the x-coordinate associated to a summary statistic
        (e.g. $k$ for the power spectrum multipoles).
        """
        data = self.load_data(self.coordinates_fn, allow_pickle=True).item()
        coord_name = data['coord_name']
        coord_values = data['coord_values']
        return coord_name, coord_values

class PowerSpectrumMultipoles(BaseSummary):
    """
    Class for the power spectrum multipoles.
    """
    def __init__(self):
        self.summary_str = 'pk'
        super().__init__()

    def coords_lhc_y(self):
        coords = {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(250)),
            'multipoles': [0, 2],
            'k': self.get_abscissa()[1],
        }

    def coords_lhc_x(self):
        return {
            'cosmo_idx': list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)),
            'hod_idx': list(range(250)),
        }

    def coords_diffsky(self):
        return {
            'multipoles': [0, 2],
            'k': self.get_abscissa()[1],
        }


class DensitySplitPowerSpectrumMultipoles(BaseSummary):
    def __init__(self):
        pass

class DensitySplitCorrelationFunctionMultipoles(BaseSummary):
    def __init__(self):
        pass

class VoxelVoidsMultipoles(BaseSummary):
    def __init__(self):
        pass

class DelaunaySpheresMultipoles(BaseSummary):
    def __init__(self):
        pass

class WaveletScatteringTransform(BaseSummary):
    def __init__(self):
        pass

class DensityPDF(BaseSummary):
    def __init__(self):
        pass

class CumulativeGeneratingFunction(BaseSummary):
    def __init__(self):
        pass

class MinkowskiFunctionals(BaseSummary):
    def __init__(self):
        pass