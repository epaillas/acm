import numpy as np
from pycorr import TwoPointCorrelationFunction
from pathlib import Path
import pandas


def read_diffsky(statistic='tpcf'):
    if statistic == 'tpcf':
        data_dir = f'/pscratch/sd/e/epaillas/emc/data_vectors/diffsky/tpcf/z{redshift}'
        data_fn = Path(data_dir) / f'tpcf_galsampled_diffsky_mock_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.npy'
        data = TwoPointCorrelationFunction.load(data_fn)[::4]
        s, multipoles = data(ells=(0, 2), return_sep=True)
        y = np.concatenate(multipoles)
    elif statistic == 'dsc_conf':
        multipoles_stat = []
        for stat in ['quantile_data_correlation', 'quantile_correlation']:
            data_dir = f'/pscratch/sd/e/epaillas/emc/data_vectors/diffsky/dsc_conf/z{redshift}'
            data_fn = Path(data_dir) / f'{stat}_galsampled_diffsky_mock_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.npy'
            data = np.load(data_fn, allow_pickle=True)
            multipoles_quantiles = []
            for q in [0, 1, 3, 4]:
                result = data[q][::4]
                # result.select((0, 150))
                s, multipoles = result(ells=(0, 2), return_sep=True)
                multipoles_quantiles.append(np.concatenate(multipoles))
            multipoles_stat.append(np.concatenate(multipoles_quantiles))
        y = np.concatenate(multipoles_stat)
    return s, y

def read_covfiles(statistic='tpcf'):
    y = []
    if statistic == 'wp':
        data_dir = Path('/pscratch/sd/e/epaillas/emc/covariance_sets/wp/z0.5/yuan23_prior')
        data_fns = list(data_dir.glob('wp_ph*_hod466.npy'))
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)
            s, wp = data.sep, data.corr
            y.append(wp)
        y = np.array(y)
    elif statistic == 'tpcf':
        data_dir = Path('/pscratch/sd/e/epaillas/emc/covariance_sets/tpcf/z0.5/yuan23_prior')
        data_fns = list(data_dir.glob('tpcf_ph*_hod466.npy'))
        for data_fn in data_fns:
            data = TwoPointCorrelationFunction.load(data_fn)[::4]
            s, multipoles = data(ells=(0, 2), return_sep=True)
            y.append(np.concatenate(multipoles))
    elif statistic == 'dsc_conf':
        for phase in range(3000, 5000):
            print(phase)
            multipoles_stat = []
            for stat in ['quantile_data_correlation', 'quantile_correlation']:
            # for stat in ['quantile_data_correlation']:
                data_dir = Path(f'/pscratch/sd/e/epaillas/emc/covariance_sets/density_split/{stat}/z0.5/yuan23_prior')
                data_fn = data_dir / f'{stat}_ph{phase:03}_hod466.npy'
                if not data_fn.exists():
                    break
                data = np.load(data_fn, allow_pickle=True)
                multipoles_quantiles = []
                for q in [0, 1, 3, 4]:
                    result = data[q][::4]
                    # result.select((0, 150))
                    multipoles = result(ells=(0, 2))
                    multipoles_quantiles.append(np.concatenate(multipoles))
                multipoles_stat.append(np.concatenate(multipoles_quantiles))
            else:
                y.append(np.concatenate(multipoles_stat))
    return np.asarray(y)


redshift = 0.5
phases = [1, 2]
adict = {0.5: '67120', 0.8: '54980'}
galsample = 'mass_conc'
version = 0.3

for phase_idx in phases:
    s, diffsky_y = read_diffsky(statistic='dsc_conf')
    print(f'Loaded Difftsky with shape: {diffsky_y.shape}')
    cov_y = read_covfiles(statistic='dsc_conf')
    print(f'Loaded covariance with shape: {cov_y.shape}')
    save_dir = f'/pscratch/sd/e/epaillas/emc/data_vectors/diffsky/dsc_conf/z{redshift}/'
    save_fn = Path(save_dir) / f'dsc_conf_galsampled_diffsky_mock_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.npy'
    cout = {'s': s, 'diffsky_y': diffsky_y, 'cov_y': cov_y}
    np.save(save_fn, cout)

    # s, diffsky_y = read_diffsky(statistic='tpcf')
    # print(f'Loaded Difftsky with shape: {diffsky_y.shape}')
    # cov_y = read_covfiles(statistic='tpcf')
    # print(f'Loaded covariance with shape: {cov_y.shape}')
    # save_dir = f'/pscratch/sd/e/epaillas/emc/data_vectors/diffsky/tpcf/z{redshift}/'
    # save_fn = Path(save_dir) / f'tpcf_galsampled_diffsky_mock_{adict[redshift]}_fixedAmp_{phase_idx:03}_{galsample}_v{version}.npy'
    # cout = {'s': s, 'diffsky_y': diffsky_y, 'cov_y': cov_y}
    # np.save(save_fn, cout)