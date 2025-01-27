import numpy as np
from pycorr import TwoPointCorrelationFunction
from pathlib import Path
import pandas


def read_lhc(statistic='tpcf', n_hod=100, phase_idx=0, return_sep=False):
    lhc_y = []
    if statistic == 'number_density':
        for cosmo_idx in cosmos:
            data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/number_density/cosmo+hod/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            data_fn = Path(data_dir) / 'nden_downsampled.npy'
            data = np.load(data_fn)
            for hod in range(n_hod):
                lhc_y.append([data[hod]])
    if statistic == 'tpcf':
        for cosmo_idx in cosmos:
            print(cosmo_idx)
            data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/tpcf/cosmo+hod_bugfix/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            for hod in range(n_hod):
                data_fn = Path(data_dir) / f'tpcf_hod{hod:03}.npy'
                data = TwoPointCorrelationFunction.load(data_fn)[::4]
                s, multipoles = data(ells=(0, 2), return_sep=True)
                lhc_y.append(np.concatenate(multipoles))
    elif statistic == 'wp':
        for cosmo_idx in cosmos:
            print(cosmo_idx)
            data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/wp/cosmo+hod/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            for hod in range(n_hod):
                data_fn = Path(data_dir) / f'wp_hod{hod:03}.npy'
                data = TwoPointCorrelationFunction.load(data_fn)
                s, wp = data.sep, data.corr
                lhc_y.append(wp)
    elif statistic == 'dsc_conf':
        for cosmo_idx in cosmos:
            print(cosmo_idx)
            for hod in range(n_hod):
                multipoles_stat = []
                for stat in ['quantile_data_correlation', 'quantile_correlation']:
                    data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/dsc_conf/cosmo+hod/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
                    data_fn = Path(data_dir) / f'{stat}_hod{hod:03}.npy'
                    data = np.load(data_fn, allow_pickle=True)
                    multipoles_quantiles = []
                    for q in [0, 1, 3, 4]:
                        result = data[q][::4]
                        # result.select((0, 150))
                        s, multipoles = result(ells=(0, 2), return_sep=True)
                        multipoles_quantiles.append(np.concatenate(multipoles))
                    multipoles_stat.append(np.concatenate(multipoles_quantiles))
                lhc_y.append(np.concatenate(multipoles_stat))
    elif statistic == 'knn':
        for cosmo_idx in cosmos:
            print(cosmo_idx)
            data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/knn/cosmo+hod/z0.5/yuan23_prior/c{cosmo_idx:03}_ph{phase_idx:03}/seed0/'
            for hod in range(n_hod):
                data_fn = Path(data_dir) / f'knn_masked_hod{hod:03}.npy'
                # data = np.load(data_fn).mean(axis=0)  # average 3 los
                data = np.load(data_fn)[0]
                lhc_y.append(data)
    elif statistic == 'mst':
        data_dir = '/pscratch/sd/k/knaidoo/ACM/MockChallenge/Outputs/'
        for cosmo_idx in cosmos:
            for hod_idx in range(n_hod):
                data_fn = Path(data_dir) / f'emulator_{cosmo_idx}_hod_{hod_idx}_smooth_10p0.npz'
                data = np.load(data_fn)
                lhc_y.append(np.concatenate([data['yd'], data['yl'], data['yb'], data['ys']]))
    elif statistic == 'minkowski':
        data_dir = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod/'
        data_fn = Path(data_dir) / 'minkowski_dummy.npy'
        data = np.load(data_fn, allow_pickle=True).item()
        s = data['delta']
        for cosmo_idx in cosmos:
            for hod_idx in range(n_hod):
                if hod_idx in data[f'c{cosmo_idx:03}_index']:
                    where = np.where(data[f'c{cosmo_idx:03}_index'] == hod_idx)[0][0]
                    lhc_y.append(data[f'c{cosmo_idx:03}_y'][where])
                else:
                    lhc_y.append(np.zeros_like(data['lhc_train_y'][0]))
    lhc_y = np.asarray(lhc_y)
    lhc_x, lhc_x_names = read_lhc_x(n_hod)
    lhc_x = lhc_x[:len(lhc_y), :]
    if return_sep:
        return s, lhc_x, lhc_y, lhc_x_names
    return lhc_x, lhc_y, lhc_x_names

def read_lhc_x(n_hod=100):
    lhc_x = []
    for cosmo_idx in cosmos:
        data_dir = '/pscratch/sd/e/epaillas/emc/cosmo+hod_params/'
        data_fn = Path(data_dir) / f'AbacusSummit_c{cosmo_idx:03}.csv'
        lhc_x_i = pandas.read_csv(data_fn)
        lhc_x_names = list(lhc_x_i.columns)
        lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
        lhc_x.append(lhc_x_i.values[:n_hod, :])
    lhc_x = np.concatenate(lhc_x)
    return lhc_x, lhc_x_names

def read_lhc_phases(nphases):
    lhc_y = []
    lhc_x = []
    for phase_idx in range(nphases):
        data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/tpcf/z0.5/yuan23_prior/c000_ph{phase_idx:03}/seed0/'
        data_fn = Path(data_dir) / f'tpcf_lhc.npy'
        data = np.load(data_fn, allow_pickle=True).item()
        lhc_y.append(data['lhc_y'])
        phase_x = [phase_idx] * len(data['lhc_x'])
        lhc_x.append(np.hstack([data['lhc_x'], np.array(phase_x)[:, None]]))

    lhc_y = np.concatenate(lhc_y)
    lhc_x = np.concatenate(lhc_x)

def read_covfiles(statistic='tpcf'):
    y = []
    if statistic == 'number_density':
        data_dir = Path('/pscratch/sd/e/epaillas/emc/covariance_sets/number_density/z0.5/yuan23_prior')
        data_fns = list(data_dir.glob('number_density_ph*_hod030.npy'))
        for data_fn in data_fns:
            data = np.load(data_fn)
            y.append([data])
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
                data_dir = Path(f'/pscratch/sd/e/epaillas/emc/covariance_sets/density_split/{stat}/z0.5/yuan23_prior')
                data_fn = data_dir / f'{stat}_ph{phase:03}_hod466.npy'
                if not data_fn.exists():
                    break
                data = np.load(data_fn, allow_pickle=True)
                multipoles_quantiles = []
                for q in [0, 1, 3, 4]:
                    result = data[q][::4]
                    multipoles = result(ells=(0, 2))
                    multipoles_quantiles.append(np.concatenate(multipoles))
                multipoles_stat.append(np.concatenate(multipoles_quantiles))
            else:
                y.append(np.concatenate(multipoles_stat))
    elif statistic == 'knn':
        data_dir = Path('/pscratch/sd/e/epaillas/emc/covariance_sets/knn/z0.5/yuan23_prior')
        data_fns = list(data_dir.glob('knn_masked_ph*_hod466.npy'))
        for data_fn in data_fns:
            data = np.load(data_fn)
            y.append(data)
    elif statistic == 'mst':
        data_dir = Path('/pscratch/sd/k/knaidoo/ACM/MockChallenge/Outputs/')
        for phase in range(3000, 5000):
            data_fn = Path(data_dir) / f'covariance_mocks_{phase}_10p0.npz'
            if not data_fn.exists():
                continue
            data = np.load(data_fn)
            y.append(np.concatenate([data['yd'], data['yl'], data['yb'], data['ys']]))
    elif statistic == 'minkowski':
        data_dir = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod'
        data_fn = Path(data_dir) / 'minkowski_dummy.npy'
        data = np.load(data_fn, allow_pickle=True).item()
        return data['y_cov']
    return np.asarray(y)

cosmos = list(range(0, 5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182))
# cosmos = list(range(0, 3)) + list(range(100, 127)) + list(range(130, 182))

statistics = ['minkowski']

if 'minkowski' in statistics:
    delta, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='minkowski', n_hod=200, return_sep=True)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    cov_y = read_covfiles(statistic='minkowski')
    print(f'Loaded covariance with shape: {cov_y.shape}')
    save_dir = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/covariance_sets/small_box'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / 'minkowski.npy'
    np.save(save_fn, {'delta': delta, 'cov_y': cov_y})
    save_dir = '/pscratch/sd/e/epaillas/emc/v1.1/abacus/training_sets/cosmo+hod'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / 'minkowski.npy'
    cout = {'delta': delta, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
    np.save(save_fn, cout)

if 'number_density' in statistics:
    lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='number_density', n_hod=100)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    cov_y = read_covfiles(statistic='number_density')
    print(f'Loaded covariance with shape: {cov_y.shape}')
    save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/number_density/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / 'number_density_lhc.npy'
    cout = {'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
    np.save(save_fn, cout)

if 'wp' in statistics:
    r_p, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='wp', n_hod=100, return_sep=True)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    cov_y = read_covfiles(statistic='wp')
    print(f'Loaded wp covariance with shape: {cov_y.shape}')
    save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/wp/cosmo+hod/z0.5/yuan23_prior/ph000/seed0/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / 'wp_lhc.npy'
    cout = {'rp': r_p, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
    np.save(save_fn, cout)

if 'tpcf' in statistics:
    s, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='tpcf', n_hod=100, return_sep=True)
    print(f'Loaded 2PCF LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    cov_y = read_covfiles(statistic='tpcf')
    print(f'Loaded 2PCF covariance with shape: {cov_y.shape}')
    save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/tpcf/cosmo+hod_bugfix/z0.5/yuan23_prior/ph000/seed0/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / 'tpcf_lhc.npy'
    cout = {'s': s, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
    np.save(save_fn, cout)

if 'dsc_conf' in statistics:
    s, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='dsc_conf', n_hod=100, return_sep=True)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    cov_y = read_covfiles(statistic='dsc_conf')
    print(f'Loaded covariance with shape: {cov_y.shape}')
    save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/dsc_conf/cosmo+hod/z0.5/yuan23_prior/ph000/seed0'
    save_fn = Path(save_dir) / 'dsc_conf_lhc.npy'
    cout = {'s': s, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
    np.save(save_fn, cout)

if 'knn' in statistics:
    lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='knn', n_hod=100, return_sep=False)
    rp = np.logspace(-0.2, 1.8, 8)
    pi = np.logspace(-0.3, 1.5, 5)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    cov_y = read_covfiles(statistic='knn')
    print(f'Loaded covariance with shape: {cov_y.shape}')
    save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/knn/cosmo+hod/z0.5/yuan23_prior/ph000/seed0'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / 'knn_lhc.npy'
    cout = {'rp': rp, 'pi': pi, 'lhc_x': lhc_x, 'lhc_y': lhc_y,
            'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
    np.save(save_fn, cout)

if 'mst' in statistics:
    lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='mst', n_hod=350, return_sep=False)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    cov_y = read_covfiles(statistic='mst')
    print(f'Loaded covariance with shape: {cov_y.shape}')
    save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/mst/cosmo+hod/z0.5/yuan23_prior/ph000/seed0'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_fn = Path(save_dir) / 'mst_lhc.npy'
    coeff_idx = np.arange(lhc_y.shape[1])
    cout = {'coeff_idx': coeff_idx, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
    np.save(save_fn, cout)