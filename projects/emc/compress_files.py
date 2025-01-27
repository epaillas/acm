import numpy as np
from pycorr import TwoPointCorrelationFunction
from pathlib import Path
import pandas


def read_lhc(statistic='tpcf', n_hod=100, phase_idx=0, return_sep=False):
    lhc_y = []
    if statistic == 'tpcf':
        data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/tpcf/z0.5/yuan23_prior/c000_ph{phase_idx:03}/seed0/'
        for hod in range(n_hod):
            data_fn = Path(data_dir) / f'tpcf_hod{hod:03}.npy'
            data = TwoPointCorrelationFunction.load(data_fn)[::4]
            s, multipoles = data(ells=(0, 2), return_sep=True)
            lhc_y.append(np.concatenate(multipoles))
    elif statistic == 'wp':
        data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/wp/z0.5/yuan23_prior/c000_ph{phase_idx:03}/seed0/'
        for hod in range(n_hod):
            data_fn = Path(data_dir) / f'wp_hod{hod:03}.npy'
            data = TwoPointCorrelationFunction.load(data_fn)
            s, wp = data.sep, data.corr
            lhc_y.append(wp)
    elif statistic == 'density_split':
        for hod in range(n_hod):
            print(hod)
            multipoles_stat = []
            for stat in ['quantile_data_correlation', 'quantile_correlation']:
                data_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/{statistic}/{stat}/z0.5/yuan23_prior/c000_ph{phase_idx:03}/seed0/'
                data_fn = Path(data_dir) / f'{stat}_hod{hod:03}.npy'
                data = np.load(data_fn, allow_pickle=True)
                multipoles_quantiles = []
                for q in [0, 1, 3, 4]:
                    result = data[q][::4]
                    s, multipoles = result(ells=(0, 2), return_sep=True)
                    multipoles_quantiles.append(np.concatenate(multipoles))
                multipoles_stat.append(np.concatenate(multipoles_quantiles))
            lhc_y.append(np.concatenate(multipoles_stat))
    elif statistic == 'number_density':
        data_dir = '/pscratch/sd/e/epaillas/emc/number_density/z0.5/yuan23_prior/small'
        data_fn = Path(data_dir) / 'number_density_downsampled.npy'
        lhc_y = np.load(data_fn)[:n_hod]
    elif statistic == 'voxel_voids':
        for hod in range(n_hod):
            print(hod)
            data_dir = f'/pscratch/sd/t/tsfraser/EMC_CHALLENGE/voxel_multipoles/HOD/voidprior/AbacusSummit_HOD{hod:03}/z0.500/'
            data_fn = Path(data_dir) / f'voxel_multipoles_HOD_Rs10_hod{hod:03}_losz.npy'
            data = TwoPointCorrelationFunction.load(data_fn)[::3]
            s, multipoles = data(ells=(0, 2), return_sep=True)
            lhc_y.append(np.concatenate(multipoles)) 
    elif statistic == 'bispectrum':
        for hod in range(n_hod):
            data_dir = f'/pscratch/sd/j/jerryou/acm/c000_ph{phase_idx:03}/Bk/nmesh512/'
            data_fn = Path(data_dir) / f'hod{hod:03}.npz'
            data = np.load(data_fn)
            s = data['k123']
            b0 = data['b0']
            b2 = data['b2']
            lhc_y.append(np.concatenate([b0, b2]))
        # lhc_y = np.asarray(lhc_y)
        # lhc_x, lhc_x_names = read_lhc_x()
        # lhc_x = lhc_x[n_hod]
        # return s, lhc_x, lhc_y, lhc_x_names
    lhc_y = np.asarray(lhc_y)
    lhc_x, lhc_x_names = read_lhc_x()
    lhc_x = lhc_x[:len(lhc_y), :]
    if return_sep:
        return s, lhc_x, lhc_y, lhc_x_names
    return lhc_x, lhc_y, lhc_x_names

def read_lhc_x():
    data_dir = '/pscratch/sd/e/epaillas/emc/hod_params/yuan23'
    data_fn = Path(data_dir) / 'hod_params_yuan23_c000.csv'
    lhc_x = pandas.read_csv(data_fn)
    lhc_x_names = list(lhc_x.columns)
    lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
    return lhc_x.values, lhc_x_names

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
    print(np.shape(lhc_y), np.shape(lhc_x))
    print(lhc_x[0])
    print(lhc_x[-1])

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
                data_dir = Path(f'/pscratch/sd/e/epaillas/emc/covariance_sets/density_split/{stat}/z0.5/yuan23_prior')
                data_fn = data_dir / f'{stat}_ph{phase:03}_hod466.npy'
                if not data_fn.exists():
                    break
                data = np.load(data_fn, allow_pickle=True)
                multipoles_quantiles = []
                for q in [0, 1, 3, 4]:
                    multipoles = data[q][::4](ells=(0, 2))
                    multipoles_quantiles.append(np.concatenate(multipoles))
                multipoles_stat.append(np.concatenate(multipoles_quantiles))
            else:
                y.append(np.concatenate(multipoles_stat))
    return np.asarray(y)
    

# LHC
# phase_idx = 1
# cov_y = read_covfiles(statistic='tpcf')
# print(f'Loaded 2PCF covariance with shape: {cov_y.shape}')
# for phase_idx in range(25):
#     print(phase_idx)
#     n_hod = 10000 if phase_idx == 0 else 25
    # s, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='tpcf', n_hod=n_hod, return_sep=True, phase_idx=phase_idx)
    # print(f'Loaded 2PCF LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
    # save_dir = f'/pscratch/sd/e/epaillas/emc/training_sets/tpcf/z0.5/yuan23_prior/c000_ph{phase_idx:03}/seed0'
    # save_fn = Path(save_dir) / 'tpcf_lhc.npy'
    # cout = {'s': s, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
    # np.save(save_fn, cout)

# read_lhc_phases(nphases=25)


# lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='number_density', n_hod=50_000, return_sep=False)
# print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
# save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/number_density_downsampled/z0.5/yuan23_prior/c000_ph000/seed0'
# save_fn = Path(save_dir) / 'number_density_downsampled_lhc.npy'
# cout = {'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
# np.save(save_fn, cout)

# r_p, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='wp', n_hod=10000, return_sep=True)
# print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
# save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/wp/z0.5/yuan23_prior/c000_ph000/seed0'
# save_fn = Path(save_dir) / 'wp_lhc.npy'
# cout = {'r_p': r_p, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
# np.save(save_fn, cout)

# s, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='density_split', n_hod=10000, return_sep=True)
# print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
# cov_y = read_covfiles(statistic='dsc_conf')
# print(f'Loaded covariance with shape: {cov_y.shape}')
# save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/dsc_conf/z0.5/yuan23_prior/c000_ph000/seed0'
# save_fn = Path(save_dir) / 'dsc_conf_lhc.npy'
# cout = {'s': s, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names, 'cov_y': cov_y}
# np.save(save_fn, cout)

# s, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='voxel_voids', n_hod=10000, return_sep=True)
# print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
# save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/voxel_voids/z0.5/yuan23_prior/c000_ph000/seed0'
# save_fn = Path(save_dir) / 'voxel_voids_lhc.npy'
# cout = {'s': s, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
# np.save(save_fn, cout)

s, lhc_x, lhc_y, lhc_x_names = read_lhc(statistic='bispectrum', n_hod=900, return_sep=True)
print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')
save_dir = '/pscratch/sd/e/epaillas/emc/training_sets/bispectrum/z0.5/yuan23_prior/c000_ph000/seed0'
save_fn = Path(save_dir) / 'bispectrum_lhc.npy'
cout = {'k123': s, 'lhc_x': lhc_x, 'lhc_y': lhc_y, 'lhc_x_names': lhc_x_names}
np.save(save_fn, cout)


# covariance

# cov_y = read_covfiles(statistic='wp')
# print(f'Loaded covariance with shape: {cov_y.shape}')
# save_dir = '/pscratch/sd/e/epaillas/emc/covariance_sets/wp/z0.5/yuan23_prior'
# save_fn = Path(save_dir) / 'wp_cov.npy'
# np.save(save_fn, cov_y)

# cov_y = read_covfiles(statistic='tpcf')
# print(f'Loaded covariance with shape: {cov_y.shape}')
# save_dir = '/pscratch/sd/e/epaillas/emc/covariance_sets/tpcf/z0.5/yuan23_prior'
# save_fn = Path(save_dir) / 'tpcf_cov.npy'
# np.save(save_fn, cov_y)

# save_dir = '/pscratch/sd/e/epaillas/emc/covariance_sets/dsc_conf/z0.5/yuan23_prior'
# save_fn = Path(save_dir) / 'dsc_conf_cov.npy'
# np.save(save_fn, cov_y)


