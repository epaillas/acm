import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_chain_fn(statistic, cosmo_idx, hod_idx):
    base_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/'
    chain_dir = Path(base_dir) / f'c{cosmo_idx:03}_hod{hod_idx:03}/base_lcdm'
    return Path(chain_dir) / f'{statistic}_k0.00-0.50_chain.npy'

def get_samples(statistic, cosmo_idx, hod_idx):
    chain_fn = get_chain_fn(statistic, cosmo_idx, hod_idx)
    print('Loading:', chain_fn)
    data = np.load(chain_fn, allow_pickle=True).item()
    samples = MCSamples(
                samples=data['samples'],
                weights=data['weights'],
                names=data['names'],
                ranges=data['ranges'],
                labels=[data['labels'][n] for n in data['names']],
            )
    return samples, data

def scatter_plot(statistic, params=['omega_cdm', 'sigma8_m']):
    fig, ax = plt.subplots(figsize=(4, 4))
    cosmo_idx = 0

    means = []
    for hod_idx in range(100):
        samples, data = get_samples(statistic, cosmo_idx, hod_idx)
        means.append(samples.mean(params))
    means = np.array(means)

    samples = MCSamples(
        samples=means,
        names=list(params),
        labels=[r'\omega_cdm', '\sigma_8'],
    )

    g = plots.get_subplot_plotter()

    g.triangle_plot(
        roots=samples,
        markers=data['markers'],
        params=params,
    )
    g.fig.axes[0].scatter(means[:, 0], means[:, 1], s=1, color='black')
    plt.savefig('test.pdf')


if __name__ == '__main__':
    statistic = 'pk+number_density'

    scatter_plot(statistic)
