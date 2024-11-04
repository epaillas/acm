import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
plt.style.use(["science", "vibrant"])

def get_ranks(
    theta_posterior_samples,
    theta,
):
    n_posteriors = theta.shape[0]
    ndim = theta.shape[1]
    ranks, mus, stds = [], [], []
    for i in range(n_posteriors):
        posterior_samples = theta_posterior_samples[i]
        mu, std = posterior_samples[i].mean(axis=0), posterior_samples.std(axis=0)
        rank = [(posterior_samples[:, j] < theta[i, j]).sum() for j in range(ndim)]
        mus.append(mu)
        stds.append(std)
        ranks.append(rank)
    mus, stds, ranks = np.array(mus), np.array(stds), np.array(ranks)
    return mus, stds, ranks

def plot_coverage(ranks, labels, plotscatter=True):
    ncounts = ranks.shape[0]
    npars = ranks.shape[-1]
    unicov = [np.sort(np.random.uniform(0, 1, ncounts)) for j in range(30)]

    fig, ax = plt.subplots(
        figsize=(3.5, 2.6),
    )
    cmap = matplotlib.cm.get_cmap("coolwarm")
    colors = cmap(np.linspace(0.01, 0.99, len(labels)))
    ax.plot([0,1],[0,1], linestyle='dashed', color='gray')
    for i in range(npars):
        xr = np.sort(ranks[:, i])
        xr = xr / xr[-1]
        cdf = np.arange(xr.size) / xr.size
        ax.plot(xr, cdf, lw=1.25, label=labels[i], color=colors[i])
    ax.set_xlabel("Confidence Level")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3)
    ax.set_ylabel("Empirical Coverage")
    ax.text(
        0.1,
        0.85,
        "Conservative",
        ha="left",
        va="top",
        transform=ax.transAxes,
        style="italic",
        fontsize=9,
    )
    ax.text(
        0.9,
        0.15,
        "Overconfident",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        style="italic",
        fontsize=9,
    )
    if plotscatter:
        for j in range(len(unicov)):
            ax.plot(unicov[j], cdf, lw=1, color="gray", alpha=0.2)
    return ax

def read_chain(data_dir, statistic, params_to_plot):
    data_dir = data_dir / f'{statistic}/oct11/'
    npy_files = list(data_dir.glob('*.npy'))
    thetas, thetas_samples = [], []
    for npy_file in npy_files:
        data = np.load(npy_file, allow_pickle=True).item()
        pnames = data['param_names']
        indices = [list(pnames).index(p) for p in params_to_plot]
        true_params = [data['true_params'][p] for p in params_to_plot]
        thetas.append(true_params)
        thetas_samples.append(data['samples'][:,indices])
    thetas = np.array(thetas)
    thetas_samples = np.array(thetas_samples)
    return data['param_labels'], thetas, thetas_samples

if __name__ == '__main__':
    statistics = ['pk+number_density',]
    params_to_plot = ["omega_cdm", "sigma8_m", "n_s",]
    data_dir = Path('/pscratch/sd/e/epaillas/emc/posteriors/hmc/')

    for statistic in statistics:
        plabels, thetas, thetas_samples = read_chain(data_dir, statistic, params_to_plot)
        mus, stds, ranks = get_ranks(thetas_samples, thetas)
        ax = plot_coverage(
            ranks=ranks,
            labels=[plabels[param] for param in params_to_plot],
        )
        plt.savefig(f"{statistic}_coverage.png", dpi=300)
        plt.savefig(f"{statistic}_coverage.png", dpi=300)
        plt.close()