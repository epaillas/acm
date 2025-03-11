import matplotlib.pyplot as plt
from acm.data.io_tools import read_lhc, read_covariance
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def plot_cosmologies():
    fig, ax = plt.subplots(figsize=(4, 3))
    nhod = int(len(lhc_y) / 85)
    for i in range(85):
        ax.plot(lhc_y[nhod * i])
    ax.set_xlabel('bin number', fontsize=15)
    ax.set_ylabel(r'$X$', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/e/epaillas/emc/plots/training_samples/{statistic}_cosmologies.pdf')

    
if __name__ == '__main__':
    statistic = 'knn'
    select_filters = {}
    slice_filters = {}

    lhc_x, lhc_y, coords = read_lhc(statistics=[statistic],
                                    select_filters=select_filters,
                                    slice_filters=slice_filters)
    print(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    covariance_matrix, n_sim = read_covariance(statistics=[statistic],
                                                select_filters=select_filters,
                                                slice_filters=slice_filters)
    print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    plot_cosmologies()