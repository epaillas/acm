import scipy.stats as st
import numpy as np
from acm.observables import BaseObservable
from sunbird.emulators import FCN
from pycorr import TwoPointCorrelationFunction
from pathlib import Path
from numpyro import distributions as dist
from sunbird.inference.hamiltonian import HMC
import pandas
import matplotlib.pyplot as plt
from getdist import plots, MCSamples



def read_lhc(return_sep=False):
    data_dir = Path('/pscratch/sd/e/epaillas/emc')
    data_fn = Path(data_dir) / 'training_sets/tpcf/z0.5/yuan23_prior/cosmopower/tpcf.npy'
    lhc_y = np.load(data_fn, allow_pickle=True,).item()
    s = lhc_y['s']
    lhc_y = lhc_y['multipoles']
    lhc_x = pandas.read_csv(data_dir / 'hod_params/yuan23/hod_params_yuan23_c000.csv')
    lhc_x_names = list(lhc_x.columns)
    lhc_x_names = [name.replace(' ', '').replace('#', '') for name in lhc_x_names]
    lhc_x = lhc_x.values[:len(lhc_y),:]
    if return_sep:
        return s, lhc_x, lhc_y, lhc_x_names
    return lhc_x, lhc_y

def read_model():
    checkpoint_fn = "/pscratch/sd/e/epaillas/emc/trained_models/tpcf/may9_leaveout_0/best-model-epoch=296-val_loss=0.07.ckpt"
    model = FCN.load_from_checkpoint(checkpoint_fn, strict=True)
    model.eval()
    return model

def read_covariance():
    data_dir = Path('/pscratch/sd/e/epaillas/emc')
    covariance_path = data_dir / 'covariance/tpcf/z0.5/yuan23_prior/'
    n_for_covariance = 1_000
    covariance_files = list(covariance_path.glob('tpcf_ph*.npy'))[:n_for_covariance]
    covariance_y = [
        TwoPointCorrelationFunction.load(file)[::4](ells=(0,2),).reshape(-1) for file in covariance_files
    ]
    prefactor = 1./8.
    return prefactor * np.cov(np.array(covariance_y).T)

covariance_matrix = read_covariance()
precision_matrix = np.linalg.inv(covariance_matrix)

# load the data
s, lhc_x, lhc_y, lhc_x_names = read_lhc(return_sep=True)
lhc_test_x = lhc_x[:3000]
lhc_test_y = lhc_y[:3000]


priors = {
    'logM_cut': dist.Uniform(low=12.5, high=13.7),
    'logM_1': dist.Uniform(low=13.6, high=15.1),
    'sigma': dist.Uniform(low=-2.99, high=0.96),
    'alpha': dist.Uniform(low=0.3, high=1.48),
    'kappa': dist.Uniform(low=0., high=0.99),
    'alpha_c': dist.Uniform(low=0., high=0.61),
    'alpha_s': dist.Uniform(low=0.58, high=1.49),
    's': dist.Uniform(low=-0.98, high=1.),
    'A_cen': dist.Uniform(low=-0.99, high=0.93),
    'A_sat': dist.Uniform(low=-1., high=1.),
    'B_cen': dist.Uniform(low=-0.67, high=0.2),
    'B_sat': dist.Uniform(low=-0.97, high=0.99),
}

getdist_priors = {
    'logM_cut': [12.5, 13.7],
    'logM_1': [13.6, 15.1],
    'sigma': [-2.99, 0.96],
    'alpha': [0.3, 1.48],
    'kappa': [0., 0.99],
    'alpha_c': [0., 0.61],
    'alpha_s': [0.58, 1.49],
    's': [-0.98, 1.],
    'A_cen': [-0.99, 0.93],
    'A_sat': [-1., 1.],
    'B_cen': [-0.67, 0.2],
    'B_sat': [-0.97, 0.99],
}

for idx_fit in range(5, 10):


    # load the model
    model = read_model()
    nn_model, nn_params = model.to_jax()

    hmc = HMC(
        observation=lhc_test_y[idx_fit],
        precision_matrix=precision_matrix,
        nn_theory_model = nn_model,
        nn_parameters = nn_params,
        priors=priors,
    )

    # prior_samples = hmc.sanity_check_prior(n_samples=100)

    # # let's plot some predictions vs truth
    # for i in range(len(prior_samples)):
    #     plt.plot(
    #         s,
    #         s**2*prior_samples[i][:50],
    #         color='lightgray',
    #         alpha=0.5,
    #     )

    # plt.show()

    posterior = hmc()

    posterior_array = np.stack(list(posterior.values()), axis=0)
    chain_getdist = MCSamples(
            samples=posterior_array.T,
            weights=np.ones(posterior_array.shape[-1]),
            names=list(posterior.keys()),
            ranges=getdist_priors,
        )

    try:
        g = plots.get_subplot_plotter()
        g.settings.constrained_layout = True
        g.settings.axis_marker_lw = 1.0
        g.settings.axis_marker_ls = ":"
        g.settings.title_limit_labels = False
        g.settings.axis_marker_color = "k"
        g.settings.legend_colored_text = True
        g.settings.figure_legend_frame = False
        g.settings.linewidth_contour = 1.0
        g.settings.legend_fontsize = 22
        g.settings.axes_fontsize = 16
        g.settings.axes_labelsize = 20
        g.settings.axis_tick_x_rotation = 45
        g.settings.axis_tick_max_labels = 6

        g.triangle_plot(
            roots=[chain_getdist],
            filled=True,
            markers=dict(zip(lhc_x_names, lhc_test_x[idx_fit]))
        )
        plt.savefig(f'posterior_hod{idx_fit}.pdf', bbox_inches='tight')
        # plt.show()
    except:
        continue