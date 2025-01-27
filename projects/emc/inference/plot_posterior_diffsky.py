import numpy as np
from getdist import plots, MCSamples
from pathlib import Path
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


labels = {
    "omega_b": r"$\omega_{\rm b}$",
    "omega_cdm": r"$\omega_{\rm cdm}$",
    "sigma8_m": r"$\sigma_8$",
    "n_s": r"$n_s$",
    "nrun": r"$\alpha_s$",
    "N_ur": r"$N_{\rm ur}$",
    "w0_fld": r"$w_0$",
    "wa_fld": r"$w_a$",
    "logM_1": r"$\log M_1$",
    "logM_cut": r"$\log M_{\rm cut}$",
    "alpha": r"$\alpha$",
    "alpha_s": r"$\alpha_{\rm vel, s}$",
    "alpha_c": r"$\alpha_{\rm vel, c}$",
    "sigma": r"$\log \sigma$",
    "kappa": r"$\kappa$",
    "A_cen": r"$A_{\rm cen}$",
    "A_sat": r"$A_{\rm sat}$",
    "B_cen": r"$B_{\rm cen}$",
    "B_sat": r"$B_{\rm sat}$",
    "s": r"$s$",
    "fsigma8": r"$f \sigma_8$",
    "Omega_m": r"$\Omega_{\rm m}$",
    "H0": r"$H_0$",
}

labels_stats = {
    'dsc_conf': 'Density-Split Multipoles',
    'dsc_conf_cross': 'Density-split (CCF)',
    'tpcf': 'Galaxy 2PCF',
    'pk': 'Galaxy P(k)',
    'tpcf+dsc_conf': 'DSC + Galaxy 2PCF',
}

chains = []
legend_labels = []

smin = 0
phase_idx = 1
truth = {
    'Omega_m': 0.3089,
    'omega_cdm': 0.1188,
    'omega_b': 0.02230,
    'h': 0.6774,
    'n_s': 0.9667,
    'sigma8_m': 0.8147
}

params = ['omega_cdm', 'sigma8_m', 'n_s']

data_dir = f'/pscratch/sd/e/epaillas/emc/posteriors/diffsky/tpcf/base/'
data_fn = Path(data_dir) / f'chain_z0.5_fixedAmp_ph{phase_idx:03}_mass_conc_v0.3_s5.00-152.00.npy'
data = np.load(data_fn, allow_pickle=True).item()
samples = MCSamples(
            samples=data['samples'],
            weights=data['weights'],
            names=data['param_names'],
            ranges=data['param_ranges'],
            labels=[labels[n] for n in data['param_names']],
        )
legend_labels.append(labels_stats['tpcf'])
chains.append(samples)

data_dir = f'/pscratch/sd/e/epaillas/emc/posteriors/diffsky/pk/base/'
data_fn = Path(data_dir) / f'chain_z0.5_fixedAmp_ph001_mass_conc_v0.3_k0.00-0.50.npy'
data = np.load(data_fn, allow_pickle=True).item()
samples = MCSamples(
            samples=data['samples'],
            weights=data['weights'],
            names=data['param_names'],
            ranges=data['param_ranges'],
            labels=[labels[n] for n in data['param_names']],
        )
legend_labels.append(labels_stats['pk'])
chains.append(samples)

data_dir = f'/pscratch/sd/e/epaillas/emc/posteriors/diffsky/dsc_fourier/base/'
data_fn = Path(data_dir) / f'chain_z0.5_fixedAmp_ph001_mass_conc_v0.3_k0.00-0.50.npy'
data = np.load(data_fn, allow_pickle=True).item()
samples = MCSamples(
            samples=data['samples'],
            weights=data['weights'],
            names=data['param_names'],
            ranges=data['param_ranges'],
            labels=[labels[n] for n in data['param_names']],
        )
legend_labels.append('Density-split')
chains.append(samples)


g = plots.get_subplot_plotter()
g.settings.constrained_layout = True
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = ":"
g.settings.title_limit_labels = False
g.settings.axis_marker_color = "k"
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth_contour = 1.0
g.settings.legend_fontsize = 20
g.settings.axes_fontsize = 16
g.settings.axes_labelsize = 20
g.settings.axis_tick_x_rotation = 45
g.settings.axis_tick_max_labels = 6

g.triangle_plot(
    roots=chains,
    legend_labels=legend_labels,
    filled=True,
    markers=truth,
    params=params,
    title_limit=1,
    # params=['logM_cut', 'logM_1']
    # legend_labels=stats
)
plt.savefig('figures/posterior_diffsky.pdf', bbox_inches='tight')
plt.show()
# plt.savefig(f'fig/posterior_test_emuerr_all_hod{idx_fit}.pdf', bbox_inches='tight')
# plt.savefig(f'fig/posterior_tpcf_hod{idx_fit}.pdf', bbox_inches='tight')