import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

labels_stats = {
    'wp': r'$\textrm{Galaxy } w_p$',
    'dsc_conf': 'Density-split',
    'dsc_conf_cross': 'Density-split (CCF)',
    'tpcf': 'Galaxy 2PCF',
    'number_density+tpcf': 'nbar + Galaxy 2PCF',
    'tpcf+dsc_conf': 'Galaxy 2PCF + DSC',
}


chains = []

cosmo_idx = 0
hod_idx = 30
smin = 0
kmin, kmax = 0.0, 1.0
# truth = dict(zip(lhc_x_names, lhc_x[hod_idx]))
legend_labels = []
# print(truth)

# params = ['omega_cdm', 'sigma8_m', 'n_s']
# params = ['omega_cdm', 'sigma8_m', 'n_s']
params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod30/base_lcdm'
data_fn = Path(data_dir) / f"pk+number_density_k0.00-0.50_chain.npy"
data = np.load(data_fn, allow_pickle=True).item()
samples = MCSamples(
            samples=data['samples'],
            weights=data['weights'],
            names=data['names'],
            ranges=data['ranges'],
            labels=[data['labels'][n] for n in data['names']],
        )
chains.append(samples)
maxl = data['samples'][data['log_likelihood'].argmax()]
legend_labels.append(r'$P_\ell(k)$')

# data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod30/base_lcdm'
# data_fn = Path(data_dir) / f"pdf+pk+number_density_k0.00-0.50_chain.npy"
# data = np.load(data_fn, allow_pickle=True).item()
# samples = MCSamples(
#             samples=data['samples'],
#             weights=data['weights'],
#             names=data['names'],
#             ranges=data['ranges'],
#             labels=[data['labels'][n] for n in data['names']],
#         )
# chains.append(samples)
# maxl = data['samples'][data['log_likelihood'].argmax()]
# legend_labels.append(r'$\textrm{CIC} + P_\ell(k)$')

# data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod30/base_lcdm'
# data_fn = Path(data_dir) / f"dsc_fourier+pdf+pk+number_density_k0.00-0.50_chain.npy"
# data = np.load(data_fn, allow_pickle=True).item()
# samples = MCSamples(
#             samples=data['samples'],
#             weights=data['weights'],
#             names=data['names'],
#             ranges=data['ranges'],
#             labels=[data['labels'][n] for n in data['names']],
#         )
# chains.append(samples)
# maxl = data['samples'][data['log_likelihood'].argmax()]
# legend_labels.append(r'$\textrm{CIC} + \textrm{DSC} + P_\ell(k)$')

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod30/base_lcdm'
data_fn = Path(data_dir) / f"wst+dsc_fourier+pdf+pk+number_density_k0.00-0.50_chain.npy"
data = np.load(data_fn, allow_pickle=True).item()
samples = MCSamples(
            samples=data['samples'],
            weights=data['weights'],
            names=data['names'],
            ranges=data['ranges'],
            labels=[data['labels'][n] for n in data['names']],
        )
chains.append(samples)
maxl = data['samples'][data['log_likelihood'].argmax()]
legend_labels.append(r'$\textrm{WST} + \textrm{CIC} + \textrm{DSC} + P_\ell(k)$')


markers = data['markers']
    
g = plots.get_subplot_plotter()
g.settings.constrained_layout = True
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = "--"
g.settings.title_limit_labels = False
g.settings.axis_marker_color = "k"
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = True
g.settings.linewidth_contour = 1.0
g.settings.legend_fontsize = 20
g.settings.axes_fontsize = 16
g.settings.axes_labelsize = 20
g.settings.axis_tick_x_rotation = 45
g.settings.axis_tick_max_labels = 6
# g.settings.solid_colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd'][::-1]
# g.settings.line_styles = g.settings.solid_colors

g.triangle_plot(
    roots=chains,
    legend_labels=legend_labels,
    # legend_labels=[r'$\textrm{Galaxy } w_p$', r'$+ \textrm{ Density-split }\xi_\ell$', r'$+ \textrm{ Galaxy }\xi_\ell$'],
    markers=markers,
    params=params,
    filled=True,
    # filled=[False, True, True, True, True],
    # title_limit=1,
    # params=['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
    # legend_labels=stats
)
plt.savefig(f'fig/posterior_AB_c{cosmo_idx:03}_hod{hod_idx:03}.pdf', bbox_inches='tight')
plt.savefig(f'fig/posterior_AB_c{cosmo_idx:03}_hod{hod_idx:03}.png', bbox_inches='tight', dpi=300)
