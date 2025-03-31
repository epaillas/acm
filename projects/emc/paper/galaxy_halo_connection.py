import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples, loadMCSamples
from sunbird.inference.samples import Chain
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

legend_labels = []

# params = ['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
# params = ['omega_cdm', 'sigma8_m', 'n_s', 'logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
# params = ['Omega_m', 'h']
params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/mar24/c000_hod030/LCDM/'
data_fn = Path(data_dir) / f"chain_number_density+corrected_wp.npy"
chain = Chain.load(data_fn)
samples = Chain.to_getdist(chain, add_derived=False)
chains.append(samples)
legend_labels.append(r'$w_p$')

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/mar24/c000_hod030/LCDM/'
data_fn = Path(data_dir) / f"chain_number_density+corrected_wp+tpcf.npy"
chain = Chain.load(data_fn)
samples = Chain.to_getdist(chain, add_derived=False)
chains.append(samples)
legend_labels.append(r'$w_p$+$\xi_\ell$')

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/mar24/c000_hod030/LCDM/'
data_fn = Path(data_dir) / f"chain_number_density+corrected_wp+tpcf+dsc_pk.npy"
chain = Chain.load(data_fn)
samples = Chain.to_getdist(chain, add_derived=False)
chains.append(samples)
legend_labels.append(r'$w_p$+$\xi_\ell$+$P^{\rm DS}_\ell$')

# data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/mar24/c000_hod030/LCDM/'
# data_fn = Path(data_dir) / f"chain_number_density+corrected_wp+tpcf+bk.npy"
# chain = Chain.load(data_fn)
# samples = Chain.to_getdist(chain, add_derived=False)
# chains.append(samples)
# legend_labels.append(r'$w_p$+$\xi_\ell$+$B_\ell$')


markers = chain.markers
    
g = plots.get_subplot_plotter()
g.settings.constrained_layout = True
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = "--"
g.settings.title_limit_labels = False
g.settings.axis_marker_color = "k"
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = True
g.settings.linewidth_contour = 1.0
g.settings.legend_fontsize = 30
g.settings.axes_fontsize = 20
g.settings.axes_labelsize = 24
g.settings.axis_tick_x_rotation = 45
g.settings.axis_tick_max_labels = 6
g.settings.solid_colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd'][::-1]
g.settings.line_styles = g.settings.solid_colors

g.triangle_plot(
    roots=chains,
    legend_labels=legend_labels,
    markers=markers,
    params=params,
    filled=True,
    legend_loc='upper right',
    # filled=[True, False, False],
    # title_limit=1,
    # params=['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
    # legend_labels=stats
)


plt.savefig('galaxy_halo_connection.png', dpi=300, bbox_inches='tight')
plt.savefig('galaxy_halo_connection.pdf', bbox_inches='tight')