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

# params = ['omega_cdm', 'sigma8_m', 'n_s', 'logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
params = ['Omega_m', 'h']
# params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/mar24/c000_hod030/LCDM/'
data_fn = Path(data_dir) / f"chain_number_density+ap_tpcf.npy"
chain = Chain.load(data_fn)
samples = Chain.to_getdist(chain, add_derived=True)
chains.append(samples)
legend_labels.append(r'$\textrm{Galaxy 2PCF}$')

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/mar24/c000_hod030/LCDM/'
data_fn = Path(data_dir) / f"chain_number_density+tpcf.npy"
chain = Chain.load(data_fn)
samples = Chain.to_getdist(chain, add_derived=True)
chains.append(samples)
legend_labels.append(r'$\textrm{Galaxy 2PCF}$''\n'r'$\textrm{(no AP effect)}$')

samples_planck = loadMCSamples('/global/cfs/cdirs/desi/users/plemos/planck/chains/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing', settings={'ignore_rows': 0.3})
samples_planck.addDerived(samples_planck.getParams().omegabh2, name='omega_b', label='\omega_{\rm cdm}')
samples_planck.addDerived(samples_planck.getParams().omegach2, name='omega_cdm', label='\omega_{\rm cdm}')
samples_planck.addDerived(samples_planck.getParams().omegam, name='Omega_m', label='\Omega_{\rm m}')
samples_planck.addDerived(samples_planck.getParams().sigma8, name='sigma8_m', label='\sigma_8')
samples_planck.addDerived(samples_planck.getParams().ns, name='n_s', label='n_s')
samples_planck.addDerived(samples_planck.getParams().H0/100, name='h', label='h')
chains.append(samples_planck)
legend_labels.append(r'Planck CMB')


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
g.settings.legend_fontsize = 16
g.settings.axes_fontsize = 16
g.settings.axes_labelsize = 20
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
    # filled=[True, False, False],
    # title_limit=1,
    # params=['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
    # legend_labels=stats
)


plt.savefig('contours_ap.png', dpi=300, bbox_inches='tight')
plt.savefig('contours_ap.pdf', bbox_inches='tight')