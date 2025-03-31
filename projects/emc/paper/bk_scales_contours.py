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

hod_idx = 430
smin = 0
kmin, kmax = 0.0, 1.0
# truth = dict(zip(lhc_x_names, lhc_x[hod_idx]))
legend_labels = [
    r'$(k_1, k_2, k_3) < (0.063, 0.083, 0.123)\, h/{\rm Mpc}$',
    r'$(k_1, k_2, k_3) < (0.083, 0.223, 0.223)\, h/{\rm Mpc}$',
    r'$(k_1, k_2, k_3) < (0.123, 0.203, 0.243)\, h/{\rm Mpc}$',
    r'$(k_1, k_2, k_3) < (0.243, 0.243, 0.243)\, h/{\rm Mpc}$',
    
]
# print(truth)

# params = ['omega_cdm', 'sigma8_m', 'n_s', 'logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
params = ['omega_cdm', 'sigma8_m', 'n_s']
# params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']

for bidx in ['0-60', '0-120', '0-180', '0-240']:

    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod030/test_feb1/'
    data_fn = Path(data_dir) / f"bk_bidx{bidx}_chain.npy"
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
g.settings.legend_fontsize = 16
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


plt.savefig('bk_scales_contours.png', dpi=300, bbox_inches='tight')