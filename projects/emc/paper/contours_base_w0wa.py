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

legend_labels = []

cosmo_idx = 0
hod_idx = 30
params = ['w0_fld', 'wa_fld']
# params = ['omega_cdm', 'sigma8_m', 'n_s']
# params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']

# data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod030/base_lcdm'
# data_fn = Path(data_dir) / f"pk+number_density_k0.00-0.50_chain.npy"
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
# legend_labels.append(r'$P_\ell(k)$')

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod030/base_w0wa'
data_fn = Path(data_dir) / f"pk+wp+number_density_k0.00-0.50_chain.npy"
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
legend_labels.append(r'$P_\ell(k) + w_p(r)$')

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/c000_hod030/base_w0wa'
data_fn = Path(data_dir) / f"wst+dsc_pk+minkowski+pk+wp+number_density_k0.00-0.50_chain.npy"
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
legend_labels.append(r'$+\textrm{DSC+MF}$')


markers = data['markers']

# g = plots.get_single_plotter(width_inch=4)
g = plots.get_single_plotter(width_inch=4.2, ratio=4.5/5)
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = True
g.settings.axes_fontsize = 15
g.settings.axes_labelsize = 17
g.plot_2d(chains, 'w0_fld', 'wa_fld', filled=True) 
g.add_legend(legend_labels, legend_loc='upper right', facecolor='w', handlelength=1.0)
g.add_x_marker(data['markers']['w0_fld'], lw=1.0)
g.add_y_marker(data['markers']['wa_fld'], lw=1.0)
plt.savefig(f'fig/posterior_base_w0wa_c{cosmo_idx:03}_hod{hod_idx:03}.pdf', bbox_inches='tight')
plt.savefig(f'fig/posterior_base_w0wa_c{cosmo_idx:03}_hod{hod_idx:03}.png', bbox_inches='tight', dpi=300)