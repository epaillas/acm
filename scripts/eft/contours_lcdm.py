from getdist import plots, MCSamples, loadMCSamples
from desilike.samples import Chain as DesilikeChain
from sunbird.inference.samples import Chain as SunbirdChain
from cosmoprimo.fiducial import AbacusSummit
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


legend_labels = []
mc_samples = []

data_fn = [f'chains/chain_folps_cosmo-base_correlation_{i}.npy' for i in range(4)]
chains = []
for fn in data_fn:
    chains.append(DesilikeChain.load(fn))
    chain = chains[0].concatenate([chain.remove_burnin(0.1) for chain in chains])
samples = DesilikeChain.to_getdist(chain, settings={'fine_bins_2D': 64, 'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.3})
mc_samples.append(samples)
legend_labels.append(r'EFT $\xi_{0,2}\,(s_{\rm min} = 30\,h^{-1}{\rm Mpc})$')

data_fn = [f'chains/chain_folps_cosmo-base_correlation_thetas_{i}.npy' for i in range(4)]
chains = []
for fn in data_fn:
    chains.append(DesilikeChain.load(fn))
    chain = chains[0].concatenate([chain.remove_burnin(0.1) for chain in chains])
samples = DesilikeChain.to_getdist(chain, settings={'fine_bins_2D': 128, 'smooth_scale_1D': 0.2, 'smooth_scale_2D': 0.2})
mc_samples.append(samples)
legend_labels.append(r'EFT $\xi_{0,2} + \theta_*$')

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/aug25/c000_hod030/cosmo-base_hod-base-VB-AB-s/'
data_fn = Path(data_dir) / f"chain_number_density+tpcf_smin30.npy"
chain = SunbirdChain.load(data_fn)
samples = SunbirdChain.to_getdist(chain, add_derived=False)
mc_samples.append(samples)
legend_labels.append(r'sunbird $\xi_{0,2}\,(s_{\rm min} = 30\,h^{-1}{\rm Mpc})$')

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/aug25/c000_hod030/cosmo-base_hod-base-VB-AB-s/'
data_fn = Path(data_dir) / f"chain_number_density+tpcf_smin5.npy"
chain = SunbirdChain.load(data_fn)
samples = SunbirdChain.to_getdist(chain, add_derived=False)
mc_samples.append(samples)
legend_labels.append(r'sunbird $\xi_{0,2} \,(s_{\rm min} = 5\,h^{-1}{\rm Mpc})$')


cosmo = AbacusSummit(0)
markers={'omega_cdm': cosmo['omega_cdm'], 'Omega0_m': cosmo.Omega0_m, 'h': cosmo.h,
         'logA': np.log(10**10 * cosmo.A_s), 'n_s': cosmo.n_s, 'omega_b': cosmo['omega_b'],
         'sigma8_m': cosmo.sigma8_m, 'w0_fld': -1.0, 'wa_fld': 0.0}
    
params = ['omega_cdm', 'sigma8_m', 'h']

g = plots.get_subplot_plotter(width_inch=6)
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = "--"
g.settings.title_limit_labels = False
g.settings.axis_marker_color = "k"
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.figure_legend_ncol = 1
g.settings.linewidth_contour = 1.5
g.settings.legend_fontsize = 20
g.settings.axes_fontsize = 17
g.settings.axes_labelsize = 20
g.settings.axis_tick_x_rotation = 45
g.settings.solid_colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']
g.settings.line_styles = ['darkslategrey'] * 10 

g.triangle_plot(
    roots=mc_samples,
    legend_labels=legend_labels,
    markers=markers,
    params=params,
    filled=[False, True, True, True],
    legend_loc='upper right',
)


plt.savefig('fig/contours_lcdm_v3.5.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig/contours_lcdm_v3.5.png', dpi=300, bbox_inches='tight')