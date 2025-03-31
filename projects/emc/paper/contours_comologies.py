import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


chains = []
bestfits = []

params = ['omega_cdm', 'sigma8_m', 'n_s']
# params = ['omega_cdm', 'sigma8_m', 'n_s']
# params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']
colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']

cosmos = [0, 4]
hods = [30, 20]

for cosmo_idx, hod_idx in zip(cosmos, hods):

    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/feb14/c{cosmo_idx:03}_hod{hod_idx:03}/lcdm'
    chain_fn = Path(data_dir) / f"chain_tpcf.npy"
    chain = np.load(chain_fn, allow_pickle=True).item()
    samples = MCSamples(
                samples=chain['samples'] - np.array([chain['markers'][n] for n in chain['names']]),
                weights=chain['weights'],
                names=chain['names'],
                ranges={n: chain['ranges'][n] - chain['markers'][n] for n in chain['names']},
                labels=[chain['labels'][n] for n in chain['names']],
            )
    chains.append(samples)
    markers = {n: 0 for n in chain['names']}

    profiles_fn = Path(data_dir) / f"profiles_tpcf.npy"
    profiles = np.load(profiles_fn, allow_pickle=True).item()
    bestfits.append({n: profiles['bestfit'][n] - chain['markers'][n] for n in chain['names']})

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
g.settings.solid_colors = colors
# g.settings.line_styles = g.settings.solid_colors

g.triangle_plot(
    roots=chains,
    legend_labels=cosmos,
    # legend_labels=[r'$\textrm{Galaxy } w_p$', r'$+ \textrm{ Density-split }\xi_\ell$', r'$+ \textrm{ Galaxy }\xi_\ell$'],
    markers=markers,
    params=params,
    filled=True,
    # filled=[False, True, True, True, True],
    # title_limit=1,
    # params=['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
    # legend_labels=stats
)

for i, cosmo in enumerate(cosmos):
    ndim = len(params)
    finished = []
    ax_idx = 0
    for param1 in params:
        for param2 in params[::-1]:
            if param2 in finished: continue
            if param1 != param2:
                g.fig.axes[ax_idx].plot(bestfits[i][param1], bestfits[i][param2],
                                        marker='*', ms=10.0, color=colors[i], mew=1.0, mfc='w')
            ax_idx += 1
        finished.append(param1)

plt.savefig(f'contours_cosmologies.pdf', bbox_inches='tight')

    