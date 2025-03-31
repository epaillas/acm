import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


chains = []
bestfits = []
phases = [1, 2]

params = ['omega_cdm', 'sigma8_m', 'n_s']
colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']

for phase in phases:
    data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/diffsky/mar11/galsampled_67120_fixedAmp_001_mass_conc_v0.3/LCDM/'
    chain_fn = Path(data_dir) / f"chain_number_density+tpcf.npy "
    chain = np.load(chain_fn, allow_pickle=True).item()
    samples = MCSamples(
                samples=chain['samples'],
                weights=chain['weights'],
                names=chain['names'],
                ranges=chain['ranges'],
                labels=[chain['labels'][n] for n in chain['names']],
            )
    chains.append(samples)
    markers = chain['markers']

    profiles_fn = Path(data_dir) / f"profiles_bk+tpcf.npy"
    profiles = np.load(profiles_fn, allow_pickle=True).item()
    bestfits.append(profiles['bestfit'])

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
    legend_labels=phases,
    # legend_labels=[r'$\textrm{Galaxy } w_p$', r'$+ \textrm{ Density-split }\xi_\ell$', r'$+ \textrm{ Galaxy }\xi_\ell$'],
    markers=markers,
    params=params,
    filled=False,
    # filled=[False, True, True, True, True],
    # title_limit=1,
    # params=['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
    # legend_labels=stats
)

for i, phase in enumerate(phases):
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

plt.savefig(f'contours_diffsky.pdf', bbox_inches='tight')

    