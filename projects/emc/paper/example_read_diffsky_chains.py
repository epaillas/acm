import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


chains = []

params = ['omega_cdm', 'sigma8_m', 'n_s']
colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']
legend_labels = []

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/diffsky/mar11/galsampled_67120_fixedAmp_001_mass_conc_v0.3/LCDM/'
chain_fn = Path(data_dir) / f"chain_number_density+tpcf.npy"
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
legend_labels.append(r'$\overline{n}_{\rm gal}$+$\textrm{Galaxy 2PCF}$')

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

g.triangle_plot(
    roots=chains,
    legend_labels=legend_labels,
    markers=markers,
    params=params,
    filled=False,
)

plt.savefig(f'example_contours_diffsky.pdf', bbox_inches='tight')

    