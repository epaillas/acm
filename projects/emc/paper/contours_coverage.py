import matplotlib.pyplot as plt
from pathlib import Path
from getdist import plots, MCSamples
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')



for cosmo_idx in [0, 1, 2]:
    samples_global = []

    params = ['w0_fld', 'wa_fld']
    # params = ['omega_cdm', 'sigma8_m', 'n_s']
    # params = ['A_cen', 'A_sat', 'B_cen', 'B_sat']
    colors = ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']

    chain_dir = '/global/cfs/cdirs/desicollab/users/epaillas/acm/fits_emc/abacus/feb14/'
    for chain_fn in list(Path(chain_dir).glob(f'c{cosmo_idx:03}_hod*/w0waCDM_nrun_Nur/chain_pk+bk.npy')):
        chain = np.load(chain_fn, allow_pickle=True).item()
        samples = MCSamples(
                    samples=chain['samples'],
                    weights=chain['weights'],
                    names=chain['names'],
                    ranges=chain['ranges'],
                    labels=[chain['labels'][n] for n in chain['names']],
                )
        # chains.append(samples)
        markers = chain['markers']
        samples_global.append(samples.mean(list(chain['names'])))

    samples_global = np.asarray(samples_global)
    chain_global = MCSamples(
                samples=samples_global,
                names=chain['names'],
                ranges=chain['ranges'],
                labels=[chain['labels'][n] for n in chain['names']],
            )

    print(markers)
    print(np.shape(samples_global))

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
        roots=chain_global,
        legend_labels=[f'{cosmo_idx:03}'],
        # legend_labels=[r'$\textrm{Galaxy } w_p$', r'$+ \textrm{ Density-split }\xi_\ell$', r'$+ \textrm{ Galaxy }\xi_\ell$'],
        markers=markers,
        params=params,
        filled=False,
        # filled=[False, True, True, True, True],
        title_limit=1,
        # params=['logM_cut', 'logM_1', 'sigma', 'kappa', 'alpha']
        # legend_labels=stats
    )

    idx = [np.where(np.array(chain['names']) == i)[0] for i in params]
    g.fig.axes[0].scatter(samples_global[:, idx[0]], samples_global[:, idx[1]], s=5.0, zorder=1)


    plt.savefig(f'contours_coverage_c{cosmo_idx:03}_w0waCDM_nrun_Nur.png', bbox_inches='tight', dpi=300)

        