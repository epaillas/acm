from getdist import plots, MCSamples

def plot_triangle(chain, priors, labels, ranges, fixed_parameters, save_fn=None, add_bestfit=False, **kwargs):
        """Plot triangle plot
        """
        import matplotlib.pyplot as plt
        names = [param for param in priors.keys() if param not in fixed_parameters]
        labels = [labels[param].strip('$') for param in names]
        data = chain
        samples = MCSamples(samples=data['samples'], weights=data['weights'], names=names,
                            loglikes=data['log_likelihood'], labels=labels, ranges=ranges)
        g = plots.get_subplot_plotter()
        g.triangle_plot(samples, **kwargs)
        maxl = data['samples'][data['log_likelihood'].argmax()]
        if add_bestfit:
            params = kwargs['params'] if 'params' in kwargs else names
            ndim = len(params)
            finished = []
            ax_idx = 0
            for i, param1 in enumerate(params):
                for j, param2 in enumerate(params[::-1]):
                    if param2 in finished: continue
                    if param1 != param2:
                        g.fig.axes[ax_idx].plot(maxl[names.index(param1)], maxl[names.index(param2)],
                                                marker='*', ms=10.0, color='k', mew=1.0, mfc='w')
                    ax_idx += 1
                finished.append(param1)
        if save_fn:
            plt.savefig(save_fn, bbox_inches='tight')
        
        return g