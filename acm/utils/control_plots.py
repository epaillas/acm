import numpy as np 
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from matplotlib.colors import to_hex
from sunbird.inference import BaseSampler

import logging


class SamplePlots(BaseSampler):
    """
    A class to generate control plots for inference chains from sunbird.BaseSampler.
    """
    def __init__(self):
        self.logger = logging.getLogger('ControlPlots')
        
        # Chains
        self.chains = []
        self.samplers = [] # Not used yet
        self.index_chain_dict = {}
    
    #%% Inference control plots
    def load_chain(self, chain_fn: str, label: str = None, ignore_checks: bool = False):
        """
        Loads the chains in the class and performs tests to check they are coherent between them

        Parameters
        ----------
        chain_fn : str
            Chain filename. Expects a numpy file containing a dictionnary 
            with the 'ranges', 'names', 'labels', 'markers' keys. 
        label : str, optional
            The name of the chain loaded, to be displayed in the corner plot later
        ignore_checks : bool, optional
            If True, will ignore the checks for chain consistency (not recommended).
            Might cause issues later if the chains are not consistent, but can be useful to compare
            chains with different parameters.
            Default is False.

        Returns
        -------
        dict:
            The contents of the file
        """
        # Load chain
        chain = np.load(chain_fn, allow_pickle=True).item()
        chain['label'] = label
        self.chains.append(chain)
        if label is not None : 
            self.index_chain_dict[label] = len(self.chains)-1 # Store the index of this chain
        
        # For later, if load_chain is implemented in BaseSampler
        # sampler = BaseSampler().load_chain(chain_fn)
        # chain = sampler.get_chain()
        # self.chains.append(chain)
        # self.samplers.append(sampler)
        
        # Setup 
        setup_keys = ['ranges', 'names', 'labels',  'markers']
        for key in setup_keys:
            if getattr(self, key, None) is None:
                setattr(self, key, chain[key])
            else: # Check that the key is the same for all chains
                if ignore_checks: continue
                assert getattr(self, key) == chain[key], f'{key} does not match'
                
        if ignore_checks:
            self.logger.warning(
                'Ignoring checks for chain consistency might cause issues later. '
                'The following keys will be overriden by the values of the last chain loaded: '
                f'{setup_keys} '
                "Please check that the longer ones are loaded last or some parameters won't be found !"
            )
            
        return chain
    
    def plot_triangle(self, add_bestfit=False, **kwargs):
        """
        Plots the triangle plot of the chains loaded in the class, using the getdist package.

        Parameters
        ----------
        add_bestfit : bool, optional
            If True, will add the best fit point for each chain on the triangle plot.
            Defaults to False.
        **kwargs : dict, optional
            Additional keyword arguments to customize the plot. 
            See the getdist documentation for more options.

        Returns
        -------
        getdist.plots.GetDistPlotter
            The plotter object containing the triangle plot.
        """
        
        # Note : Ignore the thin argument, as it causes warnings in the getdist package
        colors = kwargs.get('colors', ['k'] + [f'C{i}' for i in range(len(self.chains))])
        label_dict = kwargs.get('label_dict', {}) 
        
        params = kwargs.pop('params', None)
        if params is not None:
            kwargs['params'] = [p for p in params if p in self.names] # Remove any parameter that is not in the chain (just in case)
        
        samples = []
        for chain in self.chains:
            names = chain['names']
            labels = [chain['labels'][param].strip('$') for param in names]
            label = label_dict.get(chain['label'], chain['label']) # replace label with actual name if provided

            sample = MCSamples(
                samples = chain['samples'],
                ranges = chain['ranges'],
                label = label, # Name of the chain (getdist)
                labels = labels, # Names of the parameters
                names = names,
                weights = chain.get('weights', None),
                loglikes = chain.get('log_likelihood', None),
            )
            samples.append(sample)
        
        g = plots.get_subplot_plotter()
        g.triangle_plot(
            samples, 
            line_args=[{'color': colors[i]} for i in range(len(samples))],
            contour_colors=colors, 
            **kwargs)
        
        if add_bestfit:
            for c, chain in enumerate(self.chains):
                names = chain['names'] # To get the right index later
                params = kwargs.get('params', names) # Will luckily already be in order because params will re-order the axes in the triangle plot 
                params = [p for p in params if p in names] # Remove any parameter that is not in the chain (just in case)
                maxl = chain['samples'][chain['log_likelihood'].argmax()]
                finished = []
                ax_idx = 0
                for i, param1 in enumerate(params):
                    for j, param2 in enumerate(params[::-1]):
                        if param2 in finished: continue
                        if param1 != param2:
                            g.fig.axes[ax_idx].plot(maxl[names.index(param1)], maxl[names.index(param2)],
                                                    marker='*', ms=10.0, color=colors[c], mew=1.0, mfc='w')
                        ax_idx += 1
                    finished.append(param1)
        
        return g
    
    def plot_trace(self, label: str, **kwargs):
        """
        Plots the trace of each parameter in the chain
        """
        index = self.index_chain_dict.get(label, None)
        if index is None:
            raise KeyError(f'No chain loaded with label{label}')
        chain = self.chains[index]
        
        names = chain['names']
        params = kwargs.get('params', names)
        params = [p for p in params if p in names]
        labels = [chain['labels'][p] for p in params]
        fig, ax = plt.subplots(len(params), 1, figsize=(10, 2*len(params)))
        for i, name in enumerate(params):
            ax[i].plot(chain['samples'][:, i])
            ax[i].set_ylabel(labels[i])
        ax[i].set_xlabel('Iteration')
        fig.tight_layout()
        
        return fig, ax
    
    # Useful to get the model prediction from a chain !
    def get_chain_params(self, label:str, param_names: str, mode:str = 'mean', fixed_params: dict = None):
        """
        Returns the parameters of the chain with the given label, in the right order for the model prediction.

        Parameters
        ----------
        label : str
            The label of the chain to get the parameters from.
        param_names : str
            The names, in order, of the parameters expected in the returned array.
            Needs to be provided in order to get the right parameters from the chain.
        mode : str, optional
            The mode to use to get the parameters values from the chain. Can be 'mean' or 'maxl'.
            Defaults to 'mean'.
        fixed_params : dict, optional
            A dictionary containing the fixed parameters to add to the returned parameters.
            Defaults to None.

        Returns
        -------
        list
            A list containing the parameters in the right order for the model prediction.
        """
        
        if mode not in ['mean', 'maxl']:
            raise ValueError('mode must be either mean or maxl')
        
        index = self.index_chain_dict.get(label, None)
        if index is None:
            raise KeyError(f'No chain loaded with label {label}')
        chain = self.chains[index]
        
        names = chain['names']
        maxl = chain['samples'][chain['log_likelihood'].argmax()]
        mean = chain['samples'].mean(axis=0)
        parameters = {}
        for n in names:
            parameters[n] = mean[names.index(n)] if mode == 'mean' else maxl[names.index(n)]
        
        if fixed_params is not None:
           parameters = {**parameters, **fixed_params}
           
        params = [parameters[p] for p in param_names]
        
        return params
    
    def plot_map(self, percentile: int|list = 95, **kwargs):
        """
        Plots the Maximum A Posteriori (MAP) point, the mean and the 95% confidence interval (default) for each parameter in the chain.

        Parameters
        ----------
        percentile : int | list, optional
            The percentile to use for the error bars. 
            If an integer is provided, the error bars will both be the same and correspond to the given percentile.
            If a list is provided, the first element will be the lower percentile and the second element the upper percentile.
            Defaults to 95.
        **kwargs : dict, optional
            Additional keyword arguments to customize the plots.
            Can contain the following keys:
            - colors: list of str, the colors to use for the chains. Defaults to ['k'] + [f'C{i}' for i in range(len(self.chains))].
            - label_dict: dict, a dictionary containing the labels to use for the chains. Defaults to {}.
            - params: list of str, the parameters to plot. Defaults to self.names.
            - markers: dict, a dictionary containing the value of vertical markers to plot for each parameter. Defaults to None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
        """
        percentile_list = [percentile, percentile] if isinstance(percentile, int) else percentile
        colors = kwargs.get('colors', ['k'] + [f'C{i}' for i in range(len(self.chains))])
        label_dict = kwargs.get('label_dict', {}) 
        params = kwargs.get('params', self.names)
        names = [p for p in params if p in self.names]
        labels = [self.labels[n] for n in names]
        chain_labels = [label_dict.get(chain['label'], chain['label']) for chain in self.chains] # replace label with actual name if provided
        
        fig, ax = plt.subplots(1, len(names), sharey=True, figsize=(3*len(names), 3))

        for i, chain in enumerate(self.chains):
            maxl = chain['samples'][chain['log_likelihood'].argmax()]
            mean = chain['samples'].mean(axis=0)
            percentiles = np.percentile(chain['samples'], percentile_list, axis=0)
            
            sc = to_hex(colors[i]) # Solid color
            fc = sc + '99' # Transparent color
            
            for j, n in enumerate(names):
                if n not in chain['names']: continue # Skip if the parameter is not in the chain
                
                idx = chain['names'].index(n) # Get the right index for the parameter to plot !
                err = np.abs([mean[idx] - percentiles[0][idx], percentiles[1][idx] - mean[idx]]).reshape(2, 1) # 2 values, 1 parameter, expected shape by errorbar
                ax[j].errorbar(mean[idx], i, xerr=err, fmt='', ecolor=sc, lw=2, capsize=5)
                ax[j].plot(maxl[idx], i, 'o', mec=sc, mfc='white', ms=8)
                ax[j].plot(mean[idx], i, 'o', mec=sc, mfc=fc)
        
        # Markers
        markers = kwargs.get('markers', None)
        if markers is not None:
            for i, n in enumerate(names):
                if n not in markers: continue # Skip if the parameter is not in the markers
                ax[i].axvline(markers[n], color='k', linestyle='--', lw=0.8, alpha=0.5)
        
        # Set labels for first plot
        ax[0].set_yticks(np.arange(len(chain_labels)))
        ax[0].set_yticklabels(chain_labels)
        ax[0].set_xlabel(labels[0])
        
        # Make y ticks invisible for all other plots
        for i, a in enumerate(ax[1:], start=1):
            plt.setp(a.get_yticklines(), visible=False)
            a.set_xlabel(labels[i])
        
        fig.tight_layout()
        
        return fig, ax