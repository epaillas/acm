import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sunbird.inference import BaseSampler
from acm.observables import BaseObservable, BaseCombinedObservable
from acm.data.io_tools import summary_coords, correlation_from_covariance

import logging

class ControlPlots(BaseSampler):
    """
    A class to generale control plots for inference chains from sunbird.BaseSampler.
    """
    def __init__(self):
        self.logger = logging.getLogger('ControlPlots')
        
        # Chains
        self.chains = []
        self.samplers = [] # Not used yet
        self.index_chain_dict = {}
        
        # Observables
        self.observables = []
        self.index_observable_dict = {}
    
    #%% Model control plots
    def load_observable(self, observable: BaseObservable, label:str = None):
        """
        Load an observable into the control plots.
        
        Parameters
        ----------
        observable : BaseObservable
            The observable to be loaded.
        label : str, optional
            A label to identify the observable. If not provided, the observable's
            `stat_name` will be used as the key in the `index_observable_dict`.
            Defaults to None.
            
        Returns
        -------
        BaseObservable
            The loaded observable.
            
        Raises
        ------
        KeyError
            If an observable with the same `stat_name` is already loaded and no
            label is provided to avoid conflicts.
        """     
        self.observables.append(observable)
        if label is not None:
            self.index_observable_dict[label] = len(self.observables)-1 # Store the index of this observable
        else:
            if observable.stat_name in self.index_observable_dict:
                raise KeyError(
                    f'An observable with the same name {observable.stat_name} is already loaded. '
                    f'Please provide a label to avoid conflicts.'
                )
            self.index_observable_dict[observable.stat_name] = len(self.observables)
        return observable
    
    def plot_lhc_y(self, label: str, **kwargs):
        """
        Plots the LHC y-values for a given observable.
        
        Parameters
        ----------
        label : str
            The label of the observable to plot.
        **kwargs : dict, optional
            Additional keyword arguments to customize the plot. 
            If a 'factor' keyword argument is provided, the LHC y-values will be scaled by this factor (float or array of the size of the length of the data vector).
                
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object of the plot.
            
        Raises
        ------
        KeyError
            If no observable is loaded with the given label.
            
        Notes
        -----
        The function retrieves the LHC y-values for the specified observable, scales them by the given factor,
        and plots them. The plot is customized using the provided keyword arguments.
        """
        
        index = self.index_observable_dict.get(label, None)
        if index is None:
            raise KeyError(f'No observable loaded with label {label}')
        observable = self.observables[index]
        lhc_y = observable.lhc_y
        
        figsize = kwargs.pop('figsize', None)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        kwargs['color'] = kwargs.get('color', 'k')
        kwargs['alpha'] = kwargs.get('alpha', 0.1)
        
        f = kwargs.pop('factor', 1.0)
        lhc_y *= f
        ax.plot(lhc_y.T, **kwargs) # Transpose to plot the different observables
        
        ax.set_xlabel('bin_value index')
        ax.set_ylabel('Observable')
        ax.set_title(f'Loaded LHC y for observable {label}')
        fig.tight_layout()
        
        return fig, ax
    
    def plot_lhc_x(self, label: str, names: list[str] = None, return_histograms: bool = False, **kwargs):
        """
        Plots histograms of LHC x-values for a given observable label.
        
        Parameters:
        -----------
        label : str
            The label of the observable to plot.
        names : list of str, optional
            The names of the LHC x-values to plot. If None, all LHC x-values will be plotted.
            Defaults to None.
        return_histograms : bool, optional
            If True, the function will return the histograms along with the figure and axes.
            Defaults to False.
        **kwargs : dict
            Additional keyword arguments to pass to the `hist` function.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        ax : numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
            The axes objects containing the plots.
        histograms : dict, optional
            A dictionary containing the histograms for each LHC x-value name. Only returned if `return_histograms` is True.
            
        Raises:
        -------
        KeyError
            If no observable is loaded with the given label.
        """
        
        index = self.index_observable_dict.get(label, None)
        if index is None:
            raise KeyError(f'No observable loaded with label {label}')
        observable = self.observables[index]
        lhc_x = observable.lhc_x
        lhc_x_names = observable.lhc_x_names
        
        if names is None:
            names = lhc_x_names
        histograms = {}
        
        figsize = kwargs.pop('figsize', None)
        fig, ax = plt.subplots(len(names), 1, figsize=figsize) 
        kwargs['density'] = kwargs.get('density', True)
        kwargs['color'] = kwargs.get('color', 'k')
        kwargs['alpha'] = kwargs.get('alpha', 0.5)

        for i, name in enumerate(names):
            index = lhc_x_names.index(name)
            h = ax[i].hist(lhc_x[:, index], **kwargs)
            histograms[name] = h
            ax[i].set_xlabel(name)
        ax[0].legend()
        fig.tight_layout()
        
        if return_histograms:
            return fig, ax, histograms
        return fig, ax
    
    def get_separators(self, observable: BaseCombinedObservable, **kwargs):
        
        if not isinstance(observable, BaseCombinedObservable):
            return []
        
        # Separations
        separations = []
        for s in observable.observables:
            sc_dict = summary_coords(
                statistic=s.stat_name, 
                coord_type='emulator_error', # Get the statistic shape only (could be done by hand, but it's more elegant this way)
                bin_values=s.bin_values,
                summary_coords_dict=s.summary_coords_dict)
            dimensions = [len(val) for val in sc_dict.values()]
            data_length = np.prod(dimensions)
            separations.append(data_length)
        separations = np.asarray(separations).cumsum() # Cumulative sum of separations to add the offset of each observable
        separations = separations[:-1] # Remove the last element

        if len(separations) == 0:
            separations = [] # No need to plot separations if there is only one observable
        return separations
    
    def plot_model(self, label: str, truth: int|np.ndarray, params: list = None):
        
        index = self.index_observable_dict.get(label, None)
        if index is None:
            raise KeyError(f'No observable loaded with label {label}')
        observable = self.observables[index]
        covariance_matrix = observable.get_covariance_matrix()
        data_error = np.sqrt(np.diag(covariance_matrix))
        
        separators = self.get_separators(observable)
        
        # Get the prediction
        if params is not None:
            pred = observable.get_model_prediction(params)
        elif isinstance(truth, int):
            pred = observable.get_model_prediction(observable.lhc_x[truth]) 
        else:
            raise ValueError('Either provide a truth index or a set of parameters to predict the model')
        
        # Get the truth
        if isinstance(truth, int):
            truth = observable.lhc_y[truth]
        
        if len(pred) != len(truth):
            raise ValueError('The prediction and the truth must have the same length')
        
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 8), height_ratios=[3, 1])

        ax[0].plot(pred, label='Prediction', color='C1')
        ax[0].errorbar(list(range(len(truth))), truth, yerr=data_error, fmt='o', label='Truth', markersize=3, color='C0')
        # ax[0].fill_between(list(range(len(truth))), pred - emulator_error, pred + emulator_error, color='C1', alpha=0.5)

        ax[1].plot((truth - pred)/data_error)
        ax[1].axhline(0, color='k', linestyle='--')
        ax[1].fill_between(list(range(len(truth))), -1, 1, color='gray', alpha=0.5)

        for i in range(len(separators)):
            ax[0].axvline(separators[i], color='k', linestyle='--')
            ax[1].axvline(separators[i], color='k', linestyle='--')

        ax[0].legend()

        ax[1].set_xlabel('Bin')
        ax[0].set_ylabel('Value')
        ax[1].set_ylabel('Residual/Error')
        
        return fig, ax
    
    def plot_correlation_matrix(self, label:str, **kwargs):
        
        index = self.index_observable_dict.get(label, None)
        if index is None:
            raise KeyError(f'No observable loaded with label {label}')
        observable = self.observables[index]
        covariance_matrix = observable.get_covariance_matrix()
        correlation_matrix = correlation_from_covariance(covariance_matrix)
    
        fig, ax = plt.subplots(figsize=(5, 5))
        
        cmap = kwargs.pop('cmap', 'RdBu_r')
        kwargs['cmap'] = plt.get_cmap(cmap)
        kwargs['vmin'] = kwargs.get('vmin', -1)
        kwargs['vmax'] = kwargs.get('vmax', 1)
        kwargs['origin'] = kwargs.get('origin', 'lower')
        im = ax.imshow(correlation_matrix, **kwargs)

        for i in self.get_separators(observable):
            ax.axvline(i, color='black')
            ax.axhline(i, color='black')

        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.05)
        cb = fig.colorbar(im,cax=cax, orientation="horizontal", fraction=0.046, pad=0.05, ticks=[-1, -0.5, 0, 0.5, 1]) #colorbar on top
        cb.ax.xaxis.set_ticks_position('top')
        
        return fig, ax
        
    #%% Inference control plots
    def load_chain(self, chain_fn: str, label: str = None, ignore_checks: bool = False):
        """
        Loads the chains in the class and performs tests to check they are coherent between them

        Parameters
        ----------
        chain_fn : str
            Chain filename. Expects a numpy file containing a dictionnary 
            with the 'ranges', 'names', 'labels',  'markers' keys. 
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
        
        # Note : Ignore the thin argument, as it causes warnings in the getdist package
        colors = kwargs.get('colors', ['k'] + [f'C{i}' for i in range(len(self.chains))])
        label_dict = kwargs.get('label_dict', {}) 
        
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
    
    def plot_trace(self, label):
        """
        Plots the trace of each parameter in the chain
        """
        index = self.index_chain_dict.get(label, None)
        if index is None:
            raise KeyError(f'No chain loaded with label{label}')
        chain = self.chains[index]
        
        names = chain['names']
        labels = [chain['labels'][n] for n in names]
        fig, ax = plt.subplots(len(names), 1, figsize=(10, 2*len(names)))
        for i, name in enumerate(names):
            ax[i].plot(chain['samples'][:, i])
            ax[i].set_ylabel(labels[i])
        ax[i].set_xlabel('Iteration')
        fig.tight_layout()
        
        return fig, ax