import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from sunbird.inference import BaseSampler

class ControlPlots(BaseSampler):
    """
    A class to generale control plots for inference chains from sunbird.BaseSampler.
    """
    def __init__(self):
        self.chains = []
        self.samplers = []
        self.index_chain_dict = {}
    
    def load_chain(self, chain_fn: str, label: str = None):
        """
        Loads the chains in the class and performs tests to check they are coherent between them

        Parameters
        ----------
        chain_fn : str
            Chain filename. Expects a numpy file containing a dictionnary 
            with the 'ranges', 'names', 'labels',  'markers' keys. 
        label : str, optional
            The name of the chain loaded, to be displayed in the corner plot later

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
            self.index_chain_dict[label] = len(self.chains) # Store the index of this chain
        
        # For later
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
                assert getattr(self, key) == chain[key], f'{key} does not match'
            
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
                params = kwargs.get('params', chain['names']) # Will luckily already be in order because params will re-order the axes in the triangle plot 
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
        labels = chain['labels']
        fig, ax = plt.subplots(len(names), 1, figsize=(10, 2*len(names)))
        for i, name in enumerate(names):
            ax[i].plot(chain['samples'][:, i])
            ax[i].set_ylabel(labels[i])
        ax[i].set_xlabel('Iteration')
        fig.tight_layout()
        
        return fig, ax