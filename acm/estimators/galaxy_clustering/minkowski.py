import numpy as np
import logging
import time
from .base import BaseEnvironmentEstimator
from .src import minkowski as Mk


class MinkowskiFunctionals(BaseEnvironmentEstimator):
    """
    Class to compute the Minkowski functionals.
    """
    def __init__(self, **kwargs):

        self.logger = logging.getLogger('MinkowskiFunctionals')
        self.logger.info('Initializing MinkowskiFunctionals.')
        super().__init__(**kwargs)

    def run(self,thres_mask, thres_low, thres_high, thres_bins):
        """
        Run the Minkowski functionals.

        Returns
        -------
        MFs3D : array_like
            3D Minkowski Functionals V_0, V_1, V_2, V_3.
        """
        t0 = time.time()
        self.thres_mask = thres_mask
        self.thres_low  = thres_low
        self.thres_high = thres_high
        self.thres_bins = thres_bins
        query_positions = self.get_query_positions(self.delta_mesh, method='lattice')
        self.delta_query = self.delta_mesh.read_cic(query_positions).reshape(
            (self.delta_mesh.nmesh[0], self.delta_mesh.nmesh[1], self.delta_mesh.nmesh[2]))
        MFs = Mk(self.delta_mesh.astype(np.float32),self.data_mesh.cellsize[0],self.thres_mask,self.thres_low,self.thres_high,self.thres_bins)
        self.MFs = MFs.MFs3D
        self.logger.info(f"Minkowski functionals elapsed in {time.time() - t0:.2f} seconds.")
        return self.MFs

    def plot_MFs(self,mf_cons=[1,10**3,10**5,10**7]):
        """
        Plot the Minkowski functionals
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(constrained_layout=False,figsize=[10,10])
        spec = fig.add_gridspec(ncols=2, nrows=2, hspace=0.2, wspace=0.3)

        x   = np.linspace(self.thres_low,self.thres_high,num=self.thres_bins+1)
        ylabels = [r"$V_{0}$",
                   r"$V_{1}[10^{- "+str(int(np.log10(mf_cons[1])))+"}hMpc^{-1}]$",
                   r"$V_{2}[10^{- "+str(int(np.log10(mf_cons[2])))+"}(hMpc^{-1})^2]$",
                   r"$V_{3}[10^{- "+str(int(np.log10(mf_cons[3])))+"}(hMpc^{-1})^3]$"]
        
        for i in range(4):
            ii = i//2
            jj = i%2
            ax = fig.add_subplot(spec[ii,jj])
            ax.plot(x,self.MFs[:,i]*mf_cons[i],color="blue",label=r"$R_G = "+str(self.R_G)+"h^{-1}Mpc$")
            if i==0:ax.legend()
            if self.thres_type=='rho':
                ax.set_xlabel(r"$\delta$")
            elif self.thres_type=='nu':
                ax.set_xlabel(r"$\nu$")
                
            ax.set_ylabel(ylabels[i])
            ax.axhline(color="black")
            ax.set_xlim(self.thres_low,self.thres_high)
        
        return fig
