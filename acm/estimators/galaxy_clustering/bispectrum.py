### This has been imported from the PolyBin3D code developed by Oliver Philcox & Thomas Flöss
### (https://github.com/oliverphilcox/PolyBin3D).

import numpy as np
import multiprocessing as mp
import tqdm
from scipy.special import legendre
import time
import logging


class Bispectrum:
    """Bispectrum estimation class. This takes the binning strategy as input and a base class. 
    We can also feed in a function that applies the S^-1 operator.

    Note that we can additionally compute squeezed triangles by setting k_bins_squeeze, giving a different kmax for short and long sides.
    
    Inputs:
    - base: PolyBin class
    - k_bins: array of one-dimensional bin edges
    - lmax: (optional) maximum Legendre multipole, default: 4.
    - mask: (optional) HEALPix mask to deconvolve
    - applySinv: (optional) function which returns S^-1 ( ~ [Mask*2PCF + 1]^-1 ) applied to a given input data field. This is only needed for unwindowed estimators.
    - k_bins_squeeze: (optional) array of squeezed bin edges
    - include_partial_triangles: (optional) whether to include triangles whose centers don't satisfy the triangle conditions, default: False
    """
    def __init__(self, base, k_bins, lmax=4, mask=None, applySinv=None, k_bins_squeeze=None, include_partial_triangles=False):
        self.logger = logging.getLogger('Bispectrum')
        # Read in attributes
        self.base = base
        self.applySinv = applySinv
        self.k_bins = k_bins
        self.lmax = lmax
        self.Nk = len(k_bins)-1
        self.Nl = self.lmax//2+1
        self.include_partial_triangles = include_partial_triangles
        
        print("")
        assert np.max(self.k_bins)<np.min(base.kNy), "k_max must be less than k_Nyquist!"
        assert np.min(self.k_bins)>=np.max(base.kF), "k_min must be at least the k_fundamental!"
        assert np.max(self.k_bins)<=base.Pfid[0][-1], "Fiducial power spectrum should extend up to k_max!"
        assert np.min(self.k_bins)>=base.Pfid[0][0], "Fiducial power spectrum should extend down to k_min!"
        print("Binning: %d bins in [%.3f, %.3f] h/Mpc"%(self.Nk,np.min(self.k_bins),np.max(self.k_bins)))

        # Add squeezed binning
        if type(k_bins_squeeze)!=type(None):
            self.k_bins_squeeze = k_bins_squeeze
            self.Nk_squeeze = len(k_bins_squeeze)-1
            assert (self.k_bins_squeeze[:self.Nk+1]==self.k_bins).all(), "k_bins_squeeze must contain all non-squeezed bins!"
            assert np.max(self.k_bins_squeeze)<np.min(base.kNy), "k_max must be less than k_Nyquist!"
            assert np.min(self.k_bins_squeeze)>=np.max(base.kF), "k_min must be at least the k_fundamental!"
            assert np.max(self.k_bins_squeeze)<=base.Pfid[0][-1], "Fiducial power spectrum should extend up to k_max!"
            assert np.min(self.k_bins_squeeze)>=base.Pfid[0][0], "Fiducial power spectrum should extend down to k_min!"
            print("Squezed binning: %d bins in [%.3f, %.3f] h/Mpc"%(self.Nk_squeeze,np.min(self.k_bins_squeeze),np.max(self.k_bins_squeeze)))
        else:
            self.k_bins_squeeze = self.k_bins.copy()
            self.Nk_squeeze = self.Nk

        # Compute number of bins
        self.N3 = len(self.get_ks()[0])
        self.N_bins = self.N3*self.Nl
        print("l-max: %d"%(self.lmax))
        print("N_bins: %d"%self.N_bins)
        
        assert type(self.lmax)==int, "l-max must be an integer!"
        assert self.lmax<=4, "Only l-max<=4 is currently supported!"
        
        # Read-in mask
        if type(mask)==type(None):
            self.mask = np.ones(self.base.gridsize)
            self.const_mask = True
        else:
            self.mask = mask
            # Check if window is uniform
            if np.std(self.mask)<1e-12:
                self.const_mask = True
            else:
                self.const_mask = False
        if self.const_mask:
            print("Mask: constant")
        else:
            print("Mask: spatially varying")
        self.mask_mean = np.mean(self.mask)
        self.cube_mask_mean = np.mean(self.mask**3)
            
        # Create function for filtering to a specific k-bin
        self.bin_filt = lambda b: (self.base.modk_grid>=self.k_bins_squeeze[b])&(self.base.modk_grid<self.k_bins_squeeze[b+1])
        self.bin_filts = self.base.np.array([self.bin_filt(b) for b in range(self.Nk_squeeze)])
        self.bin_filt = lambda b: self.bin_filts[b]

        # Define legendre polynomials [for computing bispectrum multipoles]
        if self.base.sightline=='global' and self.lmax>0:
            self.legendres = {}
            self.legendres[0] = 1.
            for ell in range(2,self.lmax+1,2):
                self.legendres[ell] = legendre(ell)(self.base.muk_grid)
        
        # Define spherical harmonics in real-space [for computing bispectrum multipoles]
        if self.base.sightline=='local' and self.lmax>0:
            print("Generating spherical harmonics")
            self.Ylm_real = self.base._compute_real_harmonics(self.base.r_grids, self.lmax, odd_l=False)
            
        # Define spherical harmonics in Fourier-space [needed for normalizations]
        if self.lmax>0:
            self.Ylm_fourier = self.base._compute_real_harmonics(np.asarray(self.base.k_grids), self.lmax, odd_l=False)
        
    def test_bin(a,b,c,tol=1e-6):
        """Test bin to see if it satisfies triangle conditions, being careful of numerical overlaps.
        """
        k_lo = np.arange(k_min,k_max,dk)
        k_hi = k_lo+dk
        # Maximum k3 possible
        k_up = k_hi[a]+k_hi[b]
        if a>b: k_do = k_lo[a]-k_hi[b]
        elif a==b:
            k_do = 0.
        else:
            k_do = k_lo[b]-k_hi[a]
        if k_lo[c]+tol>=k_up or k_hi[c]-tol<=k_do:
            return 0
        else:
            return 1
    
    def _check_bin(self, bin1, bin2, bin3):
        """Return one if modes in the bin satisfy the triangle conditions, or zero else.

        This is used either for all triangles in the bin, or just the center of the bin.
        """
        if self.include_partial_triangles:
            # Maximum possible k3
            k3_hi = self.k_bins_squeeze[bin1+1]+self.k_bins_squeeze[bin2+1]
            # Minimum possible k3
            if bin1>bin2:
                k3_lo = self.k_bins_squeeze[bin1]-self.k_bins_squeeze[bin2+1]
            elif bin1==bin2:
                k3_lo = 0.
            else:
                k3_lo = self.k_bins_squeeze[bin2]-self.k_bins_squeeze[bin1+1]
            # Test bin, checking for numerical overlaps
            if self.k_bins_squeeze[bin3]>=k3_hi-1e-10 or self.k_bins_squeeze[bin3+1]<=k3_lo+1e-10:
                return 0
            else:
                return 1
        else:
            # Test center of bin
            k1 = 0.5*(self.k_bins[bin1]+self.k_bins[bin1+1])
            k2 = 0.5*(self.k_bins_squeeze[bin2]+self.k_bins_squeeze[bin2+1])
            k3 = 0.5*(self.k_bins_squeeze[bin3]+self.k_bins_squeeze[bin3+1])
            if k3<abs(k1-k2) or k3>k1+k2:
                return 0
            else:
                return 1
    
    def get_ks(self):
        """Return an array with the central k1, k2, k3 values for each bispectrum bin.
        """
        k1s, k2s, k3s = [],[],[]
        for bin1 in range(self.Nk):
            k1 = 0.5*(self.k_bins[bin1]+self.k_bins[bin1+1])
            for bin2 in range(bin1, self.Nk_squeeze):
                k2 = 0.5*(self.k_bins_squeeze[bin2]+self.k_bins_squeeze[bin2+1])
                for bin3 in range(bin2, self.Nk_squeeze):
                    k3 = 0.5*(self.k_bins_squeeze[bin3]+self.k_bins_squeeze[bin3+1])
            
                    # skip bins outside the triangle conditions
                    if not self._check_bin(bin1,bin2,bin3): continue
                    
                    # Add to output array
                    k1s.append(k1)
                    k2s.append(k2)
                    k3s.append(k3)
        
        return np.asarray([k1s,k2s,k3s])
  
    def _compute_symmetry_factor(self):
        """
        Compute symmetry factor giving the degeneracy of each bin. For l>0 this is computed using FFTs.
        """
        print("Computing degeneracy factor")
        self.sym_factor = []

        # First compute ell=0 elements
        for bin1 in range(self.Nk):
            for bin2 in range(bin1,self.Nk_squeeze):
                for bin3 in range(bin2,self.Nk_squeeze):
                    if not self._check_bin(bin1,bin2,bin3): continue

                    if bin1==bin2 and bin1==bin3 and bin1==bin3:
                        self.sym_factor.append(6.)

                    elif bin1==bin2 or bin1==bin3 or bin2==bin3:
                        self.sym_factor.append(2.)

                    else:
                        self.sym_factor.append(1.)

        # Now compute ell>0 with FFTs
        if self.lmax>0:

            # Define discrete binning functions
            bins = [self.base.to_real(self.bin_filt(b)*self.base.invPk0_grid) for b in range(self.Nk_squeeze)]

            for l in range(2,self.lmax+1,2):

                # Define Legendre-weighted binning functions
                bin23_l = []
                for b in range(self.Nk_squeeze):
                    bin23_l_temp = 0.
                    for lm_ind in range(len(self.Ylm_fourier[l])): 
                        bin23_l_temp += self.base.to_real(self.bin_filt(b)*self.base.invPk0_grid*self.Ylm_fourier[l][lm_ind])**2.
                    bin23_l.append(bin23_l_temp)

                # Iterate over bins
                for bin1 in range(self.Nk):
                    for bin2 in range(bin1,self.Nk_squeeze):
                        for bin3 in range(bin2,self.Nk_squeeze):
                            if not self._check_bin(bin1,bin2,bin3): continue

                            if bin1!=bin2 and bin2!=bin3:
                                self.sym_factor.append(1.)

                            elif bin1==bin2 and bin2!=bin3:
                                self.sym_factor.append(2.)

                            elif bin1!=bin2 and bin2==bin3:
                                self.sym_factor.append(1.+np.sum(bins[bin1]*bin23_l[bin3]).real/np.sum(bins[bin1]*bins[bin3]**2).real)

                            elif bin1==bin2 and bin1==bin3:
                                self.sym_factor.append(2.*(1.+2.*np.sum(bins[bin3]*bin23_l[bin3]).real/np.sum(bins[bin3]**3).real))

        self.sym_factor = np.asarray(self.sym_factor)
        assert len(self.sym_factor)==self.N_bins
        
    def _process_sim(self, sim, input_type='real'):
        """
        Process a single input simulation. This is used for the linear term of the bispectrum estimator.

        We return g_{b,l}(r) maps for this simulation.
        """

        if self.base.pixel_window!='none':
            if input_type=='real':
                sim_fourier = self.base.to_fourier(sim)
            else:
                sim_fourier = sim.copy()
            
            sim_fourier /= self.base.pixel_window_grid

            # Apply S^-1 to simulation and transform to Fourier space
            Sinv_sim_fourier = self.applySinv(sim_fourier, input_type='fourier', output_type='fourier')
        else:
            # Apply S^-1 to data and transform to Fourier space
            Sinv_sim_fourier = self.applySinv(sim, input_type=input_type, output_type='fourier')
            
        # Store the real-space map if necessary
        if (self.base.sightline=='local' and self.lmax>0):
            Sinv_sim_real = self.base.to_real(Sinv_sim_fourier)

        # Compute g_{b,l} maps
        g_bl_maps = []
        for l in range(0,self.lmax+1,2):

            # Monopole
            if l==0:
                g_bl_maps.append([self.base.to_real(self.bin_filt(b)*Sinv_sim_fourier).conj() for b in range(self.Nk_squeeze)])

            else:

                # Define Legendre L_ell(k.n) weighting
                if self.base.sightline=='global':
                    leg_map = self.legendres[l]*Sinv_sim_fourier
                else:
                    leg_map = 0.
                    for lm_ind in range(len(self.Ylm_fourier[l])):
                        leg_map += self.Ylm_fourier[l][lm_ind]*self.base.to_fourier(Sinv_sim_real*self.Ylm_real[l][lm_ind])

                # Compute spherical harmonics for this order
                g_bl_maps.append([self.base.to_real(self.bin_filt(b)*leg_map).conj() for b in range(self.Nk_squeeze)])

        return g_bl_maps

    def load_sims(self, load_sim, N_sims, verb=False, input_type='real', preload=True):
        """
        Load in and preprocess N_sim Monte Carlo simulations used in the linear term of the bispectrum estimator. 
        These can alternatively be generated with a fiducial spectrum using the generate_sims script.

        The input is a function which loads the simulations in real- or Fourier-space given an index (0 to N_sims-1).

        If preload=False, the simulation products will not be stored in memory, but instead accessed when necessary. This greatly reduces memory usage, but is less CPU efficient if many datasets are analyzed together.
        """
        
        self.N_it = N_sims
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        if preload:
            self.preload = True

            # Define lists of maps
            self.g_bl_maps = []
        
            # Iterate over simulations and preprocess appropriately
            for ii in range(self.N_it):
                if verb: print("Loading bias simulation %d of %d"%(ii+1,self.N_it))    
                this_sim = load_sim(ii)

                # Process simulation
                self.g_bl_maps.append(np.array(self._process_sim(this_sim, input_type=input_type)))

        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")
            
            # Simply save iterator and continue (simulations will be processed in serial later) 
            self.load_sim_data = lambda ii: self._process_sim(load_sim(ii), input_type=input_type)
           
    def generate_sims(self, N_sim, Pk_input=[], verb=False, preload=True):
        """
        Generate N_sim Monte Carlo simulations used in the linear term of the bispectrum generator. 
        These are pure GRFs simulations, generated with the input survey mask and pixel-window-function.
        
        If preload=True, we create N_it such simulations and store the relevant transformations into memory.
        If preload=False, we store only the function used to generate the sims, which will be processed later. This is cheaper on memory, but less CPU efficient if many datasets are analyzed together.
        
        We can alternatively load custom simulations using the load_sims script.
        """
        
        self.N_it = N_sim
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        def load_sim_data(ii, Pk_input):
            """Generate a single GRF simulation and produce g_bl maps"""

            # Generate simulation and Fourier transform
            if self.const_mask:
                sim_fourier = self.mask_mean*self.base.generate_data(int(1e5)+ii, Pk_input=Pk_input, output_type='fourier')
                return self._process_sim(sim_fourier, input_type='fourier')
            else:
                # Generate base simulation
                sim_real = self.mask*self.base.generate_data(int(1e5)+ii, Pk_input=Pk_input, output_type='real', include_pixel_window=False)

                # Apply pixel window and compute g_{bl}(r) maps
                if self.base.pixel_window!='none':
                    sim_fourier = self.base.to_fourier(sim_real)*self.base.pixel_window_grid
                    return self._process_sim(sim_fourier, input_type='fourier')
                else:
                    return self._process_sim(sim_real, input_type='real')

        if preload:
            self.preload = True

            # Define list of maps
            self.g_bl_maps = []

            # Iterate over simulations and preprocess appropriately
            for ii in range(self.N_it):
                if verb: print("Generating bias simulation %d of %d"%(ii+1,self.N_it))
                self.g_bl_maps.append(load_sim_data(ii, Pk_input))

        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")

            # Simply save iterator and continue (simulations will be processed in serial later) 
            self.load_sim_data = lambda ii: load_sim_data(ii, Pk_input)
    
    ### OPTIMAL ESTIMATOR
    def Bk_numerator(self, data, include_linear_term=True, verb=False):
        """
        Compute the numerator of the unwindowed bispectrum estimator, using the custom weighting function S^-1.

        We can optionally drop the linear term (at the cost of slightly enhanced large-scale variance).
        """

        if self.applySinv==None:
            raise Exception("Must supply S^-1 function to compute unwindowed estimators!")

        if (not hasattr(self, 'preload')) and include_linear_term:
            raise Exception("Need to generate or specify bias simulations!")

        # Compute symmetry factor if necessary
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()

        # First remove the pixel window function, if present
        if self.base.pixel_window!='none':
            data_fourier = self.base.to_fourier(data)
            data_fourier /= self.base.pixel_window_grid

            # Apply S^-1 to data and transform to Fourier space
            Sinv_data_fourier = self.applySinv(data_fourier, input_type='fourier', output_type='fourier')
        else:
            # Apply S^-1 to data and transform to Fourier space
            Sinv_data_fourier = self.applySinv(data, input_type='real', output_type='fourier')

        # Store the real-space map if necessary
        if (self.base.sightline=='local' and self.lmax>0):
            Sinv_data_real = self.base.to_real(Sinv_data_fourier)

        # Compute g_{b,l} maps
        g_bl_maps = []
        if verb: print("")
        for l in range(0, self.lmax+1, 2):

            if verb: print("Computing g_{b,%d}(r) maps"%l)

            # Define Legendre L_ell(k.n) weighting
            if l==0:
                leg_map = Sinv_data_fourier
            elif self.base.sightline=='global':
                leg_map = self.legendres[l]*Sinv_data_fourier
            else:
                leg_map = 0.
                for lm_ind in range(len(self.Ylm_fourier[l])):
                    leg_map += self.Ylm_fourier[l][lm_ind]*self.base.to_fourier(Sinv_data_real*self.Ylm_real[l][lm_ind])

            # Compute spherical harmonics for this order
            g_bl_maps.append([self.base.to_real(self.bin_filt(b)*leg_map).conj() for b in range(self.Nk_squeeze)])
            
        # Define output array
        B3_num = np.zeros(self.N_bins)

        # Compute numerator
        if verb: print("Computing cubic term")
        index = 0
        for l in range(0, self.lmax+1, 2):
            for bin1 in range(self.Nk):
                for bin2 in range(bin1, self.Nk_squeeze):
                    for bin3 in range(bin2, self.Nk_squeeze):
                        if not self._check_bin(bin1, bin2, bin3): continue

                        # Assemble numerator
                        B3_num[index] = np.sum(g_bl_maps[0][bin1]*g_bl_maps[0][bin2]*g_bl_maps[l//2][bin3]).real
                        index += 1

        # Compute b_1 part of cubic estimator, averaging over simulations
        B1_num = np.zeros(self.N_bins)

        if not include_linear_term:
            if verb: print("No linear correction applied!")
        else:
            if verb: print("Computing linear term")
            for ii in range(self.N_it):
                if (ii+1)%5==0 and verb: print("On simulation %d of %d"%(ii+1,self.N_it))

                # Load processed bias simulations 
                if self.preload:
                    this_g_bl_maps = self.base.np.array(self.g_bl_maps[ii])
                else:
                    this_g_bl_maps = self.load_sim_data(ii)

                index = 0
                for l in range(0, self.lmax+1, 2):
                    for bin1 in range(self.Nk):
                        for bin2 in range(bin1, self.Nk_squeeze):
                            for bin3 in range(bin2, self.Nk_squeeze):
                                if not self._check_bin(bin1, bin2, bin3): continue

                                # Assemble numerator, summing over permutations
                                out  = np.sum(g_bl_maps[0][bin1]*this_g_bl_maps[0][bin2]*this_g_bl_maps[l//2][bin3]).real
                                out += np.sum(this_g_bl_maps[0][bin1]*g_bl_maps[0][bin2]*this_g_bl_maps[l//2][bin3]).real
                                out += np.sum(this_g_bl_maps[0][bin1]*this_g_bl_maps[0][bin2]*g_bl_maps[l//2][bin3]).real
                                B1_num[index] += -out/self.N_it
                                index += 1

        # Assemble output and add normalization
        Bk_num = (B3_num+B1_num)*self.base.volume/self.base.gridsize.prod()/self.sym_factor

        return Bk_num
    
    def compute_fisher_contribution(self, seed, verb=False):
        """
        This computes the contribution to the Fisher matrix from a single pair of GRF simulations, created internally.
        """

        if self.applySinv==None:
            raise Exception("Must supply S^-1 function to compute unwindowed estimators!")

        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()

        # Initialize output
        fish = np.zeros((self.N_bins,self.N_bins))

        # Compute a random realization with known power spectrum
        if verb: print("Generating GRFs")

        # Define Q map code
        def compute_Q(weighting, Q_maps, phase):
            """
            Assemble and return the Q [ = partial_alpha zeta_ijk (weighting a)_j (weighting a)_k] maps in Fourier-space, given a weighting scheme.

            The outputs are either Q_alpha or S^-1.W.Q_alpha, added to the Q_maps arrays. We also add a phase +- 1.
            """

            k_maps = np.zeros((len(Q_maps), self.base.gridsize[0], self.base.gridsize[1], self.base.gridsize[2]), dtype=np.complex64)

            # Weight maps appropriately
            if weighting=='Sinv':
                # Compute S^-1 W a
                if self.const_mask:
                    weighted_map_fourier = self.applySinv(a_map_fourier, input_type='fourier', output_type='fourier')*self.mask_mean
                else:
                    weighted_map_fourier = self.applySinv(self.mask*a_map_real, input_type='real', output_type='fourier')
            else:
                # Compute A^-1 a
                weighted_map_fourier = self.base.applyAinv(a_map_fourier, input_type='fourier', output_type='fourier')

            # Define real-space map where necessary
            if self.base.sightline=='local' and self.lmax>0:
                weighted_map_real = self.base.to_real(weighted_map_fourier)

            # Compute indices for each triplet of fields
            indices = []
            for l in range(0, self.lmax+1, 2):
                for bin1 in range(self.Nk):
                    for bin2 in range(bin1,self.Nk_squeeze):
                        for bin3 in range(bin2,self.Nk_squeeze):
                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,bin3): continue
                            indices.append([l,bin1,bin2,bin3])
            indices = np.asarray(indices)

            # Compute g_{b,0} maps
            if verb: print("Computing g_{b,0}(r) maps")
            g_b0_maps = [self.base.to_real(self.bin_filt(b)*weighted_map_fourier).conj() for b in range(self.Nk_squeeze)]

            # Define Legendre L_ell(k.n) weighting for all ell
            leg_maps = [weighted_map_fourier]
            for ell in range(2, self.lmax+1, 2):
                if self.base.sightline=='global':
                    leg_maps.append(self.legendres[ell]*weighted_map_fourier)
                else:
                    leg_map = 0.
                    for lm_ind in range(len(self.Ylm_fourier[ell])):
                        leg_map += self.Ylm_fourier[ell][lm_ind]*self.base.to_fourier(weighted_map_real*self.Ylm_real[ell][lm_ind])
                    leg_maps.append(leg_map)
                    
            # Iterate over quadratic pairs of bins, starting with longer side
            for binB in range(self.Nk_squeeze):
                if verb: print("Computing matrix for k-bin %d of %d"%(binB+1,self.Nk_squeeze))

                # Compute all g_bB_l maps
                g_bBl_maps = [g_b0_maps[binB]]
                for ell in range(2,self.lmax+1,2):
                    g_bBl_maps.append(self.base.to_real(self.bin_filt(binB)*leg_maps[ell//2]).conj())

                # Iterate over shorter side
                for binA in range(binB+1):

                    # Find which elements of the Q3 matrix this pair is used for (with ordering)
                    these_ind1 = np.where((indices[:,1]==binA)&(indices[:,2]==binB))[0]
                    these_ind2 = np.where((indices[:,2]==binA)&(indices[:,3]==binB))[0]
                    these_ind3 = np.where((indices[:,3]==binB)&(indices[:,1]==binA))[0]
                    if len(these_ind1)+len(these_ind2)+len(these_ind3)==0: 
                        continue

                    # Compute FT[g_{0, bA},g_{ell, bB}] for all ell
                    ft_ABl = [self.base.to_fourier(g_b0_maps[binA]*g_bBl_maps[l//2]) for l in range(0, self.lmax+1, 2)]

                    def add_Q_element(binC_index, these_ind, largest_l=True, factor=2):
                        # Iterate over these elements and add to the output arrays
                        for ii in these_ind:
                            binC = indices[ii,binC_index]
                            l = indices[ii,0]

                            # Monopole
                            if l==0:
                                k_maps[ii] += factor*self.bin_filt(binC)*ft_ABl[0]
                            else:
                                # Apply legendre to external or internal leg
                                if largest_l:
                                    binC_ftAB = self.bin_filt(binC)*ft_ABl[0]
                                    if self.base.sightline=='global':
                                        # Work in Fourier-space for global line-of-sight
                                        k_maps[ii] += factor*binC_ftAB*self.legendres[l]
                                    else:
                                        # Work in real-space for Yamamoto line-of-sight
                                        real_map = 0.+0.j
                                        for lm_ind in range(len(self.Ylm_fourier[l])):
                                            real_map += self.Ylm_real[l][lm_ind]*self.base.to_real(binC_ftAB*self.Ylm_fourier[l][lm_ind])
                                        k_maps[ii] += factor*self.base.to_fourier(real_map)
                                else:
                                    k_maps[ii,:] += factor*self.bin_filt(binC)*ft_ABl[l//2]

                    # add elements, noting that the symmetries differ for each permutation
                    if weighting=='Sinv':
                        # Add all simultaneously, given symmetries
                        add_Q_element(2, these_ind3, False, 6)
                    elif weighting=='Ainv':
                        add_Q_element(3, these_ind1, True, 2) 
                        add_Q_element(1, these_ind2, False, 2)
                        add_Q_element(2, these_ind3, False, 2)

            # Optionally add S^-1. W weighting to maps
            if weighting=='Ainv':
                for i in range(len(Q_maps)):
                    if self.const_mask:
                        Q_maps[i] += phase*(self.applySinv(k_maps[i],input_type='fourier',output_type='fourier')*self.mask_mean).ravel()
                    else:
                        Q_maps[i] += phase*(self.applySinv(self.mask*self.base.to_real(k_maps[i]),input_type='real',output_type='fourier')).ravel()
            else:
                Q_maps += phase*k_maps.reshape(len(k_maps),-1)
            # End of function
            return

        # Set-up arrays
        Q_Sinv = np.zeros((self.N_bins,self.base.gridsize.prod()),dtype=np.complex64)
        Q_Ainv = np.zeros((self.N_bins,self.base.gridsize.prod()),dtype=np.complex64) 
    
        if verb: print("Allocating %.2f GB of memory"%(2*Q_Sinv.nbytes/1024./1024./1024.))    

        # Difference two random fields
        for ii in range(2):
            a_map_fourier = self.base.generate_data(seed=seed+int(1e7*(1+ii)), output_type='fourier')
            if not self.const_mask:
                a_map_real = self.base.to_real(a_map_fourier)

            # Compute Q maps
            if verb: print("\n# Computing Q[S^-1.W.a] maps for random field %d"%(ii+1))
            compute_Q('Sinv', Q_Sinv, (-1.)**ii)

            if verb: print("\n# Computing S^-1.W.Q[A^-1.a] maps for random field %d"%(ii+1))
            compute_Q('Ainv', Q_Ainv, (-1.)**ii)

        # Assemble Fisher matrix
        if verb: print("\n# Assembling Fisher matrix")

        # Compute Fisher matrix as an outer product
        fish = 1./24.*np.real(np.matmul(Q_Sinv.conj(),Q_Ainv.T))*self.base.volume/self.base.gridsize.prod()**2

        # Add normalization
        fish /= np.outer(self.sym_factor,self.sym_factor)

        return fish
    
    def compute_fisher(self, N_it, N_cpus=1, verb=False):
        """
        Compute the Fisher matrix and shot-noise term using N_it realizations. If N_cpus > 1, this parallelizes the operations.
        """
        
        if self.applySinv==None:
            raise Exception("Must supply S^-1 function to compute unwindowed estimators!")

        # Initialize output
        fish = np.zeros((self.N_bins,self.N_bins))
        
        global _iterable
        def _iterable(seed):
            return self.compute_fisher_contribution(seed, verb=verb)
        
        if N_cpus==1:
            for seed in range(N_it):
                if seed%5==0: print("## Computing Fisher contribution %d of %d"%(seed+1,N_it))
                fish += self.compute_fisher_contribution(seed, verb=verb)/N_it
        else:
            p = mp.Pool(N_cpus)
            print("## Computing Fisher contribution from %d Monte Carlo simulations on %d threads"%(N_it, N_cpus))
            all_fish = list(tqdm.tqdm(p.imap_unordered(_iterable,np.arange(N_it)),total=N_it))
            fish = np.sum(all_fish,axis=0)/N_it
            
        self.fish = fish
        self.inv_fish = np.linalg.inv(fish)
        
        return fish
    
    def Bk_unwindowed(self, data, fish=[], include_linear_term=True, verb=False):
        """
        Compute the unwindowed bispectrum estimator, using the custom weighting function S^-1.

        Note that the Fisher matrix must be computed before this is run, or it can be supplied separately.
        
        We can optionally drop the linear term (at the cost of slightly enhanced large-scale variance).
        """
        
        if self.applySinv==None:
            raise Exception("Must supply S^-1 function to compute unwindowed estimators!")

        # Compute inverse Fisher
        if len(fish)!=0:
            self.fish = fish
            self.inv_fish = np.linalg.inv(fish)
        
        if not hasattr(self,'inv_fish'):
            raise Exception("Need to compute Fisher matrix first!")
        
        # Compute numerator
        Bk_num = self.Bk_numerator(data, include_linear_term=include_linear_term, verb=verb)

        # Apply normalization and restructure
        Bk_out = np.matmul(self.inv_fish,Bk_num)
        
        # Create output dictionary of spectra
        Bk_dict = {}
        index = 0
        for l in range(0,self.lmax+1,2):
            Bk_dict['b%s'%l] = Bk_out[index:index+self.N3]
            index += self.N3

        return Bk_dict
    
    ### IDEAL ESTIMATOR
    def Bk_numerator_ideal(self, data, verb=False):
        """Compute the numerator of the idealized bispectrum estimator, weighting by 1/P_fid(k) within each bin. 

        The estimator does *not* use the mask or S_inv weighting schemes, and does not remove the linear term (which vanishes under ideal circumstances). It also applies only for even ell.

        This can compute the ideal bispectrum of simulation volumes, or, for suitably normalized input, the FKP bispectrum.
        """
        t0 = time.time()
        # Compute symmetry factor if necessary
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()

        # First remove the pixel window function
        data_fourier = self.base.to_fourier(data)
        if self.base.pixel_window!='none':
            data_fourier /= self.base.pixel_window_grid

        # Apply filtering by inverse P(k) and mask mean
        data_fourier *= self.base.invPk0_grid/(self.cube_mask_mean)**(1./3.)

        # Compute real-space map where necessary
        if (self.base.sightline=='local' and self.lmax>0):
            data_real = self.base.to_real(data_fourier)

        # Define output array
        Bk_out = np.zeros(self.N_bins)

        # Compute g_{b,0} maps
        if verb: print("Computing g_{b,0}(r) maps")
        g_b0_maps = [self.base.to_real(self.bin_filt(b)*data_fourier).conj() for b in range(self.Nk_squeeze)]

        # Compute numerator
        index = 0
        for l in range(0, self.lmax+1, 2):

            # Compute g_{b,l} maps
            if l!=0:
                if verb: print("Computing g_{b,%d}(r) maps"%l)

                # Define Legendre L_ell(k.n) weighting
                if self.base.sightline=='global':
                    leg_map = self.legendres[l]*data_fourier
                else:
                    leg_map = 0.
                    for lm_ind in range(len(self.Ylm_fourier[l])):
                        leg_map += self.Ylm_fourier[l][lm_ind]*self.base.to_fourier(data_real*self.Ylm_real[l][lm_ind])

                # Compute spherical harmonics for this order
                g_bl_maps = [self.base.to_real(self.bin_filt(b)*leg_map).conj() for b in range(self.Nk_squeeze)]

            if verb: print("Assembling l = %d numerator"%l)
            for bin1 in range(self.Nk):
                for bin2 in range(bin1, self.Nk_squeeze):
                    for bin3 in range(bin2, self.Nk_squeeze):
                        if not self._check_bin(bin1, bin2, bin3): continue

                        # Assemble numerator
                        if l==0:
                            Bk_out[index] = np.sum(g_b0_maps[bin1]*g_b0_maps[bin2]*g_b0_maps[bin3]).real
                        else:
                            Bk_out[index] = np.sum(g_b0_maps[bin1]*g_b0_maps[bin2]*g_bl_maps[bin3]).real
                        index += 1

        # Add normalization
        Bk_out *= self.base.volume/self.base.gridsize.prod()/self.sym_factor

        self.logger.info(f"Bispectrum calculated in {time.time() - t0:.2f} seconds.")
        return Bk_out

    def compute_fisher_ideal(self, discreteness_correction=True, verb=False):
        """This computes the idealized Fisher matrix for the power spectrum, weighting by 1/P_fid(k) within each bin. 

        We optionally include discreteness corrections for ell>0.

        If 'local' sightlines are being used, this assumes the distant-observer approximation.
        """
        # Compute symmetry factor if necessary
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()

        print("Computing ideal Fisher matrix")
        
        # Define output array
        fish = np.zeros((self.N_bins,self.N_bins))

        # Define discrete binning functions
        if verb: print("Computing u_{b,l}(r) maps")
            
        if discreteness_correction:
            # Include finite grid effects
            bins_l = [[self.base.to_real(self.bin_filt(b)*self.base.invPk0_grid*self.legendres[l]) for b in range(self.Nk_squeeze)] for l in np.arange(0,self.lmax+1,2)]

            # Iterate over Legendre multipoles
            for l in range(0, self.lmax+1, 2):
                for lp in range(l, self.lmax+1, 2):
                    if verb: print("Computing Fisher for (l, l') = (%d, %d)"%(l,lp))

                    # Compute double Legendre-weighted data
                    bins_llp = [self.base.to_real(self.bin_filt(b)*self.base.invPk0_grid*self.legendres[l]*self.legendres[lp]) for b in range(self.Nk_squeeze)]

                    i = 0
                    for bin1 in range(self.Nk):
                        for bin2 in range(bin1,self.Nk_squeeze):
                            for bin3 in range(bin2,self.Nk_squeeze):
                                if not self._check_bin(bin1,bin2,bin3): continue

                                # Iterate over possible symmetries
                                if bin1!=bin2 and bin2!=bin3:
                                    fish[l//2*self.N3+i,lp//2*self.N3+i] = np.sum(bins_l[0][bin1]*bins_l[0][bin2]*bins_llp[bin3]).real

                                elif bin1==bin2 and bin2!=bin3:
                                    fish[l//2*self.N3+i,lp//2*self.N3+i] = 2.*np.sum(bins_l[0][bin1]*bins_l[0][bin2]*bins_llp[bin3]).real

                                elif bin1!=bin2 and bin2==bin3:
                                    fish[l//2*self.N3+i,lp//2*self.N3+i] = np.sum(bins_l[0][bin1]*(bins_l[l//2][bin2]*bins_l[lp//2][bin3]+bins_l[0][bin2]*bins_llp[bin3])).real

                                elif bin1==bin2 and bin2==bin3:
                                    fish[l//2*self.N3+i,lp//2*self.N3+i] = 2.*np.sum(bins_l[0][bin1]*(2.*bins_l[l//2][bin2]*bins_l[lp//2][bin3]+bins_l[0][bin2]*bins_llp[bin3])).real
                                i += 1

            # Add transpose symmetry
            fish = fish+fish.T-np.diag(np.diag(fish))

            # Normalize
            fish = fish*self.base.gridsize.prod()**2/self.base.volume/np.outer(self.sym_factor, self.sym_factor)
            if verb: print("")

        else:
            bins = [self.base.to_real(self.bin_filt(b)*self.base.invPk0_grid) for b in range(self.Nk_squeeze)]

            # Assume Int L_ell(k.n) L_ell'(k.n) = 1/(2 ell + 1) Kronecker[ell, ell'] etc.
            fish_diag = np.zeros(self.N_bins)

            # Compute the number of triangles in the bin, N_0(b)
            N_triangles = np.zeros(self.N3)
            i = 0
            for bin1 in range(self.Nk):
                for bin2 in range(bin1,self.Nk_squeeze):
                    for bin3 in range(bin2,self.Nk_squeeze):
                        if not self._check_bin(bin1,bin2,bin3): continue
                        N_triangles[i] = np.sum(bins[bin1]*bins[bin2]*bins[bin3]).real
                        i += 1
            
            # Assemble Fisher matrix
            for l in range(0, self.lmax+1, 2):
                fish_diag[l//2*self.N3:(l//2+1)*self.N3] = N_triangles/(2.*l+1.)

            # Normalize
            fish = np.diag(fish_diag/self.sym_factor)*self.base.gridsize.prod()**2/self.base.volume

        self.fish_ideal = fish
        self.inv_fish_ideal = np.linalg.inv(self.fish_ideal)
        self.discreteness_correction = discreteness_correction

        return fish
    
    def Bk_ideal(self, data, fish_ideal=[], discreteness_correction=True, verb=False):
        """Compute the (normalized) idealized bispectrum estimator, weighting by 1/P_fid(k) within each bin. 
        
        The estimator does *not* use the mask or S_inv weighting schemes (except for normalizing by < mask^3 >, and does not remove the linear term (which vanishes under ideal circumstances)
        
        We optionally include 
        discreteness corrections for ell>0.
    
        This can compute the ideal bispectrum of simulation volumes, or, for suitably normalized input, the FKP power spectrum.
        """
        
        if len(fish_ideal)!=0:
            self.fish_ideal = fish_ideal
            self.inv_fish_ideal = np.linalg.inv(fish_ideal)

        if not hasattr(self,'inv_fish_ideal'):
            self.compute_fisher_ideal(discreteness_correction=discreteness_correction, verb=verb)
        else:
            if self.discreteness_correction!=discreteness_correction:
                self.compute_fisher_ideal(discreteness_correction=discreteness_correction, verb=verb)
        
        # Compute numerator
        Bk_num_ideal = self.Bk_numerator_ideal(data, verb=verb)

        # Apply normalization and restructure
        Bk_out = np.matmul(self.inv_fish_ideal,Bk_num_ideal)
        
        # Create output dictionary of spectra
        Bk_dict = {}
        index = 0
        for l in range(0,self.lmax+1,2):
            Bk_dict['b%s'%l] = Bk_out[index:index+self.N3]
            index += self.N3

        return Bk_dict