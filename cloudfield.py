import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
from numpy.random import randint, normal
import img as image
from PIL import Image
from skimage.transform import warp_polar
from numba.experimental import jitclass
from numba import types, typed
from dpm_tools.metrics import minkowski_functionals, minkowski_map


class CloudField:

    def __init__(self, bool_img=None, r_max=0, gray_img=None, thresh=None):

        """
        Parameters :
        bool_img : bool 2D array. shape = square
            Represents the cloud field.
        r_max : float
            Largest value for the radii axis (in pixels). Might be reset
            to lower value in order to match the steps, used for pair
            correlation functions.
        gray_img : float 2D array from 0. to 255. shape = sqaure
            Cloud field in gray scale, used for Mantz 2008 algorithm.
        thresh : float between 0. and 255.
            Necessary for Mantz 2008 algorithm.
        """

        # CHECK COHERENCE BETWEEN bool_img, gray_levels AND threshold

        if gray_img is not None:
            if thresh is None:
                raise ValueError('Please specify thresh value.')
            if bool_img is None:
                bool_img = (gray_img >= thresh).astype(bool)
            else:
                check = (gray_img >= thresh).astype(bool) == bool_img
                if np.sum(~check) != 0:
                    raise ValueError('Incoherent bool_img, gray_img, thresh.')
        else:
            if bool_img is None:
                raise ValueError("bool_img and gray_img can't both be None")

        # CLOUD FELD ATTRIBUTES and NUMBA IMPLEMENTED CLASS
        self._numba_impl = _CloudFieldNumba(bool_img, r_max)
        self.n_cloudy = self._numba_impl.n_cloudy
        self.n_clear = self._numba_impl.n_clear
        self.cloud_cover = self._numba_impl.cloud_cover
        self.box_cloud_cover = self._numba_impl.box_cloud_cover

        # IMAGE ATTRIBUTES 
        self.bool_img = self._numba_impl.bool_img
        self.size = self._numba_impl.size
        self.n_tot = self._numba_impl.n_tot

        # RADII X-AXIS
        self.r_max = self._numba_impl.r_max
        self.box_boundaries = self._numba_impl.box_boundaries
        self.box_img = self._numba_impl.box_img
        self.x0, self.y0, self.side = self.box_boundaries

        # GRAY LEVELS
        self.gray_img = gray_img
        self.thresh = thresh

    def g(self, n_draws=10000, r_step=2):

        """
        Delegates heavy computations to numba_implemented class. Then,
        computes statistics.

        Parameters:
        n_draws : int
            Number of pixels (i,j) that we choose randomly. For each 
            (i,j), we also pick n_draws neighbors (k,l). 
            Global complexity = cloud_cover * n_draws**2
        r_step : float
            Thickness of the radii bins.

        Returns:
        mean : float array. shape = (len(self.r),)
            Pair correlation function of the image, averaged for every r
            over the (i,j) pixels drawn at random.
        std : float array. shape = (len(self.r),)
            Standard deviation of the pair correlation functions.
        sample : int array. shape = (len(self.r),)
            Size of the sample for each r. Should be close to n_draws.        
        """

        # Delegate heavy computations to numba_implemented class
        r, g_array = self._numba_impl.compute_g(n_draws, r_step)

        # Statistics over the (i,j) pixels sample
        mean = np.nanmean(g_array, axis=0)
        std = np.nanstd(g_array, axis=0)
        sample = np.sum(~np.isnan(g_array), axis=0)

        return r, mean, std, sample

    def epsilon(self, r, g_mean, g_std, g_sample, n_montecarlo=100000):
        
        """
        Calculates equation (31.9) from Peebles p.145. Epsilon is the 
        excess cloud cover (in px) within a circle of radius r centered 
        on a cloudy pixel.

        Parameters :
        r : float array.
        g_mean : float array. shape = (len(r,)
        g_std : float array. shape = (len(r,)
        g_sample : int array. shape = (len(r,)
            Output arrays from self.g
        n_montecarlo : int
            Sample size for Monte-Carlo simulation.

        Returns :
        eps_mean : float array.
        eps_std : float array.
        """

        # Generates an array of shape (len(r), N_montecarlo) of g 
        # functions chosen randomly according to a normal law. 
        g = np.array([
            normal(m, s / np.sqrt(n), n_montecarlo)
            for m, s, n in zip(g_mean, g_std, g_sample)
        ])        

        # Right-rectangle method
        # Initialize an array of zeros which will contain the epsilon 
        # functions computed for each of the n_montecarlo g functions.
        eps = np.zeros((len(r) + 1, n_montecarlo))
        r_step = r[1] - r[0]

        for i in range(1, len(r) + 1):
            # Compute the integral
            eps[i,:] = eps[i-1,:] + (r[i-1]
                                     * (g[i-1,:] - 1)
                                     *2*np.pi*r_step*self.cloud_cover)
        
        # Statistics on Monte-Carlo simulations
        eps = eps[1:,:] # remove column of zeros  
        eps_mean = np.mean(eps, axis=1)
        eps_std = np.std(eps, axis=1)

        return eps_mean, eps_std
    
    def C(self, n_draws=100000):

        """
        Computes the equal-time spin-spin correlation function (10.1) in
        the Oxford book.
        We draw n_draws pixels (i,j) within boundary_ij. For each pixel
        (i,j), we crop the surrounding cloud field in a square of size
        2*r_max, which represents the neighborhood of pixel (i,j).

        Parameters:
        n_draws : int 
            Number of pixels (i,j) that we choose randomly.

        Returns:
        C_xy : float array
            Spin-spin correlation map.
        C_r : float array
            Spin-spin correlation integrated over angles.
        C_angle : float array
            Spin-spin correlation integrated over radii.
        """

        # Delegates heavy computations to numba-implemented class.
        C_xy = self._numba_impl.compute_C(n_draws)

        # Change coordinates of the spin-spin correlation map.
        center = (self.r_max, self.r_max)
        C_polar = warp_polar(C_xy, radius=self.r_max, center=center)

        # Integration over angles and radii
        C_r = np.mean(C_polar, axis=0)
        C_angle = np.mean(C_polar, axis=1)

        return C_xy, C_r, C_angle
    
    def C_asym(self, n_draws=100000):

        """
        Exact same function as C, but cloudy cell is +1 and clear sky
        cell is 0 instead of +1 and -1.
        """

        C_xy = self._numba_impl.compute_C_asym(n_draws)
        center = (self.r_max, self.r_max)
        C_polar = warp_polar(C_xy, radius=self.r_max, center=center)
        C_r = np.mean(C_polar, axis=0)
        C_angle = np.mean(C_polar, axis=1)

        return C_xy, C_r, C_angle
    
    def minkowski(self, correction=True):

        """
        Computes the absolute minkowski functionals M0 M1 and M2 
        using the package DPM Tools.
        """

        M0, M1, M2 = minkowski_functionals(self.bool_img, pad=False)

        # Edge effects correction
        if correction:
            M0 = (self.size / (self.size - 1))**2 * M0
            M1 = self.size / (self.size - 1) * M1

        return M0, M1, M2
       
    def MF_map(self):

        """
        Testing of the minkowski_map function from DPM Tools.
        """

        M0, M1, M2 = minkowski_map(self.bool_img, [self.r_max, self.r_max])
        map = np.zeros((self.size, self.size, 3))
        map[:,:,0] = M0
        map[:,:,1] = M1
        map[:,:,2] = M2
        return map

    def mantz2008(self):

        """
        Implementation of the weighted marching squares algorithm 
        introduced by Mantz et al. (2008), to compute Minkowski
        Functionals.
        """

        if self.gray_img is None:
            raise ValueError('Need gray_img and thresh for Mantz 2008.')

        cf_mantz = _Mantz2008(bool_img = self.bool_img, 
                              gray_img = self.gray_img, 
                              thresh = self.thresh, 
                              size = self.size)
        
        return cf_mantz.minkowski()



spec1 = [
    ('bool_img', types.boolean[:,:]),
    ('size', types.int32),
    ('n_tot', types.int32),
    ('n_cloudy', types.int32),
    ('n_clear', types.int32),
    ('cloud_cover', types.float64),
    ('r_max', types.int32),
    ('r_step', types.float64),
    ('r', types.float64[:]),
    ('x0', types.int32),
    ('y0', types.int32),
    ('side', types.int32),
    ('box_img', types.boolean[:,:]),
    ('box_cloud_cover', types.float64),
    ('box_boundaries', types.UniTuple(types.int32, 3)),
]

@jitclass(spec1)
class _CloudFieldNumba:
    # This class contains numba-optimized methods.
    
    def __init__(self, bool_img, r_max):

        """
        Parameters:
        bool_img : boolean array. shape = square
            Binary array of the cloud field.
        r_max : float
            Largest value for the radii axis (in pixels). Might be reset
            to lower value in order to match the steps.
        """
        
        # VERIFICATION
        if bool_img.shape[0] != bool_img.shape[1]:
            raise ValueError('bool_img must be square : '
                             + f'bool_img.shape = {bool_img.shape}')
        
        # IMAGE ATTRIBUTES
        self.bool_img = bool_img[:,:].copy()
        self.size = bool_img.shape[0]
        self.n_tot = self.size * self.size
        
        # CLOUD FIELD ATTRIBUTES
        self.n_cloudy = np.count_nonzero(bool_img)
        self.n_clear = self.n_tot - self.n_cloudy
        self.cloud_cover = self.n_cloudy / self.n_tot
        
        # BOX IN WHICH WE DRAW CENTER PIXELS
        self.r_max = r_max
        x0, y0, side = r_max, r_max, self.size - 2*r_max
        self.box_boundaries = (x0, y0, side)
        self.box_img = bool_img[x0:x0+side, y0:y0+side].copy()
        self.box_cloud_cover = np.sum(self.box_img) / side**2
         
    def random_draw(self, n_draws, boundaries):

        """
        Draws n_draws pairs of indices at random, within the boundaries.

        Parameters:
        boundaries : None or tuple
            If None, boundaries will be set to the borders of the image.
            To specify boundaries, enter (x, y, size). Then the indices
            will be drawn in [x ; x + size[ and [y ; y + size[. 
        n_draws : int
            Number of pixels to draw.

        Returns :
        sample : int array. shape (n_draws, 2)
            Depending on the specified boundaries, the indices can be 
            negative. In this case the cloud field is assumed to have 
            periodic boundary conditions.
        """

        # Boundaries
        if boundaries is None:          
            boundaries = (0, 0, self.size)
        x, y, size = boundaries

        # Random seed of floats between 0 and 1
        sample = np.random.random(size=(n_draws, 2))

        # Convert the random seed into integer indices
        sample[:,0] = x + sample[:,0] * size
        sample[:,1] = y + sample[:,1] * size
        sample = sample.astype(np.int32)

        return sample
   
    def compute_g(self, n_draws, r_step):

        """
        This method calculates the pair correlation function of the 
        image. The principle is the following: we select a sample of 
        cloudy pixels (i,j) in the image, and calculate the probability 
        of finding a cloudy pixel (k,l) at a distance r from (i,j). To
        do so we construct a histogram of the number of cloudy pixels 
        over the bins r.
        
        Parameters:
        n_draws : int 
            Number of pixels (i,j) that we choose randomly. For each 
            (i,j), we also pick n_draws neighbors (k,l). 
            Global complexity = cloud_cover * n_draws**2
        r_step : float
            Thickness of the radii bins.
        
        Returns :
        r : float array
            Left edges of the histogram bins.
        g_array : float and nan array
            Contains the n_draws estimates for the correlation function
            at each center pixel (i,j).
        """

        # left edges of the histogram bins
        r = np.arange(0, self.r_max - r_step, r_step)

        # g_list is the numba precursor of g_array
        g_list = typed.List.empty_list(types.float64[:])
     
        # going through the center pixels (i,j)
        for (i, j) in self.random_draw(n_draws, self.box_boundaries):
            
            # skip to next (i,j) if this (i,j) is a clear-sky pixel
            is_cloudy_ij = self.bool_img[i, j]
            if not is_cloudy_ij: continue 

            # hist_all is the distribution of all (k,l) pixels along r
            hist_all = np.zeros(len(r))
            # hist_cloudy is the distribution of cloudy (k,l) pixels
            hist_cloudy = np.zeros(len(r))

            # going through all the neighbor pixels (k,l)
            for (k, l) in self.random_draw(n_draws, None):

                # calculates the distance between (i,j) and (k,l)
                dx = i - k
                dy = j - l
                dist = np.sqrt(dx*dx + dy*dy) 

                # fills the histograms
                if dist <= self.r_max:
                    bin_idx = int(dist / r_step)
                    hist_all[bin_idx] += 1
                    # the % makes the boundary conditions periodic
                    if self.bool_img[k%self.size, l%self.size]: 
                        hist_cloudy[bin_idx] += 1

            # avoid division by 0 by isolating the cases where no (k,l) 
            # pixel was drawn at distance r from (i,j). Else, compute 
            # pair correlation g = cloud_cover(r) / global_cloud_cover
            g_list.append(np.where(hist_all==0, 
                                   np.nan, 
                                   hist_cloudy / hist_all / self.cloud_cover))
                    
        # Convert the Numba pair correlation lists into a Numpy array
        g_array = np.zeros((len(g_list), len(r)))
        for i, g_ij in enumerate(g_list):
            g_array[i,:] = g_ij

        return r, g_array

    def compute_C(self, n_draws):

        """
        Computes the equal-time spin-spin correlation function (10.1) in
        the Oxford book.
        We draw n_draws pixels (i,j) within boundary_ij. For each pixel
        (i,j), we crop the surrounding cloud field in a square of size
        2*r_max, which represents the neighborhood of pixel (i,j).

        Parameters:
        n_draws : int 
            Number of pixels (i,j) that we choose randomly.

        Returns:
        C : float array
            Spin-spin correlation map.
        """

        corr_img = np.zeros((2*self.r_max, 2*self.r_max))
        n_img = 0

        boundaries_ij = (self.r_max, self.r_max, self.size-2*self.r_max)
        for (i,j) in self.random_draw(n_draws, boundaries_ij):
            neighborhood = self.bool_img[i-self.r_max:i+self.r_max+1,
                                    j-self.r_max:j+self.r_max+1]
            if self.bool_img[i, j]:
                corr_img += neighborhood - ~neighborhood
            else:
                corr_img += ~neighborhood - neighborhood
            n_img += 1

        C = corr_img / n_img
    
        return C
    
    def compute_C_asym(self, n_draws):

        """
        Exact same function as C, but cloudy cell is +1 and clear sky
        cell is 0 instead of +1 and -1.

        Parameters:
        n_draws : int 
            Number of pixels (i,j) that we choose randomly.

        Returns:
        C : float array
            Spin-spin asymetric correlation map.
        """

        corr_img = np.zeros((2*self.r_max, 2*self.r_max))
        n_img = 0

        boundaries_ij = (self.r_max, self.r_max, self.size-2*self.r_max)
        for (i,j) in self.random_draw(n_draws, boundaries_ij):
            neighborhood = self.bool_img[i-self.r_max:i+self.r_max+1,
                                    j-self.r_max:j+self.r_max+1]
            if self.bool_img[i, j]:
                corr_img += neighborhood
            n_img += 1

        if n_img ==0: n_img = 1
        C = corr_img / n_img
    
        return C


spec2 = [
    ('bool_img', types.boolean[:,:]),
    ('gray_img', types.float64[:,:]),
    ('thresh', types.uint8),
    ('size', types.int32),
    ('M2_Table', types.float64[:]),
    ('i_rows', types.int32),
    ('i_cols', types.int32)
]

@jitclass(spec2)
class _Mantz2008:

    """
    The role of the _Mantz2008 class is to implement the weighted
    marching square algorithm from Mantz et al. (2008) to cloud fields.
    The method minkowski computes the Minkowski Functionals according to
    this algorithm.

    Parameters:
    Given by the CloudField class.
    """

    def __init__(self, bool_img, gray_img, thresh, size):

        self.bool_img = bool_img
        self.gray_img = gray_img
        self.thresh = thresh
        self.size = size
        self.M2_Table = np.array([0.0,
                                  0.25,
                                  0.25,
                                  0.0,
                                  0.25,
                                  -0.5,
                                  0.0,
                                  -0.25,
                                  0.25,
                                  0.0,
                                  -0.5,
                                  -0.25,
                                  0.0,
                                  -0.25,
                                  -0.25,
                                  0.0])

    def _configuration(self, i_rows, i_cols):
        config = 0
        if self.bool_img[i_rows, i_cols]: config |= 8
        if self.bool_img[i_rows, i_cols + 1]: config |= 4
        if self.bool_img[i_rows + 1, i_cols + 1]: config |= 2
        if self.bool_img[i_rows + 1, i_cols]: config |= 1
        return config

    def _M0_M1_config(self, config, i_rows, i_cols):
  
        # Raccourcis pour alléger l'écriture des pixels
        p_tl = self.gray_img[i_rows, i_cols]           # Top-Left
        p_tr = self.gray_img[i_rows, i_cols + 1]       # Top-Right
        p_br = self.gray_img[i_rows + 1, i_cols + 1]   # Bottom-Right
        p_bl = self.gray_img[i_rows + 1, i_cols]       # Bottom-Left

        # Config 0
        if config == 0:     
            return 0.0, 0.0

        # Config 15
        elif config == 15:  
            return 1.0, 0.0

        # Config 1 : Coupe les arêtes 3 et 4 (intercepte p_bl)
        elif config == 1:
            X3 = (self.thresh - p_br) / (p_bl - p_br)
            X4 = (self.thresh - p_bl) / (p_tl - p_bl)
            M0 = 0.5 * X4 * (1.0 - X3)
            M1 = np.sqrt(X4**2 + (1.0 - X3) ** 2)
            return M0, M1 

        # Config 2 : Coupe les arêtes 2 et 3 (intercepte p_br)
        elif config == 2:
            X2 = (self.thresh - p_tr) / (p_br - p_tr)
            X3 = (self.thresh - p_br) / (p_bl - p_br)
            M0 = 0.5 * X3 * (1.0 - X2)
            M1 = np.sqrt(X3**2 + (1.0 - X2) ** 2)
            return M0, M1 

        # Config 3 : Coupe les arêtes 2 et 4
        elif config == 3:
            X2 = (self.thresh - p_tr) / (p_br - p_tr)
            X4 = (self.thresh - p_bl) / (p_tl - p_bl)
            M0 = 0.5 * (1.0 - X2 + X4)
            M1 = np.sqrt(1.0 + (1.0 - X2 - X4) ** 2)
            return M0, M1 

        # Config 4 : Coupe les arêtes 1 et 2 (intercepte p_tr)
        elif config == 4:
            X1 = (self.thresh - p_tl) / (p_tr - p_tl)
            X2 = (self.thresh - p_tr) / (p_br - p_tr)
            M0 = 0.5 * X2 * (1.0 - X1)
            M1 = np.sqrt(X2**2 + (1.0 - X1) ** 2)
            return M0, M1 

        # Config 5 : Cas double (coupe 1, 4 et 2, 3)
        elif config == 5:
            X1 = (self.thresh - p_tl) / (p_tr - p_tl)
            X2 = (self.thresh - p_tr) / (p_br - p_tr)
            X3 = (self.thresh - p_br) / (p_bl - p_br)
            X4 = (self.thresh - p_bl) / (p_tl - p_bl)
            M0 = 1.0 - 0.5 * X1 * (1.0 - X4) - 0.5 * X3 * (1.0 - X2)
            M1 = np.sqrt(X1**2 + (1.0-X4)**2) + np.sqrt(X3**2 + (1.0-X2)**2)
            return M0, M1

        # Config 6 : Coupe les arêtes 1 et 3
        elif config == 6:
            X1 = (self.thresh - p_tl) / (p_tr - p_tl)
            X3 = (self.thresh - p_br) / (p_bl - p_br)
            M0 = 0.5 * (1.0 - X1 + X3)
            M1 = np.sqrt(1.0 + (X3 + X1 - 1.0) ** 2)
            return M0, M1

        # Config 7 : Coupe les arêtes 1 et 4
        elif config == 7:
            X1 = (self.thresh - p_tl) / (p_tr - p_tl)
            X4 = (self.thresh - p_bl) / (p_tl - p_bl)
            M0 = 1.0 - 0.5 * X1 * (1.0 - X4)
            M1 = np.sqrt(X1**2 + (1.0 - X4) ** 2)
            return M0, M1 

        # Config 8 : Coupe les arêtes 1 et 4 (symétrique à 7)
        elif config == 8:
            X1 = (self.thresh - p_tl) / (p_tr - p_tl)
            X4 = (self.thresh - p_bl) / (p_tl - p_bl)
            M0 = 0.5 * X1 * (1.0 - X4)
            M1 = np.sqrt(X1**2 + (1.0 - X4) ** 2)
            return M0, M1 

        # Config 9 : Coupe les arêtes 1 et 3 (symétrique à 6)
        elif config == 9:
            X1 = (self.thresh - p_tl) / (p_tr - p_tl)
            X3 = (self.thresh - p_br) / (p_bl - p_br)
            M0 = 0.5 * (1.0 - X3 + X1)
            M1 = np.sqrt(1.0 + (1.0 - X3 - X1) ** 2)
            return M0, M1 

        # Config 10 : Cas double (coupe 1, 2 et 3, 4)
        elif config == 10:
            X1 = (self.thresh - p_tl) / (p_tr - p_tl)
            X2 = (self.thresh - p_tr) / (p_br - p_tr)
            X3 = (self.thresh - p_br) / (p_bl - p_br)
            X4 = (self.thresh - p_bl) / (p_tl - p_bl)
            M0 = 1.0 - 0.5 * X2 * (1.0 - X1) - 0.5 * X4 * (1.0 - X3)
            M1 = np.sqrt(X4**2 + (1.0-X3)**2) + np.sqrt(X2**2 + (1.0-X1)**2)
            return M0, M1

        # Config 11 : Coupe les arêtes 1 et 2
        elif config == 11:
            X1 = (self.thresh - p_tl) / (p_tr - p_tl)
            X2 = (self.thresh - p_tr) / (p_br - p_tr)
            M0 = 1.0 - 0.5 * X2 * (1.0-X1)
            M1 = np.sqrt(X2**2 + (1.0-X1)**2)
            return M0, M1 

        # Config 12 : Coupe les arêtes 2 et 4
        elif config == 12:
            X2 = (self.thresh - p_tr) / (p_br - p_tr)
            X4 = (self.thresh - p_bl) / (p_tl - p_bl)
            M0 = 0.5 * (1.0 - X4 + X2)
            M1 = np.sqrt(1.0 + (1.0 - X4 - X2) ** 2)
            return M0, M1 

        # Config 13 : Coupe les arêtes 2 et 3
        elif config == 13:
            X2 = (self.thresh - p_tr) / (p_br - p_tr)
            X3 = (self.thresh - p_br) / (p_bl - p_br)
            M0 = 1.0 - 0.5 * X3 * (1.0 - X2)
            M1 = np.sqrt(X3**2 + (1.0 - X2) ** 2)
            return M0, M1 

        # Config 14 : Coupe les arêtes 3 et 4
        else:
            X3 = (self.thresh - p_br) / (p_bl - p_br)
            X4 = (self.thresh - p_bl) / (p_tl - p_bl)
            M0 = 1.0 - 0.5 * X4 * (1.0 - X3)
            M1 = np.sqrt(X4**2 + (1.0 - X3) ** 2)
            return M0, M1

    def minkowski(self):
        M0, M1, M2 = 0., 0., 0.

        for i_rows in range(self.size - 1):
            for i_cols in range(self.size - 1):
                
                config = self._configuration(i_rows, i_cols)
                dM0, dM1 = self._M0_M1_config(config, i_rows, i_cols)

                M0 += dM0
                M1 += dM1
                M2 += self.M2_Table[config]       # Direct access
                
        return M0, M1, M2


    
if False:
    img_path = 'Nuages/trous_OI_60m.png'
    # bool_img = image.image_to_binary_array(img_path, thresh=150, size=5000)
    gray_img = image.image_to_float_array(img_path, size=5000)
    cf = CloudField(bool_img=bool_img, gray_img=gray_img, thresh=150)
    scale = 60e-3 #km/px
    area = cf.size**2 * scale**2

    # Minkowski functionals, in pixels 
    M0, M1, M2 = cf.minkowski()

    # Scaling and normalizing
    m0 = M0 * scale**2 / area
    m1 = M1 * scale / area
    m2 = M2 / area
    print((m0, m1, m2))

    M0, M1, M2 = cf.mantz2008()

    # Scaling and normalizing
    m0 = M0 * scale**2 / area
    m1 = M1 * scale / area
    m2 = M2 / area
    print((m0, m1, m2))



    