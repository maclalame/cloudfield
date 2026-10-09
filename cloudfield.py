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

    def __init__(self, img, r_max=0):

        """
        Parameters :
        img : boolean or float [0,1] array. shape = square
            Represents the cloud field.
        r_max : float
            Largest value for the radii axis (in pixels). Might be reset
            to lower value in order to match the steps.
        """

        # CLOUD FELD ATTRIBUTES and NUMBA IMPLEMENTED CLASS
        self._numba_impl = _CloudFieldNumba(img, r_max)
        self.n_cloudy = self._numba_impl.n_cloudy
        self.n_clear = self._numba_impl.n_clear
        self.cloud_cover = self._numba_impl.cloud_cover
        self.box_cloud_cover = self._numba_impl.box_cloud_cover

        # IMAGE ATTRIBUTES 
        self.img = self._numba_impl.img
        self.size = self._numba_impl.size
        self.n_tot = self._numba_impl.n_tot

        # RADII X-AXIS
        self.r_max = self._numba_impl.r_max
        self.box_boundaries = self._numba_impl.box_boundaries
        self.box_img = self._numba_impl.box_img
        self.x0, self.y0, self.side = self.box_boundaries

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
    
    def minkowski(self):

        """
        Computes the absolute minkowski functionals M0 M1 and M2 
        using the package DPM Tools.
        """

        return minkowski_functionals(self.img, pad=False)
    
    
    def MF_map(self):

        """
        Testing of the minkowski_map function from DPM Tools.
        """

        M0, M1, M2 = minkowski_map(self.img, [self.r_max, self.r_max])
        map = np.zeros((self.size, self.size, 3))
        map[:,:,0] = M0
        map[:,:,1] = M1
        map[:,:,2] = M2
        return map



spec = [
    ('img', types.boolean[:,:]),
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

@jitclass(spec)
class _CloudFieldNumba:
    # This class contains numba-optimized methods.

    def __init__(self, img, r_max):

        """
        Parameters:
        img : boolean array. shape = square
            Binary array of the cloud field.
        r_max : float
            Largest value for the radii axis (in pixels). Might be reset
            to lower value in order to match the steps.
        """

        # VERIFICATION
        if img.shape[0] != img.shape[1]:
            raise ValueError(f'img must be square : img.shape = {img.shape}')

        # IMAGE ATTRIBUTES
        self.img = img[:,:].copy()
        self.size = img.shape[0]
        self.n_tot = self.size * self.size
        
        # CLOUD FIELD ATTRIBUTES
        self.n_cloudy = np.sum(img)
        self.n_clear = self.n_tot - self.n_cloudy
        self.cloud_cover = self.n_cloudy / self.n_tot
        
        # BOX IN WHICH WE DRAW CENTER PIXELS
        self.r_max = r_max
        x0, y0, side = r_max, r_max, self.size - 2*r_max
        self.box_boundaries = (x0, y0, side)
        self.box_img = img[x0:x0+side, y0:y0+side]
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
            is_cloudy_ij = self.img[i, j]
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
                    if self.img[k%self.size, l%self.size]: 
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
            neighborhood = self.img[i-self.r_max:i+self.r_max+1,
                                    j-self.r_max:j+self.r_max+1]
            if self.img[i, j]:
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
            neighborhood = self.img[i-self.r_max:i+self.r_max+1,
                                    j-self.r_max:j+self.r_max+1]
            if self.img[i, j]:
                corr_img += neighborhood
            n_img += 1

        if n_img ==0: n_img = 1
        C = corr_img / n_img
    
        return C
    
if False:
    img_path = 'Nuages/trous_OI_60m.png'
    img_array = image.image_to_binary_array(img_path, threshold=150)[:5000,:5000]
    cf = CloudField(img_array, r_max=100)

    scale = 60e-3 #km/px
    area = cf.size**2 * scale**2

    # Minkowski functionals, in pixels 
    M0, M1, M2 = cf.minkowski()

    # Scaling and normalizing
    m0 = M0 * scale**2 / area
    m1 = M1 * scale / area
    m2 = M2 / area



    