import numpy as np
import math
import aux_functions
from numpy.fft import fft2, ifft2
from skimage import restoration

# ctf class

def voltage_to_wavelength(voltage):
    """
    Convert from electron voltage to wavelength.
    :param voltage: float, The electron voltage in kV.
    :return: float, The electron wavelength in nm.
    """
    return 12.2643247 / math.sqrt(voltage * 1e3 + 0.978466 * voltage ** 2)

class Filter:
    def __init__(self, dim=None, radial=False):
        self.dim = dim
        self.radial = radial

    def evaluate(self, omega):
        """
        Evaluate the filter at specified frequencies.
        :param omega: A vector of size n (for 1d filters), or an array of size 2-by-n, representing the spatial
            frequencies at which the filter is to be evaluated. These are normalized so that pi is equal to the Nyquist
            frequency.
        :return: The value of the filter at the specified frequencies.
        """
        if self.radial:
            if omega.ndim > 1:
                omega = np.sqrt(np.sum(omega ** 2, axis=0))
                omega, idx = np.unique(omega, return_inverse=True)
                omega = np.vstack((omega, np.zeros_like(omega)))

        h = self._evaluate(omega)

        if self.radial:
            h = np.take(h, idx)

        return h

    def _evaluate(self, omega):
        raise NotImplementedError("Subclasses should implement this method")

    def evaluate_grid(self, L, dtype=np.float32):
        grid2d = aux_functions.grid_2d(L, dtype=dtype)
        omega = np.pi * np.vstack((grid2d["x"].flatten("F"), grid2d["y"].flatten("F"))) # need to implement/import flatten? ndarray...
        h = self.evaluate(omega)

        h = aux_functions.m_reshape(h, grid2d["x"].shape)

        return h


class CTFFilter(Filter):
    def __init__(
            self,
            pixel_size=10,
            voltage=200,
            defocus_u=15000,
            defocus_v=15000,
            defocus_ang=0,
            Cs=2.26,
            alpha=0.07,
            B=0,
    ):
        """
        A CTF (Contrast Transfer Function) Filter
        :param pixel_size:  Pixel size in angstrom
        :param voltage:     Electron voltage in kV
        :param defocus_u:   Defocus depth along the u-axis in angstrom
        :param defocus_v:   Defocus depth along the v-axis in angstrom
        :param defocus_ang: Angle between the x-axis and the u-axis in radians
        :param Cs:          Spherical aberration constant
        :param alpha:       Amplitude contrast phase in radians
        :param B:           Envelope decay in inverse square angstrom (default 0)
        """
        super().__init__(dim=2, radial=defocus_u == defocus_v)
        self.pixel_size = pixel_size
        self.voltage = voltage
        self.wavelength = voltage_to_wavelength(self.voltage)
        self.defocus_u = defocus_u
        self.defocus_v = defocus_v
        self.defocus_ang = defocus_ang
        self.Cs = Cs
        self.alpha = alpha
        self.B = B

        self.defocus_mean = 0.5 * (self.defocus_u + self.defocus_v)
        self.defocus_diff = 0.5 * (self.defocus_u - self.defocus_v)

    def _evaluate(self, omega):
        om_x, om_y = np.vsplit(omega / (2 * np.pi * self.pixel_size), 2)

        eps = np.finfo(np.pi).eps
        ind_nz = (np.abs(om_x) > eps) | (np.abs(om_y) > eps)
        angles_nz = np.arctan2(om_y[ind_nz], om_x[ind_nz])
        angles_nz -= self.defocus_ang

        defocus = np.zeros_like(om_x)
        defocus[ind_nz] = self.defocus_mean + self.defocus_diff * np.cos(2 * angles_nz)

        c2 = -np.pi * self.wavelength * defocus
        c4 = 0.5 * np.pi * (self.Cs * 1e7) * self.wavelength ** 3

        r2 = om_x ** 2 + om_y ** 2
        r4 = r2 ** 2
        gamma = c2 * r2 + c4 * r4
        h = np.sqrt(1 - self.alpha ** 2) * np.sin(gamma) - self.alpha * np.cos(gamma)

        if self.B:
            h *= np.exp(-self.B * r2)
        
        return h.squeeze()

    # apply ctf to given image
    def apply(self, image):
        """
        Apply a `Filter` object to the Image and returns a new Image.
        :param filter: An object of type `Filter`.
        :return: A new filtered `Image` object.
        """
        filter_values = self.evaluate_grid(len(image)) # check
        im_f = aux_functions.centered_fft2(image)
        im_f = filter_values * im_f
        im = aux_functions.centered_ifft2(im_f)
        im = np.real(im)

        return im

    # use wiener filter to remove ctf given snr (to be determined given images)
    # see: https://gist.github.com/danstowell/f2d81a897df9e23cc1da and https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/VELDHUIZEN/node15.html
    # and https://github.com/scikit-image/scikit-image/blob/main/skimage/restoration/deconvolution.py#L11-L147
    def remove(self, image, snr):
        # OLD IMPLEMENTATION (KINDA WORKS) COMMENTED OUT:
        filter_values = self.evaluate_grid(len(image))
        wiener_filter = np.conj(filter_values) / (np.abs(filter_values)**2 + 1/snr)
        filtered_image_fft = aux_functions.centered_fft2(wiener_filter * image)
        filtered_image = aux_functions.centered_ifft2(filtered_image_fft)
        filtered_image = np.real(filtered_image)
        
        return filtered_image
        
        # NEW implementation, using skimage.restoration unsupervised wiener filter:
        # DON'T need snr for this?
        # psf = np.real(aux_functions.centered_ifft2(self.evaluate_grid(len(image))))
        # filtered_image, _ = restoration.unsupervised_wiener(image, psf)

        # return filtered_image

class RadialCTFFilter(CTFFilter):
    def __init__(
            self, pixel_size=10, voltage=200, defocus=15000, Cs=2.26, alpha=0.07, B=0
    ):
        super().__init__(
            pixel_size=pixel_size,
            voltage=voltage,
            defocus_u=defocus,
            defocus_v=defocus,
            defocus_ang=0,
            Cs=Cs,
            alpha=alpha,
            B=B,
        )

# ctf averaging filter for M step; different than above

class CTFAveragingFilter(Filter):
    # initialize with array of ctf objects
    def __init__(self, ctf_array):
        super().__init__(dim=2, radial=False) # check radial=False?
        self.ctf_array = np.array(ctf_array)

    def _evaluate(self, omega):
        # only evaluate sum of squared reciprocals
        ctf_evaluated_array = np.array([ctf.evaluate(omega) for ctf in self.ctf_array])

        factors = np.reciprocal(np.sum(ctf_evaluated_array**2, axis=0))
        
        return factors
        
        # return np.sum(ctf_evaluated_array**2, axis=0) # ONLY FOR WIENER FILTER APPROACH

    # apply to array of (rotationally-aligned) images
    def apply(self, images):
        assert len(images) == len(self.ctf_array), "Need same number of images and ctfs"
        
        # LS APPROACH BELOW, KINDA WORKS:
        # apply ctf to each image, then fourier transform, then sum
        ctf_images = np.array([self.ctf_array[i].apply(images[i]) for i in range(len(images))]) # precompute
        ctf_images = np.array([aux_functions.centered_fft2(img) for img in ctf_images]) # ctf_images in fourier domain
        image = sum(ctf_images) # image in fourier domain
        
        # apply sum of squared reciprocals filter, from CTFFilter apply()
        filter_values = self.evaluate_grid(len(image))
        
        im_f = filter_values * image
        im = aux_functions.centered_ifft2(im_f)
        im = np.real(im)
        
        return im
        
        # # WIENER FILTER APPROACH BELOW
        # # apply ctf to each image, then fourier transform, then sum
        # ctf_images = np.array([self.ctf_array[i].apply(images[i]) for i in range(len(images))]) # pythonic?
        # image = sum(ctf_images) # image in fourier domain
        
        # psf = np.real(aux_functions.centered_ifft2(self.evaluate_grid(len(image))))
        # filtered_image, _ = restoration.unsupervised_wiener(image, psf)

        # return filtered_image

