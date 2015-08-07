from Utility.Transforms import deg_to_rad, quanta_to_energy, rad_to_deg, xyz_to_srgb, xyz_from_energy
from Utility.IO import spectra_read
from Objects.Scene import Scene
from scipy.constants import pi
from scipy.interpolate import interp1d, interp2d
from scipy.special import jv
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from math import atan, floor, tan, ceil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

__author__ = 'HJ'


class Optics:
    """
    Human optics class and optical image
    In this class, we assume human optics is shift-invariant and off-axis method is cos4th
    """
    name = "Human Optics"         # name of the class instance
    _wave = np.array([])          # wavelength samples in nm
    photons = np.array([])        # irradiance image
    f_number = 5                  # f-number of the lens
    dist = 1                      # Object distance in meters
    fov = 1                       # field of view of the optical image in degree
    focal_length = 0.017          # focal lens of optics in meters
    OTF = None                    # optical transfer functions, to get otf data at fx, fy use OTF[ii](freq)
    transmittance = np.array([])  # transmittance of optics

    def __init__(self, pupil_diameter=0.003,
                 focal_length=0.017,
                 wave=np.array(range(400, 710, 10))):
        """
        class constructor
        :return: instance of optics class
        """
        # set focal length, f-number and wavelength samples
        np.seterr(invalid='ignore')
        self.focal_length = focal_length
        self.f_number = focal_length / pupil_diameter
        self._wave = wave

        # set lens quanta transmittance
        lens_density = spectra_read("lensDensity.mat", wave)
        self.transmittance = 10**(-lens_density)

        # compute human optical transfer function
        # Reference: Marimont & Wandell, J. Opt. Soc. Amer. A,  v. 11, p. 3113-3122 (1994)
        max_freq = 90
        defocus = 1.7312 - (0.63346 / (wave*1e-3 - 0.2141))  # defocus as function of wavelength

        w20 = pupil_diameter**2 / 8 * (1/focal_length * defocus) / (1/focal_length + defocus)  # Hopkins w20 parameter
        sample_sf = np.array(range(max_freq))
        achromatic_mtf = 0.3481 + 0.6519 * np.exp(-0.1212 * sample_sf)

        # compute otf at each wavelength
        self.OTF = [None] * wave.size
        for ii in range(wave.size):
            s = 1 / tan(deg_to_rad(1)) * wave[ii] * 2e-9 / pupil_diameter * sample_sf  # reduced spatial frequency
            alpha = np.abs(4*pi / (wave[ii] * 1e-9) * w20[ii] * s)
            otf = optics_defocus_mtf(s, alpha) * achromatic_mtf

            # set otf as interpolation function to object
            # programming note:
            #   In Matlab, there is a command fftshift after otf interpolation
            #   In python, this step is taken care of it when we use the otf
            self.OTF[ii] = interp1d(sample_sf, otf, bounds_error=False, fill_value=0)

    def compute(self, scene):
        """
        Compute optical irradiance map
        :param scene: instance of scene class
        :return: None, but oi.photons is computed
        """
        # check inputs
        assert isinstance(scene, Scene), "scene should be of class Scene"

        # set field of view and wavelength samples
        self.fov = scene.fov
        scene.wave = self._wave

        # compute irradiance
        self.photons = pi / (1 + 4 * self.f_number**2 * (1 + abs(self.magnification))**2) * scene.photons

        # apply optics transmittance
        self.photons *= self.transmittance

        # apply the relative illuminant (off-axis) fall-off: cos4th function
        x, y = self.spatial_support
        s_factor = np.sqrt(self.image_distance**2 + x**2 + y**2)
        self.photons *= (self.image_distance / s_factor[:, :, None])**4

        # apply optical transfer function of the optics
        fx, fy = self.frequency_support
        for ii in range(self.wave.size):
            otf = fftshift(self.OTF[ii](np.sqrt(fx**2 + fy**2)))
            self.photons[:, :, ii] = np.abs(ifftshift(ifft2(otf * fft2(fftshift(self.photons[:, :, ii])))))

    def __str__(self):
        """
        Generate verbal description string of optics object
        :return: string that describes instance of optics object
        """
        return "Human Optics Instance: Description function not yet implemented"

    def plot(self, param, opt=None):
        """
        Generate plots for properties and attributes of optics object
        :param param: string, indicating which plot to generate
        :param opt: optional input, for some plotting param, this value is required
        :return: None, but plot will be shown
        """
        # process param
        param = str(param).lower().replace(" ", "")
        plt.ion()

        # generate plot according to param
        if param == "srgb":
            plt.imshow(self.srgb)
            plt.show()
        elif param == "otf":
            assert opt is not None, "wavelength to be plotted required as opt"
            freq = np.array(range(-80, 80))
            index = np.argmin(np.abs(self._wave - opt))
            plt.plot(freq, self.OTF[index](abs(freq)))
            plt.xlabel("Frequency (cycles/deg)")
            plt.ylabel("OTF Amplitude")
            plt.grid()
            plt.show()
        elif param == "psf":
            assert opt is not None, "Wavelength to be plotted required as opt"
            sx = np.linspace(-2e-5, 2e-5, 500)
            sy = np.linspace(-2e-5, 2e-5, 500)
            xx, yy = np.meshgrid(sx, sy)

            psf = self.psf(opt)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(xx * 1e6, yy * 1e6, psf(sx, sy), cmap=cm.coolwarm)  # plot in units of um
            plt.xlabel("Position (um)")
            plt.ylabel("Position (um)")
            plt.show()
        elif param == "transmittance":
            plt.plot(self._wave, self.transmittance)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Lens Transmittance")
            plt.grid()
            plt.show()
        else:
            raise(ValueError, "Unknown param")

    def visualize(self):
        pass

    @property
    def wave(self):
        return self._wave

    @wave.setter
    def wave(self, value):
        # interpolate photons and OTF data

        # set value of _wave in object
        self._wave = value

    @property
    def bin_width(self):  # wavelength sample bin width
        return self._wave[1] - self._wave[0]

    @property
    def shape(self):  # number of samples in class in (rows, cols)
        return self.photons.shape[0:2]

    @property
    def width(self):  # width of optical image in meters
        return 2 * self.image_distance * np.tan(deg_to_rad(self.fov)/2)

    @property
    def sample_size(self):  # length per sample in meters
        return self.width / self.n_cols

    @property
    def height(self):  # height of optical image in meters
        return self.sample_size * self.n_rows

    @property
    def image_distance(self):  # image distance / focal plane distance in meters
        return 1/(1/self.focal_length - 1/self.dist)

    @property
    def energy(self):  # optical image energy map
        return quanta_to_energy(self.photons, self._wave)

    @property
    def magnification(self):
        return -self.image_distance / self.dist

    @property
    def pupil_diameter(self):  # pupil diameter in meters
        return self.focal_length / self.f_number

    @property
    def spatial_support(self):  # spatial support of optical image in (support_x, support_y) in meters
        sy = np.linspace(-(self.n_rows - 1) * self.sample_size/2, (self.n_rows - 1) * self.sample_size/2, self.n_rows)
        sx = np.linspace(-(self.n_cols - 1) * self.sample_size/2, (self.n_cols - 1) * self.sample_size/2, self.n_cols)
        return np.meshgrid(sx, sy)

    @property
    def n_rows(self):
        return self.photons.shape[0]

    @property
    def n_cols(self):
        return self.photons.shape[1]

    @property
    def meters_per_degree(self):
        return self.width / self.fov

    @property
    def degrees_per_meter(self):  # conversion constant between degree and meter
        return 1/self.meters_per_degree

    @property
    def v_fov(self):  # field of view in vertical direction
        return 2*rad_to_deg(atan(self.height/self.image_distance/2))

    @property
    def frequency_support(self):  # frequency support of optical image in cycles / degree
        # compute max possible frequency in image
        max_freq = [self.n_cols/2/self.fov, self.n_rows/2/self.v_fov]
        fx = np.array(range(-floor(self.n_cols/2), floor(self.n_cols/2))) / ceil(self.n_cols/2) * max_freq[0]
        fy = np.array(range(-floor(self.n_rows/2), floor(self.n_rows/2))) / ceil(self.n_rows/2) * max_freq[1]
        return np.meshgrid(fx, fy)

    @property
    def xyz(self):  # xyz image of the optical image
        return xyz_from_energy(self.energy, self.wave)

    @property
    def srgb(self):  # srgb image of the optical image
        return xyz_to_srgb(self.xyz)

    def psf(self, wave):
        """
        get psf for certain wavelength
        :param wave: scalar, wavelength sample to be retrieved
        :return: point spread function for certain wavelength, to get psf data, use psf(spatial_support)
        """
        # get frequency support
        max_freq = 200
        fx, fy = np.meshgrid(range(-max_freq, max_freq), range(-max_freq, max_freq))
        freq = np.sqrt(fx**2 + fy**2)

        # get otf at wavelength
        index = np.argmin(np.abs(self._wave - wave))
        otf = self.OTF[index](freq)

        # get spatial support
        spatial_spacing = 1/2/max_freq * self.meters_per_degree  # spatial spacing in meters
        sx = np.array(range(-max_freq, max_freq)) * spatial_spacing
        sy = np.array(range(-max_freq, max_freq)) * spatial_spacing

        # compute psf
        psf = np.abs(fftshift(ifft2(otf)))
        return interp2d(sx, sy, psf, bounds_error=False, fill_value=0)


def optics_defocus_mtf(s, alpha):
    """
    Diffraction limited mtf without aberrations but with defocus
    :param s: reduced spatial frequency
    :param alpha: defocus parameter, which is related to w20 of Hopkins
    :return: diffraction limited mtf without aberration but with defocus
    """
    # compute auxiliary parameters
    nf = np.abs(s)/2
    beta = np.sqrt(1 - nf**2)
    otf = nf  # allocate space for otf

    # compute perfect spatial frequencies of OTF
    index = (alpha == 0)
    otf[index] = 2/pi * (np.arccos(nf[index]) - nf[index] * beta[index])

    # compute defocused spatial frequencies of OTF
    index = (alpha != 0)
    h1 = beta[index] * jv(1, alpha[index]) + \
        1/2 * np.sin(2*beta[index]) * (jv(1, alpha[index]) - jv(3, alpha[index])) - \
        1/4 * np.sin(4*beta[index]) * (jv(3, alpha[index]) - jv(5, alpha[index]))

    h2 = np.sin(beta[index]) * (jv(0, alpha[index]) - jv(2, alpha[index])) + \
        1/3 * np.sin(3*beta[index]) * (jv(2, alpha[index]) - jv(4, alpha[index])) - \
        1/5 * np.sin(5*beta[index]) * (jv(4, alpha[index]) - jv(6, alpha[index]))

    otf[index] = 4/pi/alpha[index] * np.cos(alpha[index] * nf[index]) * h1 - \
        4/pi/alpha[index] * np.sin(alpha[index]*nf[index]) * h2

    # normalize
    return otf / otf[0]
