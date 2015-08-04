from Utility.Transforms import deg_to_rad, quanta_to_energy, rad_to_deg
from Objects.Scene import Scene
from scipy.constants import pi
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from math import atan, floor
import numpy as np
__author__ = 'HJ'


class Optics:
    """
    Human optics class and optical image
    In this class, we assume human optics is shift invariant and off-axis method is cos4th
    """
    name = "Human Optics"         # name of the class instance
    _wave = np.array([])          # wavelength samples in nm
    photons = np.array([])        # irradiance image
    f_number = 5                  # f-number of the lens
    dist = 1                      # Object distance in meters
    fov = 1                       # field of view of the optical image in degree
    focal_length = 0.017          # focal lens of optics in meters
    OTF = None                    # optical transfer functions, to get otf data at fx, fy use OTF[ii](fx, fy)
    transmittance = np.array([])  # transmittance of optics

    def __init__(self, pupil_diameter=0.0015,
                 focal_length=0.017,
                 wave=np.array(range(400, 710, 10))):
        """
        class constructor
        :return: instance of optics class
        """
        # set focal length, f-number and wavelength samples
        self.focal_length = focal_length
        self.f_number = focal_length / pupil_diameter
        self._wave = wave

        # compute human optical transfer function

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
        self._wave = scene.wave

        # compute irradiance
        self.photons = pi / (1 + 4 * self.f_number**2 * (1 + np.abs(self.magnification))**2) * scene.photons

        # apply optics transmittance
        self.photons *= np.reshape(self.transmittance, [1, 1, self.wave.size])

        # apply the relative illuminant (off-axis) fall-off: cos4th function
        x, y = self.spatial_support
        s_factor = np.sqrt(self.image_distance**2 + x**2 + y**2)
        self.photons *= (self.image_distance / s_factor)**4

        # apply optical transfer function of the optics
        fx, fy = self.frequency_support
        for ii in range(self.wave.size):
            otf = self.OTF[ii](fx, fy)
            self.photons[:, :, ii] = np.abs(ifftshift(ifft2(otf * fft2(fftshift(self.photons[:, :, ii])))))

    def __str__(self):
        """
        Generate verbal description string of optics object
        :return: string that describes instance of optics object
        """
        return "Human Optics Instance: Description function not yet implemented"

    def plot(self, param):
        pass

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
        sx = np.linspace(-(self.n_rows - 1) * self.sample_size/2, (self.n_rows - 1) * self.sample_size/2, self.n_rows)
        sy = np.linspace(-(self.n_cols - 1) * self.sample_size/2, (self.n_cols - 1) * self.sample_size/2, self.n_cols)
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
        return 2*rad_to_deg(atan(self.height/self.dist/2))

    @property
    def frequency_support(self):  # frequency support of optical image in cycles / degree
        # compute max possible frequency in image
        max_freq = [self.n_cols/2/self.fov, self.n_rows/2/self.v_fov]
        fx = np.array(range(-floor(self.n_cols/2), floor(self.n_cols/2)+1)) / (floor(self.n_cols/2)+1) * max_freq[0]
        fy = np.array(range(-floor(self.n_rows/2), floor(self.n_rows/2)+1)) / (floor(self.n_rows/2)+1) * max_freq[1]
        return np.meshgrid(fx, fy)
