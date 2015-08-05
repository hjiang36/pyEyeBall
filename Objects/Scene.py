from Objects.Illuminant import Illuminant
from math import tan, atan2
from Objects.Display import Display
from Utility.Transforms import *
from scipy.interpolate import interp1d
import numpy as np
__author__ = 'HJ'


class Scene:
    """
    Class describing scene radiance
    """
    name = "Scene"     # name of the object
    photons = None     # scene photons
    fov = 1.0          # horizontal field of view of the scene in degree
    dist = 1.0         # viewing distance in meters
    illuminant = None  # illuminant

    def __init__(self, scene_type=None):
        """
        Class constructor, initializing parameters
        :param scene_type: type of the scene, e.g. macbeth, noise, etc.
        :return: class object with properties initialized
        """
        # check inputs
        if scene_type is None:
            return

        # initialize scene object according to scene type
        raise(ValueError, 'Unsupported scene type')

    @classmethod
    def init_with_display_image(cls, d, img, is_sub=False):
        """
        Initialize scene with image on a calibrated display
        :param d: display object instance
        :param img: rgb image to be shown on the display, rgb should be in range of [0, 1]
        :param is_sub: logical, indicating whether or not to do sub-pixel rendering
        :return: scene object
        """
        # check inputs
        assert isinstance(d, Display), "d should be of class Display"
        assert isinstance(img, np.ndarray), "img should of type numpy.ndarray"

        # init basic display parameters
        scene = cls()
        scene.illuminant = Illuminant(wave=d.wave)
        scene.dist = d.dist  # viewing distance

        # compute horizontal field of view
        scene.fov = rad_to_deg(atan2(img.shape[1] * d.meters_per_dot, d.dist))

        # set illuminant as spd of display
        scene.illuminant.photons = energy_to_quanta(d.white_spd, d.wave)

        # gamma distortion for the input image
        img = np.round(img * (d.n_levels - 1))
        for ii in range(d.n_primaries):
            img[:, :, ii] = d.gamma[img, ii]

        # sub-pixel rendering if required
        if is_sub:
            img = d.compute(img)

        # compute radiance from image
        out_sz = [img.shape[0:2], d.wave.size]
        img = rgb_to_xw_format(img)
        scene.photons = energy_to_quanta(np.dot(img, d.spd.T).T, d.wave).T

        # add ambient quanta to scene photons
        scene.photons += d.ambient[:, None]

        # reshape photons
        scene.photons = scene.photons.reshape(out_sz)
        return scene

    def adjust_illuminant(self, il):
        """
        Change illuminant of the scene
        :param il: Illuminant object instance
        :return: scene object with illuminant and photons updated

        The sampling wavelength of new wavelength will be adjusted to be same as scene wavelength samples
        """
        # check input
        assert isinstance(il, Illuminant), "il should be an instance of class Illuminant"

        # save mean luminance
        mean_lum = self.mean_luminance

        # adjust il wavelength
        il.wave = self.wave

        # update photons
        sz = self.photons.shape
        sz[0:-1] = 1
        self.photons *= np.reshape(il.photons / self.illuminant.photons, sz)

        # update illuminat spd
        self.illuminant = il

        # make sure mean luminance is not changed
        self.adjust_luminance(mean_lum)

    def adjust_luminance(self, lum):
        """
        Adjust mean luminance of the scene
        :param lum: mean luminance in cd/m2
        :return: scene object with mean luminance adjusted
        """
        self.photons /= self.mean_luminance / lum

    def __str__(self):
        """
        Generate verbal description string of scene object
        :return: description string of scene object
        """
        s = "Scene Object: " + self.name + "\n"
        s += "\tWavelength: " + str(np.min(self.wave)) + ":" + str(self.bin_width) + ":" + str(np.max(self.wave))
        s += " nm\n"
        s += "\t[Row, Col]: " + str(self.shape) + "\n"
        s += "\t[Width, Height]" + str([self.width, self.height]) + " m\n"
        s += "\tHorizontal field of view: " + str(self.fov) + " deg\n"
        s += "\tSample size: " + str(self.sample_size) + " meters/sample\n"
        s += "\tMean luminance: " + str(self.mean_luminance) + " cd/m2\n"
        return s

    @property
    def wave(self):
        return self.illuminant.wave

    @wave.setter
    def wave(self, value):  # set wavelength samples and interpolate data
        # update photons
        sz = self.photons.shape
        sz[0:-1] = 1
        f = interp1d(np.reshape(self.wave, sz), self.photons, bounds_error=False, fill_value=0)
        self.photons = f(np.reshape(value, sz))

        # update illuminant
        self.illuminant.wave = value

    @property
    def mean_luminance(self):  # mean luminance of scene in cd/m2
        return np.mean(self.luminance)

    @property
    def luminance(self):  # luminance image of the scene
        return luminance_from_energy(self.energy, self.wave)

    @property
    def shape(self):  # shape of scene in (rows, cols)
        return self.photons.shape[0:2]

    @property
    def width(self):  # width of scene in meters
        return 2 * self.dist * tan(deg_to_rad(self.fov))

    @property
    def height(self):  # height of scene in meters
        return self.sample_size * self.shape[0]

    @property
    def sample_size(self):  # size of each sample in meters
        return self.width / self.shape[1]

    @property
    def bin_width(self):  # wavelength sample interval in nm
        return self.wave[1] - self.wave[0]

    @property
    def energy(self):  # scene energy
        return quanta_to_energy(self.photons, self.wave)

    @property
    def xyz(self):  # xyz image of the scene
        return xyz_from_energy(self.energy, self.wave)

    def plot(self, param):
        """
        Generate plots for scene parameters and properties
        :param param: String, indicating which plot to generate
        :return: None, but plot will be shown
        """
        pass

    def visualize(self):
        """
        GUI for scene object
        :return: None, but GUI will be shown
        """
        pass
