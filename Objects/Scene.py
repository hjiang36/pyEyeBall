from Objects.Illuminant import Illuminant
from math import tan, atan2
from Objects.Display import Display
from Utility.Transforms import *
import numpy as np
__author__ = 'HJ'


class Scene:
    """
    Class describing scene radiance
    """
    name = "Scene"     # name of the object
    photons = None     # scene photons
    wave = None        # wavelength sampling in nm
    fov = None         # horizontal field of view of the scene in degree
    dist = None        # viewing distance in meters
    illuminant = None  # illuminant

    def __init__(self):
        """
        Class constructor, initializing parameters
        :return: class object with properties initialized
        """
        pass

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

        # init basic display parameters
        scene = cls()
        scene.illuminant = Illuminant(wave=d.wave)
        scene.wave = d.wave  # wavelength samples
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
        return scene

    def adjust_illuminant(self, il):
        """
        Change illuminant of the scene
        :param il: Illuminant object instance
        :return: scene object with illuminant and photons updated
        """
        mean_lum = self.mean_luminance

    def adjust_luminance(self, lum):
        """
        Adjust luminance of the scene
        :param lum: mean luminance in cd/m2
        :return: scene object with mean luminance adjusted
        """
        self.photons /= self.luminance / lum

    @property
    def mean_luminance(self):  # mean luminance of scene in cd/m2
        pass

    @property
    def luminance(self):  # luminance image of the scene
        pass

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



