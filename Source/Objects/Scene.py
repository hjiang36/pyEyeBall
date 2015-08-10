from .Illuminant import Illuminant
from math import tan, atan2
from .Display import Display
from ..Utility.Transforms import *
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
__author__ = 'HJ'


class Scene:
    """
    Class describing scene radiance
    """

    def __init__(self, scene_type="macbeth", wave=None, name="Scene",
                 il=Illuminant(), fov=1.0, dist=1.0, **kwargs):
        """
        Class constructor, initializing parameters
        :param scene_type: type of the scene, e.g. macbeth, noise, etc.
        :param il: Illuminant object instance
        :return: class object with properties initialized
        """
        # check inputs
        if isinstance(il, str):
            il = Illuminant(il)
        assert isinstance(il, Illuminant), "il should be an instance of Illuminant class"
        if wave is not None:  # interploate for wave
            il.wave = wave

        # Initialze instance attribute to default values
        self.name = name                # name of the object
        self.photons = np.array([])     # scene photons
        self.fov = fov                  # horizontal field of view of the scene in degree
        self.dist = dist                # viewing distance in meters
        self.illuminant = il            # illuminant

        # switch by scene_type
        scene_type = str(scene_type).lower().replace(" ", "")
        if scene_type == "macbeth":  # macbeth color checker
            if "patch_size" in kwargs:
                patch_size = kwargs["patch_size"]
            else:
                patch_size = 16
            if np.isscalar(patch_size):
                patch_size = [patch_size, patch_size]
            self.name = "Scene of Macbeth Color Checker (" + il.name + ")"
            # load surface reflectance
            surface = np.reshape(spectra_read("macbethChart.mat", self.wave).T, (4, 6, self.wave.size), order="F")
            # compute photons
            self.photons = np.zeros((4*patch_size[0], 6*patch_size[1], self.wave.size))
            for ii in range(self.wave.size):
                self.photons[:, :, ii] = np.kron(surface[:, :, ii], np.ones((patch_size[0], patch_size[1])))
            # multiply by illuminant
            self.photons *= il.photons

        elif scene_type == "noise":  # white noise pattern
            # get scene size
            if "scene_size" in kwargs:
                scene_size = kwargs["scene_sz"]
            else:
                scene_size = np.array([128, 128])
            self.name = "Scene of white noise"
            # generate noise pattern and compute photons
            noise_img = np.random.rand(scene_size[0], scene_size[1])
            self.photons = noise_img[:, :, None] * il.photons
        else:
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
        scene.fov = 2 * rad_to_deg(atan2(img.shape[1] * d.meters_per_dot / 2, d.dist))

        # set illuminant as spd of display
        scene.illuminant.photons = energy_to_quanta(d.white_spd, d.wave)

        # gamma distortion for the input image
        img = np.round(img * (d.n_levels - 1))
        for ii in range(d.n_primaries):
            img[:, :, ii] = d.gamma[img[:, :, ii].astype(int), ii]

        # sub-pixel rendering if required
        if is_sub:
            img = d.compute(img)

        # compute radiance from image
        out_sz = np.concatenate((np.array(img.shape[0:2]), [d.wave.size]))
        img = rgb_to_xw_format(img)
        scene.photons = energy_to_quanta(np.dot(img, d.spd.T), d.wave)

        # add ambient quanta to scene photons
        scene.photons += d.ambient

        # reshape photons
        scene.photons = scene.photons.reshape(out_sz, order="F")
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
        self.photons *= il.photons / self.illuminant.photons

        # update illuminant spd
        self.illuminant = il

        # make sure mean luminance is not changed
        self.mean_luminance = mean_lum

    def __str__(self):
        """
        Generate verbal description string of scene object
        :return: description string of scene object
        """
        s = "Scene Object: " + self.name + "\n"
        s += "\tWavelength: " + str(np.min(self.wave)) + ":" + str(self.bin_width) + ":" + str(np.max(self.wave))
        s += " nm\n"
        s += "\t[Row, Col]: " + str(self.shape) + "\n"
        s += "\t[Width, Height]: " + str([self.width, self.height]) + " m\n"
        s += "\tHorizontal field of view: " + str(self.fov) + " deg\n"
        s += "\tSample size: " + str(self.sample_size) + " meters/sample\n"
        s += "\tMean luminance: " + str(self.mean_luminance) + " cd/m2"
        return s

    @property
    def wave(self):
        return self.illuminant.wave

    @wave.setter
    def wave(self, value):  # set wavelength samples and interpolate data
        # update photons
        f = interp1d(self.wave, self.photons, bounds_error=False, fill_value=0)
        self.photons = f(value)

        # update illuminant
        self.illuminant.wave = value

    @property
    def mean_luminance(self):  # mean luminance of scene in cd/m2
        return np.mean(self.luminance)

    @mean_luminance.setter
    def mean_luminance(self, value):  # adjust mean luminance of the scene
        self.photons /= self.mean_luminance / value

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

    @property
    def srgb(self):  # srgb image of the scene
        return xyz_to_srgb(self.xyz)

    def plot(self, param):
        """
        Generate plots for scene parameters and properties
        :param param: String, indicating which plot to generate
        :return: None, but plot will be shown
        """
        # process param to be lowercase and without spaces
        param = str(param).lower().replace(" ", "")
        plt.ion()  # enable interactive mode

        # making plot according to param
        if param == "illuminantenergy":  # energy of illuminant
            self.illuminant.plot("energy")
        elif param == "illuminantphotons":  # photons of illuminant
            self.illuminant.plot("photons")
        elif param == "srgb":  # srgb image of the scene
            plt.imshow(self.srgb)
            plt.show()
        else:
            raise(ValueError, "Unknown parameter")

    def visualize(self):
        """
        GUI for scene object
        :return: None, but GUI will be shown
        """
        pass