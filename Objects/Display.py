from Utility.IO import spectra_read
from Utility.Transforms import rad_to_deg, xyz_from_energy
from math import atan2
import numpy as np
from os.path import isfile, join
from os import listdir
from scipy.io import loadmat
from collections import namedtuple
from Data.path import get_data_path

__author__ = 'HJ'


class Display:
    """
    Class for display simulation
    """

    # Class properties
    name = "Display"    # name of display
    gamma = None        # gamma distortion table
    wave = None         # wavelength samples in nm
    spd = None          # spectra power distribution
    dist = 0            # viewing distance in meters
    dpi = 96            # spatial resolution in dots per inch
    is_emissive = True  # is emissive display or reflective display
    ambient = None      # dark emissive spectra distribution
    dixel = None        # pixel layout structure

    def __init__(self):
        pass

    @classmethod
    def init_with_isetbio_mat_file(cls, fn):
        """
        init display from isetbio .mat file
        :param fn: isetbio display calibration file with full path
        :return: display object
        """
        # load data
        assert isfile(fn), "file not exist"
        tmp = loadmat(fn)
        tmp = tmp["d"]

        # Init display structure
        d = cls()

        # set to object field
        d.name = tmp["name"][0, 0][0]  # name of display
        d.gamma = tmp["gamma"][0, 0]   # gamma distortion table
        d.wave = np.squeeze(tmp["wave"][0, 0].astype(float))  # wavelength samples in nm
        d.spd = tmp["spd"][0, 0]  # spectral power distribution
        d.dpi = tmp["dpi"][0, 0][0][0]  # spatial resolution in dots / inch
        d.dist = tmp["dist"][0, 0][0][0]  # viewing distance in meters
        d.is_emissive = tmp["isEmissive"][0, 0][0][0]  # is_emissive
        d.ambient = tmp["ambient"][0, 0].astype(float)

        # Init dixel structure
        dixel = namedtuple("Dixel", ["intensity_map", "control_map", "n_pixels"])
        intensity_map = tmp["dixel"][0, 0][0]['intensitymap'][0][0]
        control_map = tmp["dixel"][0, 0][0]['controlmap'][0][0]
        n_pixels = tmp["dixel"][0, 0][0]['nPixels'][0][0][0]
        d.dixel = dixel(intensity_map, control_map, n_pixels)

        # return
        return d

    def compute(self):
        pass

    def visualize(self):
        pass

    def plot(self, param):
        pass

    @staticmethod
    def ls_display():
        """
        static method that can be used to list all available display calibration file
        :return: a list of available display calibration files
        """
        return [f for f in listdir(join(get_data_path(), 'Display')) if isfile(f)]

    @property
    def n_bits(self):
        return round(np.log2(self.gamma.shape[0]))

    @property
    def n_levels(self):
        return self.gamma.shape[0]

    @property
    def bin_width(self):
        if self.wave.size > 1:
            return self.wave[1] - self.wave[0]
        else:
            return None

    @property
    def n_primaries(self):
        return self.spd.shape[1]

    @property
    def rgb2xyz(self):
        return xyz_from_energy(self.spd.T, self.wave)

    @property
    def rgb2lms(self):
        cone_spd = spectra_read("stockman.mat", self.wave)
        return np.dot(self.spd.T, cone_spd)

    @property
    def white_xyz(self):
        return np.dot(np.array([1, 1, 1]), self.rgb2xyz)

    @property
    def white_lms(self):
        return np.dot(np.array([1, 1, 1]), self.rgb2lms)

    @property
    def meters_per_dot(self):
        return 0.0254 / self.dpi

    @property
    def dots_per_meter(self):
        return self.dpi / 0.0254

    @property
    def deg_per_pixel(self):
        return rad_to_deg(atan2(self.meters_per_dot, self.dist))
