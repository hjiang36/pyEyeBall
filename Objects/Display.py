from Utility.IO import spectra_read
from Utility.Transforms import rad_to_deg, xyz_from_energy, xyz_to_xy
from math import atan2
import numpy as np
from os.path import isfile, join
from os import listdir
from scipy.io import loadmat
from collections import namedtuple
from Data.path import get_data_path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
        intensity_map = tmp["dixel"][0, 0][0]['intensitymap'][0]
        control_map = tmp["dixel"][0, 0][0]['controlmap'][0]
        n_pixels = tmp["dixel"][0, 0][0]['nPixels'][0][0][0]
        d.dixel = dixel(intensity_map, control_map, n_pixels)

        # return
        return d

    def compute(self, img):
        pass

    def plot(self, param):
        """
        generate plots for display parameters and properties
        :param param: string, indicating which plot to generate, can be chosen from:
                      'spd', 'gamma', 'invert gamma', 'gamut'
        :return: None, but plot will be shown
        """

        # process param to be lowercase and without spaces
        param = str(param).lower().replace(" ", "")
        plt.ion()  # enable interactive mode

        # making plot according to param
        if param == "spd":  # plot spectra power distribution of display
            plt.plot(self.wave, self.spd)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Energy (watts/sr/m2/nm)")
            plt.grid()
            plt.show()
        elif param == "gamma":  # plot gamma table of display
            plt.plot(self.gamma)
            plt.xlabel("DAC")
            plt.ylabel("Linear")
            plt.grid()
            plt.show()
        elif param == "invertgamma":  # plot invert gamma table of display
            plt.plot(np.linspace(0, 1, self.invert_gamma.shape[0]), self.invert_gamma)
            plt.xlabel("Linear")
            plt.ylabel("DAC")
            plt.grid()
            plt.show()
        elif param == "gamut":  # plot gamut of display
            # plot human visible range
            xyz = spectra_read('XYZ.mat', self.wave)
            xy = xyz_to_xy(xyz, rm_nan=True)
            xy = np.concatenate((xy, xy[0:1, :]), axis=0)
            plt.plot(xy[:, 0], xy[:, 1])

            # plot gamut of display
            xy = xyz_to_xy(self.rgb2xyz, rm_nan=True)
            xy = np.concatenate((xy, xy[0:1, :]), axis=0)
            plt.plot(xy[:, 0], xy[:, 1])
            plt.xlabel("CIE-x")
            plt.ylabel("CIE-y")
            plt.grid()
            plt.show()
        else:
            raise(ValueError, "Unsupported input param")

    @staticmethod
    def ls_display():
        """
        static method that can be used to list all available display calibration file
        :return: a list of available display calibration files
        """
        return [f for f in listdir(join(get_data_path(), 'Display')) if isfile(f)]

    @property
    def n_bits(self):
        """
        color bit-depth of the display
        :return: scalar of color bit-depth
        """
        return round(np.log2(self.gamma.shape[0]))

    @property
    def n_levels(self):
        """
        number of DAC levels of display, usually it equals to 2^n_bits
        :return: number of DAC levels of display
        """
        return self.gamma.shape[0]

    @property
    def bin_width(self):
        """
        wavelength sample interval in nm
        :return: wavelength sample interval
        """
        if self.wave.size > 1:
            return self.wave[1] - self.wave[0]
        else:
            return None

    @property
    def n_primaries(self):
        """
        number of primaries of display, usually it equals to 3
        :return: number of primaries of display
        """
        return self.spd.shape[1]

    @property
    def invert_gamma(self, n_steps=None):
        """
        Invert gamma table which can be used to convert linear value to DAC value
        :return: Invert gamma table
        """
        # init default value of n_steps
        if n_steps is None:
            n_steps = self.n_levels

        # set up parameters
        y = range(self.n_levels)
        inv_y = np.linspace(0, 1, n_steps)
        lut = np.zeros([n_steps, self.n_primaries])

        # interpolate for invert gamma table
        for ii in range(self.n_primaries):
            # make sure gamma table is mono-increasing
            x = sorted(np.squeeze(self.gamma[:, ii]))

            # interpolate
            f = interp1d(x, y, bounds_error=False)
            lut[:, ii] = f(inv_y)

            # set extrapolation value
            lut[inv_y < np.min(x), ii] = 0
            lut[inv_y > np.max(x), ii] = self.n_levels - 1
        return lut

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

    @property
    def white_spd(self):
        return np.sum(self.spd, axis=1)
