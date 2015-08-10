from ..Utility.IO import spectra_read
from ..Utility.Transforms import quanta_to_energy, luminance_from_energy
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

__author__ = 'HJ'


class Illuminant:
    """
    Class for illuminant light
    """

    def __init__(self, fn='D65.mat', wave=np.array(range(400, 710, 10))):
        """
        Constructor of class, loading illuminant from mat file
        :param fn: illuminant file name, e.g. D65.mat
        :return: initialized illuminant object
        """
        # Initialize instance attribute to default values
        self.name = "Illuminant"  # name of object
        self._wave = wave         # wavelength samples in nm
        self.photons = None       # quanta distribution in each wavelength samples

        # set photons
        self.photons = spectra_read(fn, wave)

        # normalize current illuminant to have luminance of 100 cd/m2
        self.luminance = 100

    def __str__(self):
        """
        Generate description string of the class instance
        :return: description string
        """
        s = "Illuminant Object: " + self.name + "\n"
        s += "\tWavelength: " + str(np.min(self.wave)) + ":" + str(self.bin_width) + ":" + str(np.max(self.wave))
        return s

    def plot(self, param):
        """
        Generate plots for parameters and properties
        :param param: string, indicating which plot to generate
        :return: None, but plot will be shown
        """
        # process param
        param = str(param).lower().replace(" ", "")
        plt.ion()

        # generate plot according to param
        if param == "energy":
            plt.plot(self._wave, self.energy)
            plt.xlabel("wavelength (nm)")
            plt.ylabel("Energy")
            plt.grid()
            plt.show()
        elif param == "photons":
            plt.plot(self._wave, self.photons)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Photons")
            plt.grid()
            plt.show()
        else:
            raise(ValueError, "Unknown param")

    @property
    def energy(self):  # illuminant energy
        return quanta_to_energy(self.photons, self._wave)

    @property
    def luminance(self):  # luminance of light
        return luminance_from_energy(self.energy, self._wave)

    @luminance.setter
    def luminance(self, value):  # adjust mean luminance
        self.photons /= self.luminance/value

    @property
    def wave(self):
        return self._wave

    @property
    def bin_width(self):
        return self._wave[1] - self._wave[0]

    @wave.setter
    def wave(self, value):  # set wavelength samples and interpolate photons
        f = interp1d(self._wave, self.photons, bounds_error=False, fill_value=0, axis=0)
        self.photons = f(value)
        self._wave = value
