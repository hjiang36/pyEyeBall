from ..Utility.IO import spectra_read
from ..Utility.Transforms import quanta_to_energy, luminance_from_energy
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

""" Module for Illuminant light simulation

This module is designed to simulate different illumination lights. In current version, illumination light is assumed to
be spatially and temporally identically distributed across the whole field. In the future, we might want to allow time
varying or space aware illuminations.

There is only one class in the module: Illuminant.

Illuminant:
    Stores the spectral distribution of illuminant light in quanta units and can be adjustable based on mean luminance
    levels.

Connection with ISETBIO:
    Illuminant class is equivalent to illuminant structure defined in ISETBIO. And Illuminant instance can be created
    directly with ISETBIO illuminant files in .mat format.

"""

__author__ = 'HJ'


class Illuminant:
    """ Illuminant light simulation and computation

    This class stores illuminant spectra distribution data and does computations and analysis from there. In current
    version, there is no spatial or temporal varying features in this class.

    Args:
        name (str): name of the illuminant instance
        photons (numpy.ndarray): quanta distribution in each wavelength samples

    """

    def __init__(self, file_name='D65.mat', wave=None):
        """ Constructor of Illuminant class
        This function loads illuminant spectra data from ISETBIO .mat illuminant file. The file should contain two
        variables: data and wavelength

        Args:
            file_name (str): illuminant file name, e.g. D65.mat. The data file should be seated in Data/Illumination
            wave (numpy.ndarray): wavelength samples to be used. If None, or not specified, the default wavelength
                samples np.arange(400, 710, 10) will be used

        Examples:
            Create different illuminant
            >>> il = Illuminant("D65.mat")
            >>> il = Illuminant("D50.mat")
        """
        # Initialize instance attribute to default values
        self.name = "Illuminant"  # name of object
        if wave is None:
            self._wave = np.arange(400.0, 710.0, 10.0)
        else:
            self._wave = wave         # wavelength samples in nm
        self.photons = spectra_read(file_name, self._wave)  # quanta distribution in each wavelength samples

        # normalize current illuminant to have luminance of 100 cd/m2
        self.luminance = 100

    def __str__(self):
        """ Generate description string of the Illuminant instance

        This function generates string for Illuminant class. With the function, illuminant properties can be printed out
        easily with str(il)

        Returns:
            str: description string

        Examples:
            Description string for D50 light

            >>> il = Illuminant("D50.mat")
            >>> print(il)
            Illuminant Object: Illuminant (...and more...)
        """
        s = "Illuminant Object: " + self.name + "\n"
        s += "\tWavelength: " + str(np.min(self.wave)) + ":" + str(self.bin_width) + ":" + str(np.max(self.wave))
        return s

    def plot(self, param):
        """ Generate plots for illuminant parameters and properties

        Args:
            param (str): string which indicates the type of plot to generate. In current version, param can be chosen
                from "photons", "energy". param string is not case sensitive and blank spaces in param are ignored.

        Examples:
            plot illuminant photon distributions of D65 light
            >>> Illuminant().plot("photons")

        """
        # process param
        param = str(param).lower().replace(" ", "")

        # generate plot according to param
        if param == "energy":  # spectral distributions in energy units
            plt.plot(self._wave, self.energy)
            plt.xlabel("wavelength (nm)")
            plt.ylabel("Energy")
        elif param == "photons":  # spectral distribution in quanta units
            plt.plot(self._wave, self.photons)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Photons")
        else:
            raise(ValueError, "Unknown param")
        plt.grid
        plt.show()

    @property
    def energy(self):
        """ numpy.ndarray: illuminant energy"""
        return quanta_to_energy(self.photons, self._wave)

    @property
    def luminance(self):
        """float: mean luminance of light.
        If set it to a new value, the photons will be ajusted to match the desired luminance level
        """
        return luminance_from_energy(self.energy, self._wave)

    @luminance.setter
    def luminance(self, value):  # adjust mean luminance
        self.photons /= self.luminance/value

    @property
    def wave(self):
        """ numpy.ndarray: wavelength samples in nm
        If set it to a new value, the underlying photon data will be interpolated to new wavelength
        """
        return self._wave

    @wave.setter
    def wave(self, value):  # set wavelength samples and interpolate photons
        f = interp1d(self._wave, self.photons, bounds_error=False, fill_value=0, axis=0)
        self.photons = f(value)
        self._wave = value

    @property
    def bin_width(self):
        """float: wavelength sampling interval in nm"""
        return self._wave[1] - self._wave[0]
