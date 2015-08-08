from Objects.Optics import Optics
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Utility.IO import spectra_read

__author__ = 'HJ'


class Sensor:
    """
    Class describing human cone mosaic and isomerizations
    """
    name = "Human Cone Mosaic"        # name of the instance of Sensor class
    mosaic = np.array([])             # 2D matrix, indicating cone type at each position
    cone = ConePhotopigment()         # cone photopigments properties
    cone_width = 2e-6                 # cone width in meters, default 2 um
    cone_height = 2e-6                # cone height in meters, default 2 um
    photons = np.array([])            # cone isomerization rate
    macular = np.array([])            # relative transmittance of macular pigment
    density = np.array([.6, .3, .1])  # spatial density of three types of cones
    position = np.array([0, 0])       # eye movement positions
    
    def __init__(self):
        pass

    def __str__(self):
        pass

    def plot(self, param):
        pass

    def visualize(self):
        pass

    def compute(self, oi):
        pass

    @property
    def wave(self):
        return self.cone.wave

    @wave.setter
    def wave(self, value):
        # interpolate photons data and macular data here
        self.cone.wave = value


class ConePhotopigment:
    """
    Class describing cone photopigment properties
    """
    name = "Human Cone Photopigment"          # name of the instance of this class
    _wave = np.array(range(400, 700, 10))     # wavelength samples in nm
    absorbance = np.array([])                 # cone absorbance
    optical_density = np.array([.5, .5, .4])  # optical density of three types of cones
    peak_efficiency = np.array([2, 2, 2])/3   # peak efficiency of three types of cones

    def __init__(self, wave=None):
        if wave is not None:
            self._wave = wave
        self.absorbance = 10.0**spectra_read("coneAbsorbance.mat", self._wave)

    @property
    def wave(self):  # wavelength samples in nm
        return self._wave

    @wave.setter  # set wavelength samples and interpolate data
    def wave(self, value):
        f = interp1d(self._wave, self.absorbance, bounds_error=False, fill_value=0)
        self.absorbance = f(value)
        self._wave = value

    @property
    def absorptance(self):  # cone absorptance without ocular media
        return 1 - 10.0**(-self.absorbance*self.optical_density)

    @property
    def quanta_fundamentals(self):  # quantal fundamentals of cones
        qe = self.absorptance * self.peak_efficiency
        return qe / np.max(qe, axis=0)
