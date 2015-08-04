from Utility.IO import data_full_path, spectra_read
from scipy.io import loadmat
from Utility.Transforms import quanta_to_energy, luminance_from_energy
import numpy as np

__author__ = 'HJ'


class Illuminant:
    """
    Class for illuminant light
    """

    name = "Illuminant"  # name of object
    wave = None          # wavelength samples in nm
    photons = None       # quanta distribution in each wavelength samples

    def __init__(self, fn='D65.mat', wave=None):
        """
        Constructor of class, loading illuminant from mat file
        :param fn: illuminant file name, e.g. D65.mat
        :return: initialized illuminant object
        """
        # load mat data file
        if wave is None:
            tmp = loadmat(data_full_path(fn))
            self.wave = np.squeeze(tmp["wavelength"].astype(float))
        else:
            self.wave = wave

        # set photons
        self.photons = spectra_read(fn, self.wave)

        # normalize current illuminant to have luminance of 100 cd/m2
        self.photons /= self.luminance / 100

    @property
    def energy(self):  # illuminant energy
        return quanta_to_energy(self.photons, self.wave)

    @property
    def luminance(self): # luminance of light
        return luminance_from_energy(self.energy, self.wave)


