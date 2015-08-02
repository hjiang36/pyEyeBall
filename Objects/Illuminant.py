from Utility.IO import data_full_path, spectra_read
from scipy.io import loadmat
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
        fn = data_full_path(fn)
        if wave is None:
            tmp = loadmat(fn)
            self.wave = np.squeeze(tmp["wavelength"][0, 0].astype(float))
        else:
            self.wave = wave

        # set photons
        self.photons = spectra_read(fn, self.wave)



