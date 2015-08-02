__author__ = 'HJ'

import os.path
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from Data.path import get_data_path


def spectra_read(fn, wave, extra_val=0):
    """
    read spectra data file and interpolate to target wavelength
    :param fn: spectra data file name, e.g. "XYZ"
    :param wave: target wavelength samples
    :extra_val: extrapolation value
    :return: array of spectra data
    """

    # get full path of file
    fn = data_full_path(fn)

    # load data from file
    data_dict = loadmat(fn)
    assert "wavelength" in data_dict.keys() and "data" in data_dict.keys(), "Invalid data file"
    data = data_dict["data"]
    wavelength = np.squeeze(data_dict["wavelength"].astype(float))

    # interpolate for wavelength samples
    f = interp1d(wavelength, data, axis=0, fill_value=extra_val, bounds_error=False)
    return f(wave)


def data_full_path(fn):
    """
    Get full path of a data file
    :param fn: data file name
    :return: full path of the data file
    """
    for root, dirs, files in os.walk(get_data_path()):
        if fn in files:
            return os.path.join(root, fn)
    raise OSError("File not exists")
