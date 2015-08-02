__author__ = 'HJ'

from scipy.constants import c, h, pi
from Utility.IO import spectra_read
import numpy as np


def energy_to_quanta(energy, wavelength):
    """
    Convert data from energy units (e.g. watts) to quanta units (e.g. photons)
    :param energy: 2D data with energy at each wavelength in columns, or 3D with wavelength in third dimension
    :param wavelength: an array of wavelength samples in nm
    :return: data in quanta units

    See also: quanta_to_energy
    """
    # Check inputs
    energy = np.array(energy)
    wavelength = np.array(wavelength)
    energy_sz = energy.shape

    if energy.ndim == 3:
        energy = rgb_to_xw_format(energy)

    assert wavelength.size == energy.shape[0], "Input size mismatch"

    # Convert
    photons = energy/(h*c) * 1e-9 * wavelength[:, None]

    # return as same shape of energy input
    return photons.reshape(energy_sz)


def quanta_to_energy(photons, wavelength):
    """
    Convert data from quanta units (e.g. photons) to energy units (e.g. watts)
    :param photons: 2D data with quanta at each wavelength in columns, or 3D with wavelength in third dimension
    :param wavelength: an array of wavelength samples in nm
    :return: data in energy units

    See also: energy_to_quanta
    """
    # Check inputs
    photons = np.array(photons)
    wavelength = np.array(wavelength)
    photons_sz = photons.shape

    if photons.ndim == 3:
        photons = rgb_to_xw_format(photons)

    assert wavelength.size == photons.shape[0], "Input size mismatch"

    # Convert
    energy = h * c * 1e9 * photons / wavelength[:, None]

    # Return as same shape of input
    return energy.reshape(photons_sz)


def rgb_to_xw_format(rgb):
    """
    Convert data from 3D matrix (r*c*w) to a 2D matrix ((r*c)*w)
    :param rgb: 3D matrix (r*c*w)
    :return: 2D matrix ((r*c)*w)
    """
    assert isinstance(rgb, np.ndarray), "rgb input should be an 3D array"
    assert rgb.ndim == 3, "rgb input should be an 3D array"
    return rgb.reshape(rgb.shape[0]*rgb.shape[1], rgb.shape[2])


def xw_to_rgb_format(xw, sz):
    """
    Convert data from 2D spatial-wavelength representation to 3D representation
    :param xw: data in 2D spatial-wavelength representation
    :param sz: desired output data size
    :return: data in shape of sz
    """
    assert isinstance(xw, np.ndarray), "xw input should be an 3D array"
    assert xw.ndim == 2, "xw input should be an 2D array"
    return xw.reshape(sz)


def deg_to_rad(deg):
    """
    convert degree to radians units
    :param deg: data in degrees
    :return: data in radians
    """
    return deg / 180 * pi


def rad_to_deg(rad):
    """
    convert radians to degree units
    :param rad: data in radians
    :return: data in degrees
    """
    return rad * 180 / pi


def xyz_from_energy(energy, wave):
    """
    Compute XYZ value from energy distributions
    :param energy: matrix with wavelength samples in rows
    :param wave: array of wavelength samples in nm
    :return: XYZ values
    """
    # check input energy type
    if energy.ndim == 3:
        is_3d = True
        sz = energy.shape
        energy = rgb_to_xw_format(energy)
    else:
        is_3d = False
    s = spectra_read("XYZ.mat", wave)
    xyz = 683 * (wave[1]-wave[0]) * np.dot(energy, s)

    # make sure xyz are in same shape as input
    if is_3d:
        xyz = xw_to_rgb_format(xyz, [sz[0:2], 3])
    return xyz
