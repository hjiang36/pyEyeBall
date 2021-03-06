__author__ = 'HJ'

from scipy.constants import c, h, pi
from .IO import spectra_read
import numpy as np


def energy_to_quanta(energy, wavelength):
    """
    Convert data from energy units (e.g. watts) to quanta units (e.g. photons)
    :param energy: 2D data with energy at each wavelength in rows, or 3D with wavelength in third dimension
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

    assert wavelength.size == energy.shape[-1], "Input size mismatch"

    # Convert
    photons = energy/(h*c) * 1e-9 * wavelength

    # return as same shape of energy input
    return photons.reshape(energy_sz, order="F")


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

    assert wavelength.size == photons.shape[-1], "Input size mismatch"

    # Convert

    energy = h * c * 1e9 * photons / wavelength

    # Return as same shape of input
    return energy.reshape(photons_sz, order="F")


def rgb_to_xw_format(rgb):
    """
    Convert data from 3D matrix (r*c*w) to a 2D matrix ((r*c)*w)
    :param rgb: 3D matrix (r*c*w)
    :return: 2D matrix ((r*c)*w)
    """
    assert isinstance(rgb, np.ndarray), "rgb input should be an 3D array"
    assert rgb.ndim == 3, "rgb input should be an 3D array"
    return rgb.reshape((rgb.shape[0]*rgb.shape[1], rgb.shape[2]), order='F')


def xw_to_rgb_format(xw, sz):
    """
    Convert data from 2D spatial-wavelength representation to 3D representation
    :param xw: data in 2D spatial-wavelength representation
    :param sz: desired output data size
    :return: data in shape of sz
    """
    assert isinstance(xw, np.ndarray), "xw input should be an 2D array"
    return xw.reshape(sz, order='F')


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


def xyz_to_xy(xyz, rm_nan=False):
    """
    Convert XYZ value to xy
    :param xyz: N-by-3 array with xyz in rows
    :param rm_nan: logical, indicating whether or not to remove output rows with nan values
    :return: N-by-2 array with xy in rows
    """
    np.seterr(divide='ignore', invalid='ignore')
    row_sum = np.sum(xyz, axis=1)
    xyz = xyz / row_sum[:, None]
    if rm_nan:
        xyz = xyz[~np.isnan(xyz).any(axis=1), :]
    return xyz[:, 0:2]


def xyz_to_srgb(xyz):
    """
    Convert XYZ value to sRGB value
    :param xyz: XYZ values in 2D or 3D matrix format, the last dimension of xyz should be 3
    :return: sRGB values in same shape of input xyz
    Reference: https://en.wikipedia.org/wiki/SRGB
    """
    # make sure that max of Y is no larger than 1
    xyz = np.array(xyz)
    if np.max(xyz[:, :, 1]) > 1:
        xyz /= np.max(xyz[:, :, 1])

    sz = xyz.shape
    if xyz.ndim == 3:  # 3D matrix, need convert to XW format
        xyz = rgb_to_xw_format(xyz)

    # convert to linear rgb
    conversion_matrix = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.dot(conversion_matrix, xyz.T).T

    # clip linear rgb to be between 0 and 1
    rgb[rgb > 1] = 1
    rgb[rgb < 0] = 0

    # non-linear distortion in srgb
    index = rgb > 0.0031308
    rgb[np.logical_not(index)] *= 12.92
    rgb[index] = 1.055 * rgb[index]**(1/2.4) - 0.055

    # convert to same shape as xyz
    return np.reshape(rgb, sz, order="F")


def xyz_from_energy(energy, wave):
    """
    Compute XYZ value from energy distributions
    :param energy: matrix with wavelength samples in rows
    :param wave: array of wavelength samples in nm
    :return: XYZ values
    """
    # check input energy type
    sz = np.array(energy.shape[0:2])
    if energy.ndim == 3:
        is_3d = True
        energy = rgb_to_xw_format(energy)
    else:
        is_3d = False
    s = spectra_read("XYZ.mat", wave)
    xyz = 683 * (wave[1]-wave[0]) * np.dot(energy, s)

    # make sure xyz are in same shape of input
    if is_3d:
        xyz = xw_to_rgb_format(xyz, np.concatenate((sz, [3])))
    return xyz


def luminance_from_energy(energy, wave):
    """
    Compute luminance of light from energy
    :param energy: energy of light at each wavelength
    :param wave: wavelength samples in nm
    :return: light luminance in cd/m2
    """
    # check input energy type
    sz = energy.shape
    if energy.ndim == 3:
        is_3d = True
        energy = rgb_to_xw_format(energy)
    else:
        is_3d = False

    # load luminosity data
    s = spectra_read("luminosity.mat", wave)
    lum = 683 * (wave[1] - wave[0]) * np.dot(np.squeeze(energy), np.squeeze(s))

    # make sure lum are in same shape of input
    if is_3d:
        lum = xw_to_rgb_format(lum, sz[0:2])
    return lum
