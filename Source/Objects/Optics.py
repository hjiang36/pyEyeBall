from ..Utility.Transforms import deg_to_rad, quanta_to_energy, rad_to_deg, xyz_to_srgb, xyz_from_energy
from ..Utility.IO import spectra_read
from ..Data.path import get_data_path
from .Scene import Scene
from scipy.constants import pi
from scipy.interpolate import interp1d, interp2d
from scipy.special import jv
from scipy.misc import imresize
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from math import atan, floor, tan, ceil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PyQt4 import QtGui, QtCore
import pickle

""" Module for human optics charaterization, optical image related computations and visualizations

This module is used to characterize human optics properties and computes spectral irradiance image as well as other
useful statistics.

There are two classes in this module: Optics and OpticsGUI.

Optics:
    Optics class contains attributes and computational routines for human ocular components. Human optics are simulated
    with Marimont & Wandell (1995) algorithm. In current version, human optics is assumed to have a circular point
    spread function and is spatially shift invariant. Lens and macular pigment properties are also included in this
    class.

Optics GUI:
    This is a GUI that visualize optics properties with PyQt4. In most cases, instance of OpticsGUI class should not
    be created directly. Instead, to show the GUI for a certain human optics, call optics.visualize()

Connections with ISETBIO:
    Optics class is equivalent to oi structure defined in ISETBIO. ISETBIO Optics structure is re-organized and become
    some attributes in pyEyeBall.Optics class.

    OI structure in ISETBIO supports more non-human simulations (mainly for cameras). For example, there are pieces of
    code doing diffraction limited lens simulations.
    Macular pigment and lens transmittance are handled within sensor (human cone outer segment) structure in ISETBIO,
    while they are placed in Optics class in pyEyeBall. Thus, the optical images in pyEyeBall could be more yellowish
    than those from ISETBIO

    Wavefront toolbox in ISETBIO has not been transplant here. But this would be done in the near future (HJ).

"""

__author__ = 'HJ'


class Optics:
    """ Human optics and optical image

    The optics class converts scene radiance data (see Scene class) through customized human ocular system and form the
    irradiance image. The optics class stores optical transfer function at all wavelength, optical irradiance data,
    lens and macular pigment transmittance and other fundamental human optical properties (e.g. focal length, etc.).
    There are also a lot of computed properties and computational routines in this class. In current version, we assume
    human optics is shift-invariant and off-axis method is cos4th

    Attributes:
        name (str): name of the Optics instance
        photons (numpy.ndarray): optical image, spectral irradiance data
        fov (float): horizontal field of view of the optical image in degrees
        dist (float): the distance between the scene and the observer
        focal_length (float): focal length of human optics in meters
        lens_transmittance (numpy.ndarray): quanta tranmittance of the lens
        macualr_transmittance (numpy.ndarray): quanta tranmittance of macular pigment

    Note:

        1) wavelength samples and otf are stored as private attribute (see _wave and _otf). Property wave is defined as
        a computed attribute. Thus, for external usage, wavelength samples can be accessed or altered with optics.wave.
        To get otf at given wavelength, call class method otf() with desired wavelength and frequency support.

        2) In pyEyeBall.Optics, we use focal_length and pupil_diameter as fundamental properties while in ISETBIO.oi,
           f-number is used as fundamental property

    """

    def __init__(self, pupil_diameter=0.003, focal_length=0.017, dist=1.0, wave=None):
        """ Constructor for Optics Class
        Initialize parameters for human ocular system

        Args:
            pupil_diameter (float): pupil diameter in meters, usually human pupil diameter should be between 2 and 8
            focal_length (float): focal length of human, usually this is around 17 mm
            dist (float): object distance in meters, will be overrrided by scene.dist in compute method

        Note:
            To alter lens and macular pigment transmittance, we need to create a default optics instance first and then
            set the corresponding parameters

        Examples:
            >>> oi = Optics(pupil_diameter=0.5)
            >>> oi.macular_transmittance[:] = 1

        """
        # check inputs
        if wave is None:
            wave = np.arange(400.0, 710.0, 10)

        # turn off numpy warning for invalid input
        np.seterr(invalid='ignore')

        # initialize instance attribute to default values
        self.name = "Human Optics"                   # name of the class instance
        self._wave = wave.astype(float)              # wavelength samples in nm
        self.photons = np.array([])                  # irradiance image
        self.pupil_diameter = pupil_diameter         # pupil diameter in meters
        self.dist = dist                             # Object distance in meters
        self.fov = 1.0                               # field of view of the optical image in degree
        self.focal_length = focal_length             # focal lens of optics in meters
        self._otf = None                             # optical transfer function

        # set lens quanta transmittance
        self.lens_transmittance = 10**(-spectra_read("lensDensity.mat", wave))

        # set macular pigment quanta transmittance
        self.macular_transmittance = 10**(-spectra_read("macularPigment.mat", wave))

        # compute human optical transfer function
        # Reference: Marimont & Wandell, J. Opt. Soc. Amer. A,  v. 11, p. 3113-3122 (1994)
        max_freq = 90
        defocus = 1.7312 - (0.63346 / (wave*1e-3 - 0.2141))  # defocus as function of wavelength

        w20 = pupil_diameter**2 / 8 * (1/focal_length * defocus) / (1/focal_length + defocus)  # Hopkins w20 parameter
        sample_sf = np.arange(max_freq)
        achromatic_mtf = 0.3481 + 0.6519 * np.exp(-0.1212 * sample_sf)

        # compute otf at each wavelength
        otf = np.zeros([sample_sf.size, wave.size])
        for ii in range(wave.size):
            s = 1 / tan(deg_to_rad(1)) * wave[ii] * 2e-9 / pupil_diameter * sample_sf  # reduced spatial frequency
            alpha = 4*pi / (wave[ii] * 1e-9) * w20[ii] * s
            otf[:, ii] = self.optics_defocus_mtf(s, alpha) * achromatic_mtf

        # set otf as 2d interpolation function to object
        # To get otf at given wavelength and frequency, use object.otf() method
        self._otf = interp2d(self.wave, sample_sf, otf, bounds_error=False, fill_value=0)

    def compute(self, scene: Scene):
        """ Compute optical irradiance map
        Computation proccedure:
            1) convert radiance to irradiance
            2) apply lens and macular transmittance
            3) apply off-axis fall-off (cos4th)
            4) apply optical transfert function

        Args:
            scene (pyEyeBall.Scene): instance of Scene class, containing the radiance and other scene information

        Examples:
            >>> oi = Optics()
            >>> oi.compute(Scene())
        """
        # set field of view and wavelength samples
        self.fov = scene.fov
        scene.wave = self._wave
        self.dist = scene.dist

        # compute irradiance
        self.photons = pi / (1 + 4 * self.f_number**2 * (1 + abs(self.magnification))**2) * scene.photons

        # apply ocular transmittance
        self.photons *= self.ocular_transmittance

        # apply the relative illuminant (off-axis) fall-off: cos4th function
        x, y = self.spatial_support
        s_factor = np.sqrt(self.image_distance**2 + x**2 + y**2)
        self.photons *= (self.image_distance / s_factor[:, :, None])**4

        # apply optical transfer function of the optics
        for ii in range(self.wave.size):
            otf = fftshift(self.otf(self._wave[ii], self.frequency_support_x, self.frequency_support_y))
            self.photons[:, :, ii] = np.abs(ifftshift(ifft2(otf * fft2(fftshift(self.photons[:, :, ii])))))

    def __str__(self):
        """ Generate description string for optics instance
        This function generates string for Optics class. With the function, optics properties can be printed out
        easily with str(oi)

        Returns:
            str: string of optics description

        Examples:
            >>> print(Optics())
            Human Optics Instance: Human Optics (...and more...)
        """
        s = "Human Optics Instance: " + self.name + "\n"
        s += "  Wavelength: " + str(np.min(self.wave)) + ":" + str(self.bin_width) + ":" + str(np.max(self.wave))
        s += " nm\n"
        s += "  Horizontal field of view: %.4g" % self.fov + " deg\n"
        s += "  Pupil Diameter: %.4g" % (self.pupil_diameter*1000) + " mm\n"
        s += "  Focal Length: %.4g" % (self.focal_length * 1000) + " mm\n"
        if self.photons.size > 0:
            s += "  [Row, Col]: " + str(self.shape) + "\n"
            s += "  [Width, Height]: %.4g" % (self.width*1000) + ", %.4g" % (self.height*1000) + " mm\n"
            s += "  Sample size: %.4g" % self.sample_size + " m\n"
            s += "  Image distance: %.4g" % (self.image_distance*1000) + "mm\n"
        return s

    def plot(self, param, opt=None):
        """Generate plots for optics parameters and properties

        Args:
            param (str): string which indicates the type of plot to generate. In current version, param can be chosen
                from "srgb", "otf", "psf", "lens transmittance", "macular transmittance" and "ocular transmittance".
                param string is not case sensitive and blank spaces in param are ignored.
            opt (float): optional parameters. For case "otf" and "psf", this parameter is required and a floating
                number of desired wavelength should be specified.
        Examples:
            Show otf and psf of the default human optics
            >>> oi = Optics()
            >>> oi.plot("otf", 420)
            >>> oi.plot("psf", 540)
        """
        # process param
        param = str(param).lower().replace(" ", "")

        # generate plot according to param
        if param == "srgb":
            plt.imshow(self.srgb)
        elif param == "otf":
            assert opt is not None, "wavelength to be plotted required as opt"
            freq = np.arange(-90.0, 90.0)
            fx, fy = np.meshgrid(freq, freq)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(fx, fy, self.otf(opt, freq, freq))
            plt.xlabel("Frequency (cycles/deg)")
            plt.ylabel("Frequency (cycles/deg)")
        elif param == "psf":
            assert opt is not None, "Wavelength to be plotted required as opt"
            spatial_support = np.arange(-6e-5, 6e-5, 3e-7)
            sx, sy = np.meshgrid(spatial_support, spatial_support)
            
            psf = self.psf(opt, spatial_support, spatial_support)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(sx * 1e6, sy * 1e6, psf)  # plot in units of um
            plt.xlabel("Position (um)")
            plt.ylabel("Position (um)")
        elif param == "lenstransmittance":
            plt.plot(self._wave, self.lens_transmittance)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Lens Transmittance")
            plt.grid()
        elif param == "maculartransmittance":
            plt.plot(self._wave, self.macular_transmittance)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Macular Pigment Transmittance")
            plt.grid()
        elif param == "oculartransmittance":
            plt.plot(self._wave, self.ocular_transmittance)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Ocular Transmittance")
            plt.grid()
        else:
            raise(ValueError, "Unknown param")
        plt.show()

    def visualize(self):
        """Initialize and show GUI for optics object

        Examples:
            >>> oi = Optics()
            >>> oi.compute(Scene())
            >>> oi.visualize()
        """
        app = QtGui.QApplication([''])
        OpticsGUI(self)
        app.exec_()

    def get_photons(self, wave):
        """numpy.ndarray: get photons at given wavelength samples"""
        f = interp1d(self._wave, self.photons, bounds_error=False, fill_value=0)
        return f(wave)

    @property
    def wave(self):
        """numpy.ndarray: wavelength samples in nm

        When this quantity is set to new values, optics irradiance, lens and macular pigment transmittance data are
        interpolated to match the new wavelength samples
        """
        return self._wave

    @wave.setter
    def wave(self, new_wave):  # adjust wavelength samples
        # check if wavelength samples really changed
        if np.array_equal(self._wave, new_wave):
            return

        # interpolate photons
        if self.photons.size > 0:
            f = interp1d(self._wave, self.photons, bounds_error=False, fill_value=0)
            self.photons = f(new_wave)

        # interpolate lens transmittance
        f = interp1d(self._wave, self.lens_transmittance, bounds_error=False, fill_value=1)
        self.lens_transmittance = f(new_wave)

        # interpolate macular pigment transmittance
        f = interp1d(self._wave, self.macular_transmittance, bounds_error=False, fill_value=1)
        self.macular_transmittance = f(new_wave)

        # set value of _wave in object
        self._wave = new_wave
        raise(Exception, "OTF data not interpolated. Should fix this in the future")

    @property
    def bin_width(self):
        """float: wavelength sample bin width"""
        return self._wave[1] - self._wave[0]

    @property
    def shape(self):
        """numpy.ndarray: number of samples in class in [n_rows, n_cols]"""
        return np.array(self.photons.shape[0:2])

    @property
    def width(self):
        """float: width of optical image in meters"""
        return 2 * self.image_distance * np.tan(deg_to_rad(self.fov)/2)

    @property
    def sample_size(self):
        """float: length per sample in meters"""
        return self.width / self.n_cols

    @property
    def height(self):
        """float: height of optical image in meters"""
        return self.sample_size * self.n_rows

    @property
    def image_distance(self):
        """float: image distance / focal plane distance in meters"""
        return 1/(1/self.focal_length - 1/self.dist)

    @property
    def energy(self):
        """numpy.ndarray: optical image in energy units"""
        return quanta_to_energy(self.photons, self._wave)

    @property
    def magnification(self):
        """float: magnification of the image, usually this is negative for human lens"""
        return -self.image_distance / self.dist

    @property
    def f_number(self):
        """float: f-number for human optics"""
        return self.focal_length/self.pupil_diameter

    @property
    def spatial_support_x(self):
        """numpy.ndarray: spatial support in x direction as 1D array"""
        return np.linspace(-(self.n_cols - 1) * self.sample_size/2, (self.n_cols - 1) * self.sample_size/2, self.n_cols)

    @property
    def spatial_support_y(self):
        """numpy.ndarray: spatial support in y direction as 1D array"""
        return np.linspace(-(self.n_rows - 1) * self.sample_size/2, (self.n_rows - 1) * self.sample_size/2, self.n_rows)

    @property
    def spatial_support(self):
        """numpy.ndarray: spatial support of optical image in meshgrid of (support_x, support_y) in meters"""
        return np.meshgrid(self.spatial_support_x, self.spatial_support_y)

    @property
    def n_rows(self):
        """int: number of rows in the irradiance image"""
        return self.photons.shape[0]

    @property
    def n_cols(self):
        """int: number of columns in the irradiance image"""
        return self.photons.shape[1]

    @property
    def meters_per_degree(self):
        """float: conversion constant between meters and degree"""
        return self.width / self.fov

    @property
    def degrees_per_meter(self):
        """float: conversion constant between degree and meter"""
        return 1/self.meters_per_degree

    @property
    def v_fov(self):
        """float: field of view in vertical direction"""
        return 2*rad_to_deg(atan(self.height/self.image_distance/2))

    @property
    def frequency_support(self):
        """numpy.ndarray: frequency support (meshgrid) of optical image in cycles / degree"""
        return np.meshgrid(self.frequency_support_x, self.frequency_support_y)

    @property
    def frequency_support_x(self):
        """numpy.ndarray: frequency support in x direction in cycles / degree as 1D vector"""
        return np.linspace(-self.n_cols/2/self.fov, self.n_cols/2/self.fov, self.n_cols)

    @property
    def frequency_support_y(self):
        """numpy.ndarray: frequency support in y direction in cycles / degree as 1D vector"""
        return np.linspace(-self.n_rows/2/self.fov, self.n_rows/2/self.fov, self.n_rows)

    @property
    def xyz(self):
        """numpy.ndarray: XYZ image of the optical irradiance map"""
        return xyz_from_energy(self.energy, self.wave)

    @property
    def srgb(self):
        """numpy.ndarray: srgb image of the optical irradiance map"""
        return xyz_to_srgb(self.xyz)

    @property
    def ocular_transmittance(self):
        """numpy.ndarray: ocular transmittance, including lens and macular transmittance"""
        return self.lens_transmittance * self.macular_transmittance

    def otf(self, wave: float, fx=None, fy=None):
        """ Optical transfer function of the optics
        Retrieve optical transfer function of optics at given wavelength and frequency

        Args:
            wave (float): wavelength in nm
            fx (numpy.ndarray): frequency in x direction as 1D vector in cycles/deg, default is np.arange(-90.0, 90.0)
            fy (numpy.ndarray): frequency in y direction as 1D vector in cycles/deg, default is np.arange(-90.0, 90.0)

        Returns:
            numpy.ndarray: 2D otf image at specified wavelength and frequencies

        Examples:
            >>> import numpy as np
            >>> oi=Optics()
            >>> oi.otf(550.0)
            >>> oi.otf(450.0, np.arange(-60.0, 60.0), np.arange(-50.0, 50.0))
        """
        # check inputs
        if fx is None:
            fx = np.arange(-90.0, 90.0)

        if fy is None:
            fy = np.arange(-90.0, 90.0)

        # define frequencies
        fx, fy = np.meshgrid(fx, fy)
        freq = np.sqrt(fx**2 + fy**2).flatten(order="F")

        index = freq.argsort()
        otf = np.zeros(freq.shape)
        otf[index] = self._otf(wave, freq.flatten())[:, 0]
        return otf.reshape(fx.shape, order="F")

    def psf(self, wave, sx=None, sy=None):
        """ Point spread function of optics
        Get point spread function of optics for given wavelength and spatial frequency

        Args:
            wave (float): wavelength sample to be retrieved in nm
            sx (numpy.ndarray): spatial support in x direction in meters, default np.arange(-8e-5, 8e-5, 4e-7)
            sy (numpy.ndarray): spatial support in y direction in meters, default np.arange(-8e-5, 8e-5, 4e-7)

        Returns:
            numpy.ndarray: point spread function

        Note:
          In this function, we assume that sx and sy are equally spaced grid, symmetric about 0

        Examples:
            >>> oi = Optics()
            >>> oi.psf(450.0)
        """
        # check inputs
        if sx is None:
            sx = np.arange(-8e-5, 8e-5, 4e-7)
        if sy is None:
            sy = np.arange(-8e-5, 8e-5, 4e-7)

        # compute spatial spacing and max frequency
        spatial_spacing = np.array([sx[1]-sx[0], sy[1]-sy[0]])
        max_freq = 1/2/spatial_spacing * self.meters_per_degree  # in units cycles/deg

        # get frequency support
        fx = np.linspace(-max_freq[0], max_freq[0], num=sx.size)
        fy = np.linspace(-max_freq[1], max_freq[1], num=sy.size)

        # get otf at wavelength
        otf = self.otf(wave, fx, fy)

        # compute psf
        return np.abs(fftshift(ifft2(otf)))

    @staticmethod
    def optics_defocus_mtf(s, alpha):
        """ Compute diffraction limited mtf without aberrations but with defocus

        Args:
            s (numpy.ndarray): reduced spatial frequency
            alpha (numpy.ndarray): defocus parameter, which is related to w20 of Hopkins

        Returns:
            numpy.ndarray: diffraction limited mtf without aberration but with defocus
        """
        # compute auxiliary parameters
        nf = np.abs(s)/2
        nf[nf > 1] = 1
        beta = np.sqrt(1 - nf**2)
        otf = nf  # allocate space for otf

        # compute perfect spatial frequencies of OTF
        index = (alpha == 0)
        otf[index] = 2/pi * (np.arccos(nf[index]) - nf[index] * beta[index])

        # compute defocused spatial frequencies of OTF
        index = (alpha != 0)
        h1 = beta[index] * jv(1, alpha[index]) + \
            1/2 * np.sin(2*beta[index]) * (jv(1, alpha[index]) - jv(3, alpha[index])) - \
            1/4 * np.sin(4*beta[index]) * (jv(3, alpha[index]) - jv(5, alpha[index]))

        h2 = np.sin(beta[index]) * (jv(0, alpha[index]) - jv(2, alpha[index])) + \
            1/3 * np.sin(3*beta[index]) * (jv(2, alpha[index]) - jv(4, alpha[index])) - \
            1/5 * np.sin(5*beta[index]) * (jv(4, alpha[index]) - jv(6, alpha[index]))

        otf[index] = 4/pi/alpha[index] * np.cos(alpha[index] * nf[index]) * h1 - \
            4/pi/alpha[index] * np.sin(alpha[index]*nf[index]) * h2

        # normalize
        return otf / otf[0]


class OpticsGUI(QtGui.QMainWindow):
    """
    Class for Scene GUI
    """

    def __init__(self, oi: Optics):
        """
        Initialization method for display gui
        :param oi: instance of Optics class
        :return: None, optics gui window will be shown
        """
        super(OpticsGUI, self).__init__()

        self.oi = oi  # save instance of Optics class to this object
        if oi.photons.size == 0:
            raise(Exception, "no data stored in oi")

        # QImage require data to be 32 bit aligned. Thus, need to make sure image size is even
        out_size = (round(oi.n_rows*150/oi.n_cols)*2, 300)
        self.image = imresize(oi.srgb, out_size, interp='nearest')

        # set status bar
        self.statusBar().showMessage("Ready")

        # set menu bar
        menu_bar = self.menuBar()
        menu_file = menu_bar.addMenu("&File")
        menu_plot = menu_bar.addMenu("&Plot")

        # add load optics to file menu
        load_oi = QtGui.QAction("Load Optics", self)
        load_oi.setStatusTip("Load optics from file")
        load_oi.triggered.connect(self.menu_load_oi)
        menu_file.addAction(load_oi)

        # add save optics to file menu
        save_oi = QtGui.QAction("Save Optics", self)
        save_oi.setStatusTip("Save optics to file")
        save_oi.setShortcut("Ctrl+S")
        save_oi.triggered.connect(self.menu_save_oi)
        menu_file.addAction(save_oi)

        # add lens transmittance to plot menu
        plot_lens_transmittance = QtGui.QAction("Lens Transmittance", self)
        plot_lens_transmittance.setStatusTip("Plot lens transmittance (Quanta)")
        plot_lens_transmittance.triggered.connect(lambda: self.oi.plot("lens transmittance"))
        menu_plot.addAction(plot_lens_transmittance)

        # add macular transmittance to plot menu
        plot_macular_transmittance = QtGui.QAction("Macular Transmittance", self)
        plot_macular_transmittance.setStatusTip("Plot macular transmittance (Quanta)")
        plot_macular_transmittance.triggered.connect(lambda: self.oi.plot("macular transmittance"))
        menu_plot.addAction(plot_macular_transmittance)

        # add ocular transmittance to plot menu
        plot_transmittance = QtGui.QAction("Ocular Transmittance", self)
        plot_transmittance.setStatusTip("Plot ocular transmittance (Lens + Macular)")
        plot_transmittance.triggered.connect(lambda: self.oi.plot("ocular transmittance"))
        menu_plot.addAction(plot_transmittance)

        # set up left panel
        left_panel = self.init_image_panel()

        # set up right panel
        right_panel = self.init_control_panel()

        splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        QtGui.QApplication.setStyle(QtGui.QStyleFactory().create('Cleanlooks'))

        widget = QtGui.QWidget()
        hbox = QtGui.QHBoxLayout(widget)
        hbox.addWidget(splitter)

        self.setCentralWidget(widget)

        # set size and put window to center of the screen
        self.resize(600, 400)
        qr = self.frameGeometry()
        qr.moveCenter(QtGui.QDesktopWidget().availableGeometry().center())
        self.move(qr.topLeft())

        # set title and show
        self.setWindowTitle("Optics GUI: " + oi.name)
        self.show()

    def init_image_panel(self):
        """
        Init image panel on the left
        """
        # initialize panel as QFrame
        panel = QtGui.QFrame(self)
        panel.setFrameStyle(QtGui.QFrame.StyledPanel)

        # set components
        vbox = QtGui.QVBoxLayout(panel)

        label = QtGui.QLabel(self)
        q_image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888)
        qp_image = QtGui.QPixmap(q_image)
        label.setPixmap(qp_image)

        vbox.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(label)

        return panel

    def init_control_panel(self):
        """
        Init control panel on the right
        """
        # initialize panel as QFrame
        panel = QtGui.QFrame(self)
        panel.setFrameStyle(QtGui.QFrame.StyledPanel)

        # set components
        vbox = QtGui.QVBoxLayout(panel)
        vbox.setSpacing(15)
        vbox.addWidget(self.init_summary_panel())
        vbox.addWidget(self.init_edit_panel())

        return panel

    def init_summary_panel(self):
        """
        Initialize summary group-box
        """
        # initialize panel as QGroupBox
        panel = QtGui.QGroupBox("Summary")
        vbox = QtGui.QVBoxLayout(panel)

        # set components
        text_edit = QtGui.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QtGui.QTextEdit.NoWrap)
        text_edit.setText(str(self.oi))
        vbox.addWidget(text_edit)

        return panel

    def init_edit_panel(self):
        """
        Initialize edit panel
        """
        # Initialize panel as QGroupBox
        panel = QtGui.QGroupBox("Edit Properties")
        grid = QtGui.QGridLayout(panel)
        grid.setSpacing(10)

        # set components
        fov = QtGui.QLabel("Field of View (deg)")
        distance = QtGui.QLabel("Viewing Distance (m)")

        fov_edit = QtGui.QLineEdit()
        fov_edit.setText("%.4g" % self.oi.fov)
        fov_edit.editingFinished.connect(self.edit_fov)

        distance_edit = QtGui.QLineEdit()
        distance_edit.setText("%.4g" % self.oi.dist)
        distance_edit.editingFinished.connect(self.edit_distance)

        grid.addWidget(fov, 1, 0)
        grid.addWidget(fov_edit, 1, 1)
        grid.addWidget(distance, 2, 0)
        grid.addWidget(distance_edit, 2, 1)

        return panel

    def edit_fov(self):
        self.oi.fov = float(self.sender().text())

    def edit_distance(self):
        self.oi.dist = float(self.sender().text())

    def menu_load_oi(self):
        """
        load scene instance from file
        """
        file_name = QtGui.QFileDialog().getOpenFileName(self, "Choose Optics File", get_data_path(), "*.pkl")
        with open(file_name, "rb") as f:
            self.oi = pickle.load(f)

    def menu_save_oi(self):
        """
        save scene instance to file
        """
        file_name = QtGui.QFileDialog().getSaveFileName(self, "Save Optics to File", get_data_path(), "*.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(self.oi, f, pickle.HIGHEST_PROTOCOL)
