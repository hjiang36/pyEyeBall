from ..Utility.IO import spectra_read
from ..Utility.Transforms import rad_to_deg, xyz_from_energy, xyz_to_xy, rgb_to_xw_format, xyz_to_srgb
from math import atan2
import numpy as np
from os.path import isfile, join
from os import listdir
from scipy.io import loadmat
from collections import namedtuple
from ..Data.path import get_data_path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PyQt4 import QtGui, QtCore
import pickle
from scipy.misc import imresize, imread

""" Module for display simulation and GUI

This module is used to simulate display performance and compute useful statistics from display image.
There are two classes in this module: Display and DisplayGUI.

Display:
    Display class contains attributes and computational routines for calibrated display. Instance of Display class is
    used together with an RGB image in generating full spectral scene radiance. See init_with_display_image() method in
    Scene class for more details

Display GUI:
    This is a GUI that visualize display properties with PyQt4. In most cases, instance of DisplayGUI class should not
    be created directly. Instead, to show the display GUI for a certain calibrated display d, call d.visualize()

Attributes:
    dixel (collections.namedtuple): the fine structure and layout of one display repeating block (e.g. one pixel).
        Actually, dixel could be declared as a instance-level property of class Display. It is declared as module
        attribute because only in this display instance can be properly saved / load with pickle. dixel contains three
        fields: intensity_map, control_map and n_pixels. dixel.intensity defines the sub-pixel layout and actual
        intensity of primaries at each over-sampled position. dixel.control_map is a 2D array defining which pixel is
        controlling each point. For most display, the repeating pattern is just one pixel and in this case, the control
        map equals zero for all elements. For more complex case, the elements of control map can be chosen from
        range(dixel.n_pixels). dixel.n_pixels represents the number of pixels in one repeating pattern.

Connections with ISETBIO:
    Display class is equivalent to display structure defined in ISETBIO. And display instance can be created directly
    with ISETBIO display calibration files in .mat format.

"""

__author__ = 'HJ'

dixel = namedtuple("dixel", ["intensity_map", "control_map", "n_pixels"])


class Display:
    """ Display simulation, computation and visualization

    The class stores display calibration data and a lot of computed properties and analysis plots can be generated from
    there. At this point, only emissive displays (CRT, LCD, OLED) are supported, though we have a field 'is_emissive'
    left for future extensions.

    Attributes:
        name (str): name of display instance
        gamma (numpy.ndarray): gamma distortion table of the display of shape (n_levels, n_primaries)
        spd (numpy.ndarray): spectral power distribution of the primaries in shape (n_wave, n_primaries). This fields
            represents *mean* spectral power distribution at pixel level and it's not equivalent to the peak spd of the
            materials of primaries. spd are stored in energy units.
        dist (float): distance between display and observer in meters
        dpi (float): display spatial resolution in dots per inch. We assume that spatial resolution along horizontal
            and vertical directions are the same
        is_emissive (bool): is emissive display or reflective display, should always be True for current version
        ambient (numpy.ndarray): spectra power distribution of display at darkest level
        dixel (collections.namedtuple): pixel layout structure, see description in module attribute for more detail

    """

    def __init__(self, name="display", gamma=np.array([]), wave=np.array([]), spd=np.array([]), dist=1.0, dpi=96.0,
                 is_emissive=True, ambient=np.array([]), dixel_data=None):
        """ Initialize display instance with calibrated parameters

        The display attributes are initialized with default values. Those values are usually empty or None, which means
        user have to set up fields manually before using other features and functions.

        Args:
            name (str): name of display instance
            gamma (numpy.ndarray): gamma distortion table of the display of shape (n_levels, n_primaries)
            spd (numpy.ndarray): spectral power distribution of the primaries in shape (n_wave, n_primaries). This
                fields represents *mean* spectral power distribution at pixel level and it's not equivalent to the peak
                spd of the materials of primaries. spd are stored in energy units.
            dist (float): distance between display and observer in meters
            dpi (float): display spatial resolution in dots per inch. We assume that spatial resolution along horizontal
                and vertical directions are the same
            is_emissive (bool): is emissive display or reflective display, should always be True for current version
            ambient (numpy.ndarray): spectra power distribution of display at darkest level
            dixel (collections.namedtuple): pixel layout structure, see description in module attribute for more detail
        """
        # Initialize instance attribute to default values
        self.name = name                # name of display
        self.gamma = gamma              # gamma distortion table
        self._wave = wave               # wavelength samples in nm
        self.spd = spd                  # spectra power distribution
        self.dist = dist                # viewing distance in meters
        self.dpi = dpi                  # spatial resolution in dots per inch
        self.is_emissive = is_emissive  # is emissive display or reflective display
        self.ambient = ambient          # dark emissive spectra distribution
        self.dixel = dixel_data         # pixel layout as a named tuple

    @classmethod
    def init_with_isetbio_mat_file(cls, file_name):
        """ Init display using isetbio display calibration file in .mat format

        Args:
            file_name (str): isetbio display calibration file name. If file is not in Data/Display folder, full path of
                the file is required

        Returns:
            display instance

        Examples:
            Loading a Sony OLED display

            >>> d = Display().init_with_isetbio_mat_file("OLED-Sony.mat")
            >>> print(d.dpi)
            90
        """
        # load data
        if not isfile(file_name):
            file_name = join(get_data_path(), "Display", file_name)
        assert isfile(file_name), "Display calibration file not found"

        tmp = loadmat(file_name)
        tmp = tmp["d"]

        # Init display structure
        d = cls()

        # set to object field
        d.name = tmp["name"][0, 0][0]  # name of display
        d.gamma = tmp["gamma"][0, 0]   # gamma distortion table
        d._wave = np.squeeze(tmp["wave"][0, 0].astype(float))  # wavelength samples in nm
        d.spd = tmp["spd"][0, 0]  # spectral power distribution
        d.dpi = tmp["dpi"][0, 0][0][0]  # spatial resolution in dots / inch
        d.dist = tmp["dist"][0, 0][0][0]  # viewing distance in meters
        d.is_emissive = tmp["isEmissive"][0, 0][0][0]  # is_emissive
        d.ambient = np.squeeze(tmp["ambient"][0, 0].astype(float))

        # Init dixel structure
        intensity_map = tmp["dixel"][0, 0][0]['intensitymap'][0]
        control_map = tmp["dixel"][0, 0][0]['controlmap'][0]
        n_pixels = tmp["dixel"][0, 0][0]['nPixels'][0][0][0]
        d.dixel = dixel(intensity_map, control_map, n_pixels)

        # return
        return d

    def compute(self, img):
        pass

    def visualize(self, img=None):
        """ Initialize and show the GUI for display instance

        Args:
            img (numpy.ndarray): image to be shown in the display gui. Image data should be floating point values
                between 0 and 1. By default, an eagle image will be loaded and used

        Examples:
            An eagle image on an LCD display

            >>> Display().init_with_isetbio_mat_file("LCD-Apple.mat").visualize()

        """
        app = QtGui.QApplication([''])
        DisplayGUI(self, img)
        app.exec_()

    def plot(self, param):
        """ Generate plots for display parameters and properties

        Args:
            param (str): string which indicates the type of plot to generate. In current version, param can be chosen
                from "spd", "gamma", "invert gamma" and "gamut". param string is not case sensitive and blank spaces in
                param are ignored.

        Examples:
            Show spectral power distribution and gamut of a Sony OLED display

            >>> d = Display.init_with_isetbio_mat_file("OLED-Sony.mat")
            >>> d.plot("spd")
            >>> d.plot("gamut")

        """
        # process param to be lowercase and without spaces
        param = str(param).lower().replace(" ", "")

        # making plot according to param
        if param == "spd":  # plot spectra power distribution of display
            plt.plot(self._wave, self.spd)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Energy (watts/sr/m2/nm)")
        elif param == "gamma":  # plot gamma table of display
            plt.plot(self.gamma)
            plt.xlabel("DAC")
            plt.ylabel("Linear")
        elif param == "invertgamma":  # plot invert gamma table of display
            plt.plot(np.linspace(0, 1, self.invert_gamma.shape[0]), self.invert_gamma)
            plt.xlabel("Linear")
            plt.ylabel("DAC")
        elif param == "gamut":  # plot gamut of display
            # plot human visible range
            xyz = spectra_read('XYZ.mat', self._wave)
            xy = xyz_to_xy(xyz, rm_nan=True)
            xy = np.concatenate((xy, xy[0:1, :]), axis=0)
            plt.plot(xy[:, 0], xy[:, 1])

            # plot gamut of display
            xy = xyz_to_xy(self.rgb2xyz, rm_nan=True)
            xy = np.concatenate((xy, xy[0:1, :]), axis=0)
            plt.plot(xy[:, 0], xy[:, 1])
            plt.xlabel("CIE-x")
            plt.ylabel("CIE-y")
        else:
            raise(ValueError, "Unsupported input param")
        plt.grid()
        plt.show()

    @staticmethod
    def ls_display():
        """ List available display calibration file

        This is a static method of Display class and it can be used to retrieve available display data files in
        Data/Display folder

        Returns:
            list: a list of strings of names available display calibration files

        Examples:
            Get a list of calibrated display

            >>> display_names = Display.ls_display()
            >>> print(display_names[0])
            CRT-Dell.mat
        """
        return [f for f in listdir(join(get_data_path(), 'Display'))]

    def __str__(self):
        """ Generate description string of display instance

        This function generates string for Display class. With the function, display properties can be printed out
        easily with str(d)

        Returns:
            str: string of display description

        Examples:
            Description string for an CRT display

            >>> print(Display.init_with_isetbio_mat_file('CRT-Dell.mat'))
            Display Object: CRT-Dell (...and more...)
        """
        s = "Display Object: " + self.name + "\n"
        s += "  Wavelength: " + str(np.min(self._wave)) + ":" + str(self.bin_width) + ":" + str(np.max(self._wave))
        s += " nm\n"
        s += "  Number of primaries: " + str(self.n_primaries) + "\n"
        s += "  Color bit depth: " + str(self.n_bits)
        return s

    def lookup_digital(self, dac):
        """ Convert quantized digital values to linear RGB values through display gamma table

        Args:
            dac (numpy.ndarray): dac values of image in range 0 to n_levels-1 as int

        Returns:
            numpy.ndarray: linear RGB values of image in same shape as dac input

        Examples:
            Compute linear RGB with gamma table

            >>> d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
            >>> print(d.lookup_digital(np.array([[[150, 130, 120]]])))
            [[[ 0.00786591  0.01027275  0.01069388]]]
        """
        dac = dac.astype(int)
        rgb = np.zeros(dac.shape)
        for ii in range(self.n_primaries):
            rgb[:, :, ii] = self.gamma[dac[:, :, ii].astype(int), ii]
        return rgb

    def lookup_linear(self, rgb):
        """ Convert linear RGB values to digital values through invert gamma table

        Args:
            rgb (numpy.ndarray): linear RGB values in range 0 to 1

        Returns:
            numpy.ndarray: quantized digital values converted by invert gamma table as int

        Note:
            The returned dac values are from 0 - n_levels. For some display, the color bit depth is more than 8 and the
            dac values could be larger than 255. See examples below.

        Examples:
            Compute dac value from linear RGB

            >>> d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
            >>> print(d.lookup_linear(np.array([[[0.5, 0.3, 0.2]]])))
            [[[732 567 470]]]
        """
        rgb = ((self.n_levels-1)*rgb).astype(int)
        dac = np.zeros(rgb.shape, int)
        for ii in range(self.n_primaries):
            dac[:, :, ii] = self.invert_gamma[rgb[:, :, ii], ii]
        return dac

    def compute_xyz(self, rgb):
        """ Compute XYZ values of an RGB image on the display instance

        Args:
            rgb (numpy.ndarray): linear RGB image with values between 0-1

        Returns:
            numpy.ndarray: XYZ values in an array of same shape as input rgb image

        Examples:
            >>> d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
            >>> d.compute_xyz(np.array([[[0.5, 0.3, 0.2]]]))
            array([[[ 39.76024121,  39.48678582,  25.55437927]]])
        """
        return np.reshape(np.dot(rgb_to_xw_format(rgb), self.rgb2xyz), rgb.shape, order="F")

    def compute_srgb(self, rgb: np.ndarray):
        """ Compute equivalent sRGB image from an display RGB image

        This function first computes the XYZ image from the display image and then converts the XYZ image to an sRGB
        image. Then, RGB image on display instance should look exactly the same as sRGB image on a *standard* display

        Note:
            The XYZ values of display RGB and sRGB could be slightly different if the color gamut of display instance
            is larger than the *standard* display

            The sRGB color space is defined with a normalized luminance (Y). If the input is only 3 values, the output
            could be close to 1

        Args:
            rgb (numpy.ndarray): display linear RGB image with values between 0 and 1

        Returns:
            numpy.ndarray: sRGB image with same XYZ values as display RGB image

        Examples:
            >>> d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
            >>> d.compute_srgb(np.array([[[0.5, 0.2, 0.1]]]))
            array([[[ 1.        ,  0.91325868,  0.61450081]]])
        """
        return xyz_to_srgb(self.compute_xyz(rgb))

    @property
    def wave(self):
        """numpy.ndarray: wavelength samples in nm

        When new wavelength is set to the display instance, all spectral data (spd, ambient) will be interpolated to
        the new wavelength samples
        """
        return self._wave

    @wave.setter
    def wave(self, value):  # set wavelength samples and interpolate data
        # interpolate spd
        f = interp1d(self._wave, self.spd, axis=0, bounds_error=False, fill_value=0)
        self.spd = f(value)

        # interpolate ambient
        f = interp1d(self._wave, self.ambient, bounds_error=False, fill_value=0)
        self.ambient = f(value)

        # update wavelength sample record in this instance
        self._wave = value

    @property
    def n_bits(self):
        """int: Color bit-depth of the display"""
        return round(np.log2(self.gamma.shape[0]))

    @property
    def n_levels(self):
        """int: number of DAC levels of display, usually it equals to 2^n_bits"""
        return self.gamma.shape[0]

    @property
    def bin_width(self):
        """float: Wavelength sample interval in nm"""
        if self._wave.size > 1:
            return self._wave[1] - self._wave[0]
        else:
            raise(ValueError, "not enough wavelength samples")

    @property
    def n_primaries(self):
        """int: number of primaries of display, usually it equals to 3"""
        return self.spd.shape[1]

    @property
    def invert_gamma(self):
        """numpy.ndarray: Invert gamma table which can be used to convert linear value to DAC value"""
        # set up parameters
        y = range(self.n_levels)
        inv_y = np.linspace(0, 1, self.n_levels)
        lut = np.zeros([self.n_levels, self.n_primaries])

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
        """numpy.ndarray: rgb2xyz transformation matrix"""
        return xyz_from_energy(self.spd.T, self._wave)

    @property
    def rgb2lms(self):
        """numpy.ndarray: rgb2lms transformation matrix"""
        cone_spd = spectra_read("stockman.mat", self._wave)
        return np.dot(self.spd.T, cone_spd)

    @property
    def white_xyz(self):
        """numpy.ndarray: XYZ values of display when all its primaries are set to the highest level"""
        return np.dot(np.array([1, 1, 1]), self.rgb2xyz)

    @property
    def white_lms(self):
        """numpy.ndarray: LMS values of display when all its primaries are set to the highest level"""
        return np.dot(np.array([1, 1, 1]), self.rgb2lms)

    @property
    def meters_per_dot(self):
        """float: size of one display pixel in meters"""
        return 0.0254 / self.dpi

    @property
    def dots_per_meter(self):
        """float: number of pixels per meter"""
        return self.dpi / 0.0254

    @property
    def deg_per_pixel(self):
        """float: visual angle per pixel in degrees"""
        return rad_to_deg(atan2(self.meters_per_dot, self.dist))

    @property
    def white_spd(self):
        """numpy.ndarray: spectra power distribution of display when all its primaries are set to the highest level"""
        return np.sum(self.spd, axis=1)

    @property
    def peak_luminance(self):
        """float: maximum achievable luminance of the display in cd/m2

        When this quantity is set by the new value, the spd of the display is adjusted by a scale factor to match the
        desired peak luminance level
        """
        return self.white_xyz[1]

    @peak_luminance.setter
    def peak_luminance(self, lum):
        self.spd *= lum / self.peak_luminance


class DisplayGUI(QtGui.QMainWindow):
    """
    Class for Display GUI
    """

    def __init__(self, d, image=None):
        """
        Initialization method for display gui
        :param d: instance of display class
        :param image: rgb image in range 0-1 optional
        :return: None, display gui window will be shown
        """
        assert isinstance(d, Display), "d should be an instance of Display class"
        super(DisplayGUI, self).__init__()

        self.d = d  # save instance of Display class to this object
        if image is None:
            fn = join(get_data_path(), 'Image', 'eagle.jpg')
            image = imread(fn).astype(float)/255.0
        out_size = (round(image.shape[0]*250/image.shape[1])*2, 500)
        self.image = imresize(d.compute_srgb(image), out_size, interp='nearest')

        # set status bar
        self.statusBar().showMessage("Ready")

        # set menu bar
        menu_bar = self.menuBar()
        menu_file = menu_bar.addMenu("&File")
        menu_plot = menu_bar.addMenu("&Plot")

        # add load display to file menu
        load_display = QtGui.QAction("Load Display", self)
        load_display.setStatusTip("Load display from file")
        load_display.triggered.connect(self.menu_load_display)
        menu_file.addAction(load_display)

        # add save display to file menu
        save_display = QtGui.QAction("Save Display", self)
        save_display.setStatusTip("Save display to file")
        save_display.setShortcut("Ctrl+S")
        save_display.triggered.connect(self.menu_save_display)
        menu_file.addAction(save_display)

        # add spd to plot menu
        plot_spd = QtGui.QAction("SPD", self)
        plot_spd.setStatusTip("Plot spectra power distribution of the primaries")
        plot_spd.triggered.connect(lambda: self.d.plot("spd"))
        menu_plot.addAction(plot_spd)

        # add gamma to plot menu
        plot_gamma = QtGui.QAction("Gamma Table", self)
        plot_gamma.setStatusTip("Plot gamma distortion of the display")
        plot_gamma.triggered.connect(lambda: self.d.plot("gamma"))
        menu_plot.addAction(plot_gamma)

        # add invert gamma to plot menu
        plot_invert_gamma = QtGui.QAction("Invert Gamma Table", self)
        plot_invert_gamma.setStatusTip("Plot invert gamma distortion of the display")
        plot_invert_gamma.triggered.connect(lambda: self.d.plot("invert gamma"))
        menu_plot.addAction(plot_invert_gamma)

        # add gamut to plot menu
        plot_gamut = QtGui.QAction("Color Gamut", self)
        plot_gamut.setStatusTip("Plot color gamut of display")
        plot_gamut.triggered.connect(lambda: self.d.plot("gamut"))
        menu_plot.addAction(plot_gamut)

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
        self.resize(800, 600)
        qr = self.frameGeometry()
        qr.moveCenter(QtGui.QDesktopWidget().availableGeometry().center())
        self.move(qr.topLeft())

        # set title
        self.setWindowTitle("Display GUI: " + d.name)

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
        vbox.setSpacing(10)
        vbox.addWidget(self.init_summary_panel())
        vbox.addWidget(self.init_edit_panel())
        vbox.addWidget(self.init_pixel_panel())

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
        text_edit.setText(str(self.d))
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
        peak_lum = QtGui.QLabel("Peak Lum (cd/m2)")
        ppi = QtGui.QLabel("Pixel per Inch")

        peak_lum_edit = QtGui.QLineEdit()
        peak_lum_edit.setText("%.4g" % self.d.peak_luminance)
        peak_lum_edit.editingFinished.connect(self.edit_peak_lum)

        ppi_edit = QtGui.QLineEdit()
        ppi_edit.setText("%.4g" % self.d.dpi)
        ppi_edit.editingFinished.connect(self.edit_ppi)

        grid.addWidget(peak_lum, 1, 0)
        grid.addWidget(peak_lum_edit, 1, 1)
        grid.addWidget(ppi, 2, 0)
        grid.addWidget(ppi_edit, 2, 1)

        return panel

    def init_pixel_panel(self):
        """
        Initialize pixel panel
        """
        # Initialize panel as QGroupBox
        panel = QtGui.QGroupBox("Pixel Layout")

        # Set pixel image
        img = self.d.dixel.intensity_map.copy()
        img *= 255.0 / np.max(img)

        out_size = np.array([160, 160, 3])
        img = np.kron(img.astype(np.uint8), np.ones(out_size/np.array(img.shape), np.uint8))
        q_img = QtGui.QImage(img.data, out_size[1], out_size[0], QtGui.QImage.Format_RGB888)
        qp_img = QtGui.QPixmap().fromImage(q_img)

        label = QtGui.QLabel(self)
        label.setPixmap(qp_img)

        vbox = QtGui.QVBoxLayout(panel)
        vbox.setAlignment(QtCore.Qt.AlignHCenter)
        vbox.addWidget(label)

        return panel

    def edit_peak_lum(self):
        self.d.peak_luminance = float(self.sender().text())

    def edit_ppi(self):
        self.d.dpi = float(self.sender().text())

    def menu_load_display(self):
        """
        load display instance from file
        """
        file_name = QtGui.QFileDialog().getOpenFileName(self, "Choose Display File", get_data_path(), "*.pkl")
        with open(file_name, "rb") as f:
            self.d = pickle.load(f)

    def menu_save_display(self):
        """
        save display instance to file
        """
        file_name = QtGui.QFileDialog().getSaveFileName(self, "Save Display to File", get_data_path(), "*.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(self.d, f, pickle.HIGHEST_PROTOCOL)
