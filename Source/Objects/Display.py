from ..Utility.IO import spectra_read
from ..Utility.Transforms import rad_to_deg, xyz_from_energy, xyz_to_xy
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

__author__ = 'HJ'

# define name tuple as global structure so that pickle can use it
dixel = namedtuple("dixel", ["intensity_map", "control_map", "n_pixels"])


class Display:
    """
    Class for display simulation
    """

    def __init__(self):
        """
        Constructor for Display class
        :return:
        """
        # Initialize instance attribute to default values
        self.name = "Display"        # name of display
        self.gamma = np.array([])    # gamma distortion table
        self._wave = np.array([])    # wavelength samples in nm
        self.spd = np.array([])      # spectra power distribution
        self.dist = 0                # viewing distance in meters
        self.dpi = 96                # spatial resolution in dots per inch
        self.is_emissive = True      # is emissive display or reflective display
        self.ambient = np.array([])  # dark emissive spectra distribution
        self.dixel = None            # pixel layout as a named tuple

    @classmethod
    def init_with_isetbio_mat_file(cls, fn):
        """
        init display from isetbio .mat file
        :param fn: isetbio display calibration file with full path
        :return: display object
        """
        # load data
        if not isfile(fn):
            fn = join(get_data_path(), "Display", fn)
        assert isfile(fn), "Display calibration file not found"

        tmp = loadmat(fn)
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

    def visualize(self):
        pass

    def plot(self, param):
        """
        generate plots for display parameters and properties
        :param param: string, indicating which plot to generate, can be chosen from:
                      'spd', 'gamma', 'invert gamma', 'gamut'
        :return: None, but plot will be shown
        """

        # process param to be lowercase and without spaces
        param = str(param).lower().replace(" ", "")
        plt.ion()  # enable interactive mode

        # making plot according to param
        if param == "spd":  # plot spectra power distribution of display
            plt.plot(self._wave, self.spd)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Energy (watts/sr/m2/nm)")
            plt.grid()
            plt.show()
        elif param == "gamma":  # plot gamma table of display
            plt.plot(self.gamma)
            plt.xlabel("DAC")
            plt.ylabel("Linear")
            plt.grid()
            plt.show()
        elif param == "invertgamma":  # plot invert gamma table of display
            plt.plot(np.linspace(0, 1, self.invert_gamma.shape[0]), self.invert_gamma)
            plt.xlabel("Linear")
            plt.ylabel("DAC")
            plt.grid()
            plt.show()
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
            plt.grid()
            plt.show()
        else:
            raise(ValueError, "Unsupported input param")

    @staticmethod
    def ls_display():
        """
        static method that can be used to list all available display calibration file
        :return: a list of available display calibration files
        """
        return [f for f in listdir(join(get_data_path(), 'Display')) if isfile(f)]

    def __str__(self):
        """
        generate verbal string description of display object
        :return: string of display description
        """
        s = "Display Object: " + self.name + "\n"
        s += "\tWavelength: " + str(np.min(self._wave)) + ":" + str(self.bin_width) + ":" + str(np.max(self._wave))
        s += " nm\n"
        s += "\tNumber of primaries: " + str(self.n_primaries) + "\n"
        s += "\tColor bit depth: " + str(self.n_bits)
        return s

    @property
    def wave(self):  # wavelength samples in nm
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
        """
        color bit-depth of the display
        :return: scalar of color bit-depth
        """
        return round(np.log2(self.gamma.shape[0]))

    @property
    def n_levels(self):
        """
        number of DAC levels of display, usually it equals to 2^n_bits
        :return: number of DAC levels of display
        """
        return self.gamma.shape[0]

    @property
    def bin_width(self):
        """
        wavelength sample interval in nm
        :return: wavelength sample interval
        """
        if self._wave.size > 1:
            return self._wave[1] - self._wave[0]
        else:
            raise(ValueError, "not enough wavelength samples")

    @property
    def n_primaries(self):
        """
        number of primaries of display, usually it equals to 3
        :return: number of primaries of display
        """
        return self.spd.shape[1]

    @property
    def invert_gamma(self, n_steps=None):
        """
        Invert gamma table which can be used to convert linear value to DAC value
        :return: Invert gamma table
        """
        # init default value of n_steps
        if n_steps is None:
            n_steps = self.n_levels

        # set up parameters
        y = range(self.n_levels)
        inv_y = np.linspace(0, 1, n_steps)
        lut = np.zeros([n_steps, self.n_primaries])

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
        return xyz_from_energy(self.spd.T, self._wave)

    @property
    def rgb2lms(self):
        cone_spd = spectra_read("stockman.mat", self._wave)
        return np.dot(self.spd.T, cone_spd)

    @property
    def white_xyz(self):
        return np.dot(np.array([1, 1, 1]), self.rgb2xyz)

    @property
    def white_lms(self):
        return np.dot(np.array([1, 1, 1]), self.rgb2lms)

    @property
    def meters_per_dot(self):
        return 0.0254 / self.dpi

    @property
    def dots_per_meter(self):
        return self.dpi / 0.0254

    @property
    def deg_per_pixel(self):
        return rad_to_deg(atan2(self.meters_per_dot, self.dist))

    @property
    def white_spd(self):
        return np.sum(self.spd, axis=1)


class DisplayGUI(QtGui.QMainWindow):
    """
    Class for Display GUI
    """

    def __init__(self, d, img=None):
        """
        Initialization method for display gui
        :param d: instance of display class
        :param img: rgb image, optional
        :return: None, display gui window will be shown
        """
        assert isinstance(d, Display), "d should be an instance of Display class"
        super(DisplayGUI, self).__init__()

        self.d = d  # save instance of Display class to this object

        # set status bar
        self.statusBar().showMessage("Ready")

        # set menu bar
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        menu_file = menu_bar.addMenu("&File")
        menu_edit = menu_bar.addMenu("&Edit")
        menu_plot = menu_bar.addMenu("&Plot")

        # add load display event to file menu
        load_display = QtGui.QAction("Load Display", self)
        load_display.setStatusTip("Load display from file")
        self.connect(load_display, QtCore.SIGNAL('triggered()'), self.menu_load_display)
        menu_file.addAction(load_display)

        # add save display event to file menu
        save_display = QtGui.QAction("Save Display", self)
        save_display.setStatusTip("Save display to file")
        save_display.setShortcut("Ctrl+S")
        self.connect(save_display, QtCore.SIGNAL('triggered()'), self.menu_save_display)
        menu_file.addAction(save_display)

        # add quit to file menu
        quit_gui = QtGui.QAction("Quit", self)
        quit_gui.setStatusTip("Quit Display GUI")
        quit_gui.setShortcut("Ctrl+Q")
        self.connect(quit_gui, QtCore.SIGNAL('triggered()'), QtGui.qApp.quit)
        menu_file.addAction(quit_gui)

        # add spd to plot menu
        plot_spd = QtGui.QAction("SPD", self)
        plot_spd.setStatusTip("Plot spectra power distribution of the primaries")
        self.connect(plot_spd, QtCore.SIGNAL('triggered()'), self.plot_spd)
        menu_plot.addAction(plot_spd)

        # set size and put window to center of the screen
        self.resize(800, 600)
        qr = self.frameGeometry()
        qr.moveCenter(QtGui.QDesktopWidget().availableGeometry().center())
        self.move(qr.topLeft())

        # set title
        self.setWindowTitle("Display GUI: " + d.name)

        self.show()

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

    def plot_spd(self):  # plot spd of the display
        self.d.plot("spd")
