from .Illuminant import Illuminant
from math import tan, atan2
from .Display import Display
from ..Utility.Transforms import *
from scipy.interpolate import interp1d
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
from PyQt4 import QtGui, QtCore
from ..Data.path import get_data_path
import pickle

""" Module for scene radiance simulation and GUI

This module is used to characterize scene spectral radiance image and compute useful statistics from scene data.
There are two classes in this module: Scene and SceneGUI.

Scene:
    Scene class contains attributes and computational routines for full spectra 2D image. In current version, there is
    no 3D support (no depth information). And HJ personally thinks there will not be 3D scene support in the near
    future.

Scene GUI:
    This is a GUI that visualize scene properties with PyQt4. In most cases, instance of SceneGUI class should not
    be created directly. Instead, to show the GUI for a certain scene, call scene.visualize()

Connections with ISETBIO:
    Scene class is equivalent to scene structure defined in ISETBIO. There are more default scenes available in ISETBIO.
    And there is some potential of 3D support in ISETBIO in the future (see Scene 3D project)

"""

__author__ = 'HJ'


class Scene:
    """ Scene spectral radiance characterization

    The scene class stores illuminant data (see Illuminant class), scene spectra radiance and some other fundamental
    properties. And a lot more, such as reflectance, are implemented as computed properties. There are two different
    ways to create a scene: 1) create preset scenes by scene name and 2) define scene as an image on a calibrated
    display.

    Attributes:
        name (str): name of the Scene instance
        photons (numpy.ndarray): scene spectra radiance data
        fov (float): horizontal field of view of the scene in degrees
        dist (float): the distance between the scene and the observer
        illuminant (pyEyeBall.Illuminant): illumination properties

    Note:
        wavelength samples are not directly stored as fundamental scene attribute. There is a computed property called
        wave in the scene instance and it's tightly linked with wavelength samples in underlying illuminant instance.
        Thus, wavelength samples can still be get and set with scene.wave. But there will be no redundancies and
        inconsistencies between the scene and illuminant wavelength.
    """

    def __init__(self, scene_type="macbeth", wave=None, name="Scene",
                 il=None, fov=1.0, dist=1.0, **kwargs):
        """Constructor for Scene class
        Initialize preset scene instance by scene name

        Args:
            scene_type (str): type of the scene. In current version, scnene_type can be chosen from 'macbeth' (default)
                and 'noise'
            wave (numpy.ndarray): wavelength samples in nm
            name (str): name of this scene instance
            il (pyEyeBall.Illuminant): Illuminant instance. D65 illuminant light is applied by default
            fov (float): scene field of view in degrees
            dist (float): distance between scene plane and observer in meters

        Examples:
            Create scene of macbeth color checker and noise
            >>> scene = Scene("macbeth", patch_size=32)
            >>> scene = Scene("noise", scene_sz=[256, 256])
        """
        # check inputs
        if il is None:
            il = Illuminant()
        assert isinstance(il, Illuminant), "il should be Illuminant instance"

        # interpolate for wavelength samples if specified
        if wave is not None:
            il.wave = wave

        # Initialize instance attribute to default values
        self.name = name                # name of the object
        self.photons = np.array([])     # scene photons
        self.fov = fov                  # horizontal field of view of the scene in degree
        self.dist = dist                # viewing distance in meters
        self.illuminant = il            # illuminant

        # switch by scene_type
        scene_type = str(scene_type).lower().replace(" ", "")
        if scene_type == "macbeth":  # macbeth color checker
            if "patch_size" in kwargs:
                patch_size = kwargs["patch_size"]
            else:
                patch_size = 16
            if np.isscalar(patch_size):
                patch_size = [patch_size, patch_size]
            self.name = "Macbeth Color Checker"
            # load surface reflectance
            surface = np.reshape(spectra_read("macbethChart.mat", self.wave).T, (4, 6, self.wave.size), order="F")
            # compute photons
            self.photons = np.zeros((4*patch_size[0], 6*patch_size[1], self.wave.size))
            for ii in range(self.wave.size):
                self.photons[:, :, ii] = np.kron(surface[:, :, ii], np.ones((patch_size[0], patch_size[1])))
            # multiply by illuminant
            self.photons *= il.photons

        elif scene_type == "noise":  # white noise pattern
            # get scene size
            if "scene_size" in kwargs:
                scene_size = kwargs["scene_sz"]
            else:
                scene_size = np.array([128, 128])
            self.name = "White noise"
            # generate noise pattern and compute photons
            noise_img = np.random.rand(scene_size[0], scene_size[1])
            self.photons = noise_img[:, :, None] * il.photons
        else:
            raise(ValueError, 'Unsupported scene type')

    @classmethod
    def init_with_display_image(cls, d, image, is_sub=False):
        """Initialize scene spectra radiance with an image on a calibrated display
        Compute the full radiance of scene of an RGB image on a calibrated display

        Args:
            d (pyEyeBall.Display): instance of Display class
            image (numpy.ndarray): rgb image to be shown on the display, rgb should be in range of [0, 1]
            is_sub (bool): flag indicating whether or not to do sub-pixel rendering (coming soon)

        Examples:
            create a scene with an eagle image on OLED display (suppose image is loaded)
            >>> from scipy.misc import imread
            >>> from os.path import join
            >>> image = imread(join(get_data_path(), 'Image', 'eagle.jpg')).astype('float')/255
            >>> d = Display.init_with_isetbio_mat_file("OLED-Sony.mat")
            >>> scene = Scene.init_with_display_image(d, image)
        """
        # init basic display parameters
        scene = cls()
        scene.illuminant = Illuminant(wave=d.wave)
        scene.dist = d.dist  # viewing distance

        # compute horizontal field of view
        scene.fov = 2 * rad_to_deg(atan2(image.shape[1] * d.meters_per_dot / 2, d.dist))

        # set illuminant as spd of display
        scene.illuminant.photons = energy_to_quanta(d.white_spd, d.wave)

        # gamma distortion for the input image
        image = d.lookup_digital((image * (d.n_levels - 1)).astype(int))

        # sub-pixel rendering if required
        if is_sub:
            image = d.compute(image)

        # compute radiance from image
        out_sz = np.concatenate((np.array(image.shape[0:2]), [d.wave.size]))
        image = rgb_to_xw_format(image)
        scene.photons = energy_to_quanta(np.dot(image, d.spd.T), d.wave)

        # add ambient quanta to scene photons
        scene.photons += d.ambient

        # reshape photons
        scene.photons = scene.photons.reshape(out_sz, order="F")
        return scene

    def adjust_illuminant(self, il):
        """Adjust illuminant of the scene
        Adjust illuminant of the scene, assuming that the reflectance is unchanged. Thus, when illuminant changed,
        scene radiance (photons) field will be changed accordingly.


        Args:
            il (pyEyeBall.Illuminant): instance of Illuminant class

        Note:
            1) The sampling wavelength of new wavelength will be adjusted to be same as scene wavelength samples.
            2) The mean luminance of the scene will be kept unchanged.

        Examples:
            >>> scene = Scene()
            >>> scene.adjust_illuminant(Illuminant("D50.mat"))
        """
        # store mean luminance
        mean_lum = self.mean_luminance

        # adjust il wavelength
        il.wave = self.wave

        # update photons
        self.photons *= il.photons / self.illuminant.photons

        # update illuminant spd
        self.illuminant = il

        # make sure mean luminance is not changed
        self.mean_luminance = mean_lum

    def __str__(self):
        """Generate description string for scene instance
        This function generates string for Scene class. With the function, scene properties can be printed out
        easily with str(scene)

        Returns:
            str: string of scene description

        Examples:
            >>> print(Scene())
            Scene Object: Macbeth Color Checker (...and more...)
        """
        s = "Scene Object: " + self.name + "\n"
        s += "  Wavelength: " + str(np.min(self.wave)) + ":" + str(self.bin_width) + ":" + str(np.max(self.wave))
        s += " nm\n"
        s += "  [Row, Col]: " + str(self.shape) + "\n"
        s += "  [Width, Height]: [" + "%.2g" % self.width + ", " + "%.2g" % self.height + "] m\n"
        s += "  Horizontal field of view: " + "%.2g" % self.fov + " deg\n"
        s += "  Sample size: " + "%.2g" % self.sample_size + " meters/sample\n"
        s += "  Mean luminance: " + "%.2g" % self.mean_luminance + " cd/m2"
        return s

    @property
    def wave(self):
        """numpy.ndarray: wavelength samples in nm

        When this quantity is set to new values, illuminant data and spectra radiance data will be interpolated to
        match the new wavelength samples
        """
        return self.illuminant.wave

    @wave.setter
    def wave(self, value):  # set wavelength samples and interpolate data
        # update photons
        f = interp1d(self.wave, self.photons, bounds_error=False, fill_value=0)
        self.photons = f(value)

        # update illuminant
        self.illuminant.wave = value

    @property
    def mean_luminance(self):
        """float: mean luminance of scene in cd/m2

        When this quantity is set to new values, spectra radiance data will be scaled to match the new value
        """
        return np.mean(self.luminance)

    @mean_luminance.setter
    def mean_luminance(self, value):  # adjust mean luminance of the scene
        self.photons /= self.mean_luminance / value

    @property
    def luminance(self):
        """numpy.ndarray: luminance image of the scene"""
        return luminance_from_energy(self.energy, self.wave)

    @property
    def shape(self):
        """ tuple: shape of scene in (rows, cols)"""
        return self.photons.shape[0:2]

    @property
    def n_cols(self):
        """float: number of columns"""
        return self.photons.shape[1]

    @property
    def n_rows(self):
        """float: number of rows"""
        return self.photons.shape[0]

    @property
    def width(self):
        """float: width of scene in meters"""
        return 2 * self.dist * tan(deg_to_rad(self.fov))

    @property
    def height(self):
        """float: height of scene in meters"""
        return self.sample_size * self.shape[0]

    @property
    def sample_size(self):
        """float: size of each sample in meters"""
        return self.width / self.shape[1]

    @property
    def bin_width(self):
        """float: wavelength sample interval in nm"""
        return self.wave[1] - self.wave[0]

    @property
    def energy(self):
        """numpy.ndarray: scene radiance in energy units"""
        return quanta_to_energy(self.photons, self.wave)

    @property
    def xyz(self):
        """numpy.ndarray: XYZ image of the scene"""
        return xyz_from_energy(self.energy, self.wave)

    @property
    def srgb(self):
        """numpy.ndarray: equivalent srgb image of the scene"""
        return xyz_to_srgb(self.xyz)

    def plot(self, param):
        """Generate plots for scene parameters and properties

        Args:
            param (str): string which indicates the type of plot to generate. In current version, param can be chosen
                from "illuminant energy", "illuminant photons", "srgb". param string is not case sensitive and blank
                spaces in param are ignored.

        Examples:
            Show illuminant energy and srgb image of the scene
            >>> scene = Scene()
            >>> scene.plot("illuminant energy")
            >>> scene.plot("srgb")
        """
        # process param to be lowercase and without spaces
        param = str(param).lower().replace(" ", "")
        plt.ion()  # enable interactive mode

        # making plot according to param
        if param == "illuminantenergy":  # energy of illuminant
            self.illuminant.plot("energy")
        elif param == "illuminantphotons":  # photons of illuminant
            self.illuminant.plot("photons")
        elif param == "srgb":  # srgb image of the scene
            plt.imshow(self.srgb)
            plt.show()
        else:
            raise(ValueError, "Unknown parameter")

    def visualize(self):
        """Initialize and show GUI for scene object

        Examples:
            >>> scene = Scene()
            >>> scene.visualize()
        """
        app = QtGui.QApplication([''])
        SceneGUI(self)
        app.exec_()


class SceneGUI(QtGui.QMainWindow):
    """
    Class for Scene GUI
    """

    def __init__(self, scene: Scene):
        """
        Initialization method for display gui
        :param scene: instance of Scene class
        :return: None, scene gui window will be shown
        """
        super(SceneGUI, self).__init__()

        self.scene = scene  # save instance of Scene class to this object
        if scene.photons.size == 0:
            raise(Exception, "no data stored in scene")

        # QImage require data to be 32 bit aligned. Thus, we need to make sure out_size is even
        out_size = (round(scene.n_rows * 150/scene.n_cols)*2, 300)
        self.image = imresize(scene.srgb, out_size, interp='nearest')

        # set status bar
        self.statusBar().showMessage("Ready")

        # set menu bar
        menu_bar = self.menuBar()
        menu_file = menu_bar.addMenu("&File")
        menu_plot = menu_bar.addMenu("&Plot")

        # add load scene to file menu
        load_scene = QtGui.QAction("Load Scene", self)
        load_scene.setStatusTip("Load scene from file")
        load_scene.triggered.connect(self.menu_load_scene)
        menu_file.addAction(load_scene)

        # add save scene to file menu
        save_scene = QtGui.QAction("Save Scene", self)
        save_scene.setStatusTip("Save scene to file")
        save_scene.setShortcut("Ctrl+S")
        save_scene.triggered.connect(self.menu_save_scene)
        menu_file.addAction(save_scene)

        # add illuminant energy to plot menu
        plot_il_energy = QtGui.QAction("Illuminant (Energy)", self)
        plot_il_energy.setStatusTip("Plot spectra power distribution of scene illuminant")
        plot_il_energy.triggered.connect(lambda: self.scene.plot("illuminant energy"))
        menu_plot.addAction(plot_il_energy)

        # add illuminant photons to plot menu
        plot_il_quanta = QtGui.QAction("Illuminant (Photons)", self)
        plot_il_quanta.setStatusTip("Plot spectra power distribution of scene illuminant")
        plot_il_quanta.triggered.connect(lambda: self.scene.plot("illuminant photons"))
        menu_plot.addAction(plot_il_quanta)

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
        self.setWindowTitle("Scene GUI: " + scene.name)
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
        text_edit.setText(str(self.scene))
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
        mean_lum = QtGui.QLabel("Mean Lum (cd/m2)")
        fov = QtGui.QLabel("Field of View (deg)")
        distance = QtGui.QLabel("Viewing Distance (m)")

        mean_lum_edit = QtGui.QLineEdit()
        mean_lum_edit.setText("%.4g" % self.scene.mean_luminance)
        mean_lum_edit.editingFinished.connect(self.edit_mean_lum)

        fov_edit = QtGui.QLineEdit()
        fov_edit.setText("%.4g" % self.scene.fov)
        fov_edit.editingFinished.connect(self.edit_fov)

        distance_edit = QtGui.QLineEdit()
        distance_edit.setText("%.4g" % self.scene.dist)
        distance_edit.editingFinished.connect(self.edit_distance)

        grid.addWidget(mean_lum, 1, 0)
        grid.addWidget(mean_lum_edit, 1, 1)
        grid.addWidget(fov, 2, 0)
        grid.addWidget(fov_edit, 2, 1)
        grid.addWidget(distance, 3, 0)
        grid.addWidget(distance_edit, 3, 1)

        return panel

    def edit_mean_lum(self):
        self.scene.mean_luminance = float(self.sender().text())

    def edit_fov(self):
        self.scene.fov = float(self.sender().text())

    def edit_distance(self):
        self.scene.dist = float(self.sender().text())

    def menu_load_scene(self):
        """
        load scene instance from file
        """
        file_name = QtGui.QFileDialog().getOpenFileName(self, "Choose Scene File", get_data_path(), "*.pkl")
        with open(file_name, "rb") as f:
            self.scene = pickle.load(f)

    def menu_save_scene(self):
        """
        save scene instance to file
        """
        file_name = QtGui.QFileDialog().getSaveFileName(self, "Save Scene to File", get_data_path(), "*.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(self.scene, f, pickle.HIGHEST_PROTOCOL)

