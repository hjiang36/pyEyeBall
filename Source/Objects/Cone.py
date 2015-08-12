import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.stats import rv_discrete
from ..Utility.IO import spectra_read
from ..Data.path import get_data_path
from .Optics import Optics
from ..Utility.Transforms import rad_to_deg, rgb_to_xw_format
import matplotlib.pyplot as plt
import copy
import pickle
from PyQt4 import QtGui, QtCore
from scipy.misc import imresize


__author__ = 'HJ'


class ConePhotopigment:
    """
    Class describing cone photopigment properties
    Programming Note:
    In this class, we only care about three types of cones (LMS) and in ConePhotopigmentMosaic class, the order of cone
    type is K,L,M,S. Thus, need to pad a column for K in that class
    """

    def __init__(self, wave=np.array(range(400, 710, 10)),
                 name="Human Cone Photopigment", absorbance=None,
                 optical_density=np.array([.5, .5, .4]), peak_efficiency=np.array([2, 2, 2])/3):
        """
        Constructor for ConePhotopigment class
        :param wave: sample wavelength in nm
        :param name: name of the instance of this class
        :param absorbance: absorbance of L,M,S cones
        :param optical_density: optical density of three types of cones
        :param peak_efficiency: peak efficiency of three types of cones
        :return: instance of ConePhotopigment class with attribute set properly
        """
        # Initialize attributes
        self._wave = wave
        self.name = name
        self.optical_density = optical_density
        self.peak_efficiency = peak_efficiency

        # set / load absorbance
        if absorbance is not None:
            self.absorbance = absorbance
        else:
            self.absorbance = 10.0**spectra_read("coneAbsorbance.mat", wave)

    @property
    def wave(self):  # wavelength samples in nm
        return self._wave

    @wave.setter  # set wavelength samples and interpolate data
    def wave(self, value):
        if not np.array_equal(self._wave, value):
            f = interp1d(self._wave, self.absorbance, bounds_error=False, fill_value=0)
            self.absorbance = f(value)
            self._wave = value

    @property
    def absorptance(self):  # cone absorptance without ocular media
        return 1 - 10.0**(-self.optical_density*self.absorbance)

    @property
    def quanta_efficiency(self):  # quanta efficiency of cones
        return self.absorptance * self.peak_efficiency

    @property
    def quanta_fundamentals(self):  # quantal fundamentals of cones
        qe = self.quanta_efficiency
        return qe / np.max(qe, axis=0)


class ConePhotopigmentMosaic:
    """
    Class describing human cone mosaic and isomerizations
    """
    
    def __init__(self, wave=np.array(range(400, 710, 10)), name="Human Cone Mosaic",
                 mosaic=None, cone_width=2e-6, cone_height=2e-6,
                 density=np.array([.0, .6, .3, .1]), position=np.array([[0, 0]]),
                 integration_time=0.05, size=np.array([72, 88])):
        """
        Constructor for class
        :param wave: wavelength sample of this class
        :param name: name of the instance of this class
        :param mosaic: 2D matrix, indicating cone type at each position, 0~3 represents K,L,M,S respectively
        :param cone_width: width of the cone in meters
        :param cone_height: height of the cone in meters
        :param density: spatial density (proportional) of different cone types in order of K,L,M,S
        :param position: N-by-2 matrix indicating eye movement positions in (x, y) in units of number of cones
        :param integration_time: integration time of cone in secs
        :param size: size of cone mosaic to be generated, only used when mosaic is not given
        :return: instance of class with attributes set
        """
        # Initialize instance attribute
        self.name = name
        self._wave = wave
        self.cone_width = cone_width
        self.cone_height = cone_height
        self.density = density / np.sum(density)
        self.position = position
        self.integration_time = integration_time

        # Initialize spectral quanta efficiency of the cones
        self.quanta_efficiency = np.concatenate((np.zeros([wave.size, 1]),
                                                ConePhotopigment(wave=wave).quanta_efficiency), axis=1)

        # Initialize photons (isomerization) as empty array
        self.photons = np.array([])  # cone isomerization rate

        # Generate cone mosaic if not given
        if mosaic is not None:
            self.mosaic = mosaic
        else:
            self.mosaic = rv_discrete(values=(range(4), self.density)).rvs(size=size)

    def __str__(self):
        """
        Generate descriptive string for class instance
        :return: descriptive string
        """
        s = "Human Cone Mosaic: " + self.name + "\n"
        s += "  [Height, Width]: [%.4g" % (self.height*1000) + ", %.4g" % (self.width*1000) + "] mm\n"
        s += "  Field of view: %.4g" % self.fov + " deg\n"
        s += "  Cone size: (%.4g" % (self.cone_height*1e6) + ", %.4g" % (self.cone_width*1e6) + ") um\n"
        s += "  Integration time: %.4g" % (self.integration_time*1000) + " ms\n"
        s += "  Spatial density (K,L,M,S):" + str(self.density)
        return s

    def plot(self, param):
        """
        generate plots for class attributes and properties
        :param param: string, indicating what should be plotted
        :return: None, but plot will be shown
        """
        # process input param
        param = str(param).lower().replace(" ", "")
        plt.ion()

        # generate plot
        if param == "rgb":  # rgb visualization of the cone mosaic
            plt.imshow(self.rgb)
        elif param == "quantaefficiency":  # quanta efficiency of the cones
            plt.plot(self.wave, self.quanta_efficiency)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Quanta Efficiency")
            plt.grid()
        elif param == "mosaic":  # cone mosaic
            # define color for L, M, S
            color = np.array([[228.0, 26.0, 28.0], [77.0, 175.0, 74.0], [55.0, 126.0, 184.0]])/255
            rgb = np.zeros([self.n_rows, self.n_cols, 3])
            for cone_type in range(1, 4):
                rgb += (self.mosaic == cone_type)[:, :, None] * color[cone_type-1, :]
            plt.imshow(rgb)
        elif param == "eyemovent" or param == "positions":
            plt.plot(self.position_x, self.position_y)
            plt.xlabel("Eye position (number of cones)")
            plt.ylabel("Eye position (number of cones)")
            plt.grid()
        else:
            raise(ValueError, "Unknown param")

    def visualize(self):
        app = QtGui.QApplication([''])
        ConeGUI(self)
        app.exec_()

    def compute_noisefree(self, oi, full_lms=False):
        """
        Compute cone photon absorptions without adding any noise
        :param oi: instance of Optics class with irradiance image computed
        :param full_lms: bool, indicating if photons should be 2D computed according to mosaic or 3D as full lms image
        :return: instance of class with noise free cone absorptions stored in attribute photons

        Programming note:
        This function does not care about eye movement (self.position) and thus the
        """
        # check inputs
        assert isinstance(oi, Optics), "oi should be instance of class Optics"

        # allocate space for photons
        if full_lms:
            self.photons = np.zeros([self.n_rows, self.n_cols, 3])
        else:
            self.photons = np.zeros(self.size)

        # compute signal current density in amp/m2
        photons = rgb_to_xw_format(oi.get_photons(self._wave))
        signal_current = np.dot(photons, self.quanta_efficiency) * self.bin_width
        signal_current = np.reshape(signal_current, (oi.n_rows, oi.n_cols, 4), order="F")

        # compute cone absorptions
        oi_sx = oi.spatial_support_x
        oi_sy = oi.spatial_support_y
        cone_sx = self.spatial_support_x
        cone_sy = self.spatial_support_y
        for cone_type in range(1, 4):
            # interpolate for current cone type
            f = interp2d(oi_sx, oi_sy, signal_current[:, :, cone_type], bounds_error=False, fill_value=0)
            cone_photons = f(cone_sx, cone_sy).reshape(self.size, order="F") * self.cone_area * self.integration_time

            # store photons
            if full_lms:
                self.photons[:, :, cone_type-1] = cone_photons
            else:
                # select data according to cone mosaic
                self.photons += cone_photons * (self.mosaic == cone_type)
        return self

    def compute(self, oi: Optics, add_noise=True):
        """
        Compute cone photon absorption with noise and eye movement
        :param oi: instance of Optics class with irradiance computed
        :param add_noise: bool, indicate whether or not to add photon shot noise
        :return: instance of this class with photons computed and stored
        """
        # allocate space for photons
        self.photons = np.zeros([self.n_rows, self.n_cols, self.n_positions])

        # increase cone mosaic size according to eye movement
        pos_x = self.position_x
        pos_y = self.position_y
        rows_to_pad = np.array([-min(np.min(pos_y), 0), max(np.max(pos_y), 0)])
        cols_to_pad = np.array([-min(np.min(pos_x), 0), max(np.max(pos_x), 0)])
        size = self.size + np.array([np.sum(rows_to_pad), np.sum(cols_to_pad)])  # increased cone mosaic size

        cone_mosaic = copy.deepcopy(self)  # make a copy
        cone_mosaic.size = size

        # compute noise-free absorptions
        cone_mosaic.compute_noisefree(oi, full_lms=True)

        # convert cone mosaic to binary mask representation
        mask = np.zeros([self.n_rows, self.n_cols, 3])
        for ii in range(3):
            mask[:, :, ii] = (self.mosaic == ii+1)

        # pick cone positions according to eye movement
        for pos in range(self.n_positions):
            photons = cone_mosaic.photons[(rows_to_pad[0]+pos_y[pos]):(rows_to_pad[0]+pos_y[pos]+self.n_rows),
                                          (cols_to_pad[0]+pos_x[pos]):(cols_to_pad[0]+pos_x[pos]+self.n_cols), :]
            self.photons = np.sum(photons * mask, axis=2)

        # add noise
        if add_noise:
            self.photons = np.random.poisson(lam=self.photons).astype(float)
        return self

    @property
    def wave(self):
        return self._wave

    @wave.setter
    def wave(self, new_wave):  # adjust wavelength samples
        if not np.array_equal(self.wave, new_wave):
            # interpolate spectral quanta efficiency
            f = interp1d(self._wave, self.quanta_efficiency, axis=0, bounds_error=False, fill_value=0)
            self.quanta_efficiency = f(new_wave)

    @property
    def bin_width(self):  # wavelength sample interval in nm
        return self._wave[1] - self._wave[0]

    @property
    def n_rows(self):  # number of rows of cones
        return self.mosaic.shape[0]

    @property
    def n_cols(self):  # number of columns of cones
        return self.mosaic.shape[1]

    @property
    def size(self):  # shape of the cone mosaic in (n_rows, n_cols)
        return np.array(self.mosaic.shape)

    @size.setter
    def size(self, new_size):  # re-generate cone mosaic according to new size
        self.mosaic = rv_discrete(values=(range(4), self.density)).rvs(size=new_size)
        self.photons = np.array([])

    @property
    def height(self):  # height of the cone mosaic in meters
        return self.cone_height * self.n_rows

    @property
    def width(self):  # width of the cone mosaic in meters
        return self.cone_width * self.n_cols

    @property
    def cone_area(self):  # area of one cone in m2
        return self.cone_height * self.cone_width

    @property
    def spatial_support_x(self):  # x position of each cone in meters (1D array)
        return np.linspace(-(self.n_cols-1)*self.cone_width/2, (self.n_cols-1)*self.cone_width/2, self.n_cols)

    @property
    def spatial_support_y(self):  # y position of each cone in meters (1D array)
        return np.linspace(-(self.n_rows-1)*self.cone_height/2, (self.n_rows-1)*self.cone_height/2, self.n_rows)

    @property
    def spatial_support(self):  # position of each cone in meters (2D array)
        return np.meshgrid(self.spatial_support_x, self.spatial_support_y)

    @property
    def degrees_per_cone(self):  # cone width in degree
        return self.fov/self.n_cols

    @property
    def position_x(self):  # eye movement position in x direction
        return self.position[:, 0]

    @property
    def position_y(self):  # eye movement position in y direction
        return self.position[:, 1]

    @property
    def n_positions(self):  # number of eye movement positions, same position is counted multiple times
        return self.position.shape[0]

    @property
    def rgb(self):  # rgb image of the cone mosaic photon absorptions
        # define color for L, M, S
        color = np.array([[228.0, 26.0, 28.0], [77.0, 175.0, 74.0], [55.0, 126.0, 184.0]])/255

        # Normalize photon absorptions
        if self.photons.ndim == 3:
            photons = self.photons[:, :, 0]
        else:
            photons = self.photons
        photons /= np.max(photons)

        # allocate space for rgb image
        rgb = np.zeros([self.n_rows, self.n_cols, 3])
        for cone_type in range(1, 4):
            rgb += (photons * (self.mosaic == cone_type))[:, :, None] * color[cone_type-1, :]
        return rgb

    def get_fov(self, oi=Optics()):  # sensor field of view in degree
        return 2 * rad_to_deg(np.arctan(self.width/oi.image_distance/2))

    def set_fov(self, new_fov, oi=Optics()):  # adjust sensor size according to field of view
        self.size = np.round(new_fov*self.size/self.get_fov(oi))

    fov = property(get_fov, set_fov)


class ConeGUI(QtGui.QMainWindow):
    """
    Class for Scene GUI
    """

    def __init__(self, cone: ConePhotopigmentMosaic):
        """
        Initialization method for display gui
        :param cone: instance of ConePhotopigmentMosaic class
        :return: None, cone gui window will be shown
        """
        super(ConeGUI, self).__init__()

        self.cone = cone  # save instance of ConePhotopigmentMosaic class to this object
        if cone.photons.size == 0:
            raise(Exception, "no data stored in cone")

        # QImage require data to be 32-bit aligned. Thus, we need to make sure image rows/cols is even
        rows = round(cone.n_rows * 150 / cone.n_cols) * 2
        self.image = imresize(cone.rgb, (rows, 300), interp='nearest')

        # set status bar
        self.statusBar().showMessage("Ready")

        # set menu bar
        menu_bar = self.menuBar()
        menu_file = menu_bar.addMenu("&File")
        menu_plot = menu_bar.addMenu("&Plot")

        # add load optics to file menu
        load_cone = QtGui.QAction("Load Cone Mosaic", self)
        load_cone.setStatusTip("Load cone mosaic from file")
        load_cone.triggered.connect(self.menu_load_cone)
        menu_file.addAction(load_cone)

        # add save optics to file menu
        save_cone = QtGui.QAction("Save Cone Mosaic", self)
        save_cone.setStatusTip("Save cone mosaic to file")
        save_cone.setShortcut("Ctrl+S")
        save_cone.triggered.connect(self.menu_save_cone)
        menu_file.addAction(save_cone)

        # add cone quanta efficiency to plot menu
        plot_cone_quanta_efficiency = QtGui.QAction("Quanta Efficiency", self)
        plot_cone_quanta_efficiency.setStatusTip("Plot cone quanta efficiency")
        plot_cone_quanta_efficiency.triggered.connect(lambda: self.cone.plot("quanta efficiency"))
        menu_plot.addAction(plot_cone_quanta_efficiency)

        # add eye movement to plot menu
        plot_eye_path = QtGui.QAction("Eyemovment Path", self)
        plot_eye_path.setStatusTip("Plot eye movement path")
        plot_eye_path.triggered.connect(lambda: self.cone.plot("eyemovement"))
        menu_plot.addAction(plot_eye_path)

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
        self.setWindowTitle("Cone GUI: " + cone.name)
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
        text_edit.setText(str(self.cone))
        vbox.addWidget(text_edit)

        return panel

    def menu_load_cone(self):
        """
        load scene instance from file
        """
        file_name = QtGui.QFileDialog().getOpenFileName(self, "Choose Cone Mosaic File", get_data_path(), "*.pkl")
        with open(file_name, "rb") as f:
            self.cone = pickle.load(f)

    def menu_save_cone(self):
        """
        save scene instance to file
        """
        file_name = QtGui.QFileDialog().getSaveFileName(self, "Save Cone Mosaic to File", get_data_path(), "*.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(self.cone, f, pickle.HIGHEST_PROTOCOL)