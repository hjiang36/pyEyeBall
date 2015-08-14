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
from scipy.constants import pi


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


class FixationalEyeMovement:
    """
    Class describes fixational eye movement
    """

    def __init__(self, name='Human Eye Movement', flag=np.array([True, True, True]), tremor_interval=0.012,
                 tremor_interval_sd=0.001, tremor_amplitude=0.05, drift_speed=0.05, drift_speed_sd=0.016,
                 msaccade_interval=0.6, msaccade_interval_sd=0.3, msaccade_direction_sd=5,
                 msaccade_speed=15, msaccade_speed_sd=5
                 ):
        """
        Constructor for fixational eye movement class
        :param name: string, name of the instance of this class
        :param flag: np.ndarray of three values, indicating whether to include tremor, drift, micro-saccade respectively
        :param tremor_interval: float, time interval between two tremors in seconds
        :param tremor_interval_sd: float, standard deviation of tremor_interval
        :param tremor_amplitude: amplitude of tremor in degree
        :param drift_speed: float, speed of drift in degree / sec
        :param drift_speed_sd: float, standard deviation of drift
        :param msaccade_interval: float, time interval between two micro-saccade in seconds
        :param msaccade_interval_sd: float, standard deviation of micro-saccade interval
        :param msaccade_direction_sd: float, deviation of direction of movement towards fixation point
        :param msaccade_speed: float, micro-saccade speed in degree/second
        :param msaccade_speed_sd: float, standard deviation of micro-saccade speed

        Reference:
        1) Susana Martinez-Conde et. al, The role of fixational eye movements in visual perception, Nature reviews |
           neuroscience, Vol. 5, 2004, page 229~240
        2) Susana Martinez-Conde et. al, Microsaccades: a neurophysiological analysis, Trends in Neurosciences,
           Volume 32, Issue 9, September 2009, Pages 463~475
        """
        # set parameters
        self.name = name
        self.flag = flag
        self.tremor_interval = tremor_interval
        self.tremor_interval_sd = tremor_interval_sd
        self.tremor_amplitude = tremor_amplitude
        self.drift_speed = drift_speed
        self.drift_speed_sd = drift_speed_sd
        self.msaccade_interval = msaccade_interval
        self.msaccade_interval_sd = msaccade_interval_sd
        self.msaccade_direction_sd = msaccade_direction_sd
        self.msaccade_speed = msaccade_speed
        self.msaccade_speed_sd = msaccade_speed_sd

    def __str__(self):
        """
        Generate description string for this class
        """
        s = 'Fixational Eye Movement: ' + self.name + '\n'
        s += '  flag: ' + str(self.flag) + '\n'
        s += '  Tremor parameters: \n'
        s += '    interval: %.4g' % (self.tremor_interval*1000) + ' ms\n'
        s += '    interval sd: %.4g' % (self.tremor_interval_sd * 1000) + ' ms\n'
        s += '    amplitude: %.4g' % (self.tremor_amplitude * 3600) + ' arcsec\n'
        s += '  Drift parameters: \n'
        s += '    speed: %.4g' % self.drift_speed + ' deg/sec\n'
        s += '    speed sd: %.4g' % self.drift_speed_sd + 'deg/sec\n'
        s += '  Micro-saccade parameters: \n'
        s += '    interval: %.4g' % (self.msaccade_interval * 1000) + 'ms\n'
        s += '    interval sd: %.4g' % (self.msaccade_interval_sd * 1000) + 'ms\n'
        s += '    speed: %.4g' % self.msaccade_speed + ' deg/sec\n'
        s += '    speed sd: %.4g' % self.msaccade_speed_sd + ' deg/sec\n'
        s += '    direction sd: %.4g' % self.msaccade_direction_sd + ' deg'
        return s

    def generate_path(self, sample_time=0.001, n_samples=1000, start=np.array([[0, 0]])):
        """
        Generate fixational eye movement path
        :param sample_time: float, time between samples in sec
        :param n_samples: int, number of samples to generate
        :param start: initial position at time 0
        :return: np.ndarray, n_samples x 2 array representing the x, y position at each time sample in degrees

        Programming Note:
          We will first generate eye movement path at temporal resolution of 1 ms and then interpolate to the desired
          sample_time.
        """
        # Initialize position, allocate space
        n_pos = round(n_samples*sample_time/0.001)
        position = np.zeros([n_pos, 2]) + start  # position at every 1 ms

        # Generate eye movement for tremor
        if self.flag[0]:
            # compute when tremor occurs
            t = self.tremor_interval + self.tremor_interval_sd * np.random.randn(n_pos)
            t[t < 0.001] = 0.001  # get rid of negative values
            t_pos = np.round(np.cumsum(t) / 0.001)
            t_pos = t_pos[0:np.argmax(t_pos >= n_pos)].astype(int)

            # randomize direction and compute direction
            direction = 2 * np.random.rand(t_pos.size, 2) - 1
            position[t_pos, :] = direction / np.sqrt(np.sum(direction**2, axis=1))[:, None] * self.tremor_amplitude
            position = np.cumsum(position, axis=0)

        # Generate eye movement for drift
        if self.flag[1]:
            # set direction to be gradually changed over time
            theta = pi*np.random.rand() + 0.1 * pi/180 * np.arange(n_pos)
            direction = np.zeros([n_pos, 2])
            direction[:, 0] = np.cos(theta)
            direction[:, 1] = np.sin(theta)

            # generate random moves
            s = self.drift_speed + self.drift_speed_sd * np.random.randn(n_pos, 1)
            position += direction * s * sample_time

        # Generate eye movement for micro-saccade
        if self.flag[2]:
            # compute when micro-saccade occur
            t = self.msaccade_interval + self.msaccade_interval_sd * np.random.randn(n_pos)
            t[t < 0.3] = 0.3 + 0.1 * np.random.rand(np.sum(t < 0.3))  # get rid of negative value
            t_pos = np.round(np.cumsum(t)/0.001)
            t_pos = t_pos[0:np.argmax(t_pos >= n_pos)].astype(int)

            for cur_t_pos in t_pos:
                cur_pos = position[cur_t_pos, :]
                duration = round(np.sqrt(np.sum(cur_pos**2)) / self.msaccade_speed / 0.001)

                direction = np.arctan2(cur_pos[1], cur_pos[0]) + self.msaccade_direction_sd * pi/180 * np.random.randn()
                direction = np.array([np.cos(direction), np.sin(direction)])
                direction = np.abs(direction) * (2*(cur_pos < 0) - 1)

                offset = np.zeros([n_pos, 2])
                cur_speed = self.msaccade_speed + self.msaccade_speed_sd * np.random.randn()
                if cur_speed < 0:
                    cur_speed = self.msaccade_speed
                offset[cur_t_pos:min(cur_t_pos + duration, n_pos-1), 0] = cur_speed*direction[0] * sample_time
                offset[cur_t_pos:min(cur_t_pos + duration, n_pos-1), 1] = cur_speed*direction[1] * sample_time

                position += np.cumsum(offset, axis=0)

        # Interpolate to desired sample time
        if sample_time != 0.001:
            f = interp1d(np.arange(n_pos), position, axis=0)
            position = f(np.arange(0, n_samples*sample_time, sample_time))
        return position


class ConeOuterSegmentMosaic:
    """
    Class describing human cone mosaic and isomerization
    """
    
    def __init__(self, wave=np.array(range(400, 710, 10)), name="Human Cone Mosaic",
                 mosaic=None, cone_width=2e-6, cone_height=2e-6,
                 density=np.array([.0, .6, .3, .1]), position=np.array([[0, 0]]),
                 integration_time=0.05, sample_time=0.001, size=np.array([72, 88])):
        """
        Constructor for class
        :param wave: wavelength sample of this class
        :param name: name of the instance of this class
        :param mosaic: 2D matrix, indicating cone type at each position, 0~3 represents K,L,M,S respectively
        :param cone_width: float, width of the cone in meters
        :param cone_height: float, height of the cone in meters
        :param density: spatial density (proportional) of different cone types in order of K,L,M,S
        :param position: N-by-2 matrix indicating eye movement positions in (x, y) in units of number of cones
        :param integration_time: float, integration time of cone in secs
        :param sample_time: float, sample time interval for self.position in secs
        :param size: size of cone mosaic to be generated, only used when mosaic is not given
        :return: instance of class with attributes set

        Todo:
          1. Cone adaptation and cone current
          2. Replace cone_width and cone_height with cone_diameter
          3. Have spatial support and cone diameter vary with eccentricty
        """
        # Initialize instance attribute
        self.name = name
        self._wave = wave
        self.cone_width = cone_width
        self.cone_height = cone_height
        self.density = density / np.sum(density)
        self.position = position
        self.integration_time = integration_time
        self.sample_time = sample_time

        # Initialize spectral quanta efficiency of the cones
        # pad a column for black holes (K)
        self.quanta_efficiency = np.pad(ConePhotopigment(wave=wave).quanta_efficiency,
                                        ((0, 0), (1, 0)), mode='constant', constant_values=0)

        # Initialize photons (isomerization) as empty array
        self.photons = np.array([])

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
        elif param == "eyemovement" or param == "positions":
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

    def init_eye_movement(self, n_samples=1000, em=FixationalEyeMovement()):
        """
        convert eye path to positions in cone mosaic (self.position)
        :param n_samples: int, number of samples of eye positions
        :param em: instance of FixationalEyeMovement class
        :return: None, but self.position will be set
        """
        position = em.generate_path(self.sample_time, n_samples)  # position in degrees
        self.position = np.round(position * self.cones_per_degree)

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
            self.photons[:, :, pos] = np.sum(photons * mask, axis=2)

        # add noise
        if add_noise:
            self.photons = np.random.poisson(lam=self.photons).astype(float)

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
    def current_noisefree(self):  # current without cone noise
        """
        Get noise free cone membrane current with a temporal dynamic model by Fred Rieke
        We assume eye is adapted to the first frame as the initial steady state
        """
        # check if photon absorptions have been computed
        assert self.photons.size > 0, "photon absorptions not computed"

        # init parameters
        sigma = 22.0     # rhodopsin activity decay rate (1/sec)
        phi = 22.0       # phosphodiesterase activity decay rate (1/sec)
        eta = 2000	     # phosphodiesterase activation rate constant (1/sec)
        gdark = 20.5     # concentration of cGMP in darkness
        k = 0.02         # constant relating cGMP to current
        h = 3            # cooperativity for cGMP to current
        cdark = 1        # dark calcium concentration
        beta = 9	     # rate constant for calcium removal in 1/sec
        beta_slow = 0.4  # rate constant for slow calcium modulation of channels
        n = 4   	     # cooperativity for cyclase, hill coef
        k_gc = 0.5        # hill affinity for cyclase
        opsin_gain = 10  # rate of increase in opsin activity per R*/sec
        dt = 0.001       # time interval in differential equation simulation

        # Compute more parameters - steady state constraints among parameters
        q = 2 * beta * cdark / (k * gdark**h)
        smax = eta/phi * gdark * (1 + (cdark / k_gc)**n)

        # pre-pad photon absorption by the first frame
        n_pad = 500  # pad 500 frames before first frame for steady state adaptation
        p_rate = self.photons / self.integration_time  # photons per second
        p_rate = np.concatenate((np.tile(p_rate[:, :, :1], (1, 1, n_pad)), p_rate), axis=2)

        # set initial state
        opsin = 560
        pde = 110
        ca = ca_slow = 1
        c_gmp = 20

        # compute membrane current by simulating differential equations
        current = np.zeros(p_rate.shape)
        for ii in range(p_rate.shape[2]):
            opsin += dt * (opsin_gain * p_rate[:, :, ii] - sigma * opsin)
            pde += dt * (opsin + eta - phi * pde)
            ca += dt * (q*k * c_gmp**h/(1 + ca_slow/cdark)-beta*ca)
            ca_slow -= dt*beta_slow * (ca_slow - ca)
            st = smax / (1 + (ca/k_gc)**n)
            c_gmp += dt * (st - pde * c_gmp)
            current[:, :, ii] = -k * c_gmp**h / (1 + ca_slow/cdark)
        return current[:, :, n_pad:]

    @property
    def current(self):  # get cone current
        # model noise with spd
        noise = np.random.randn(self.n_rows, self.n_cols, self.n_positions)
        noise_fft = np.fft.fft(noise) * np.sqrt(self.cone_noise_spd)
        return self.current_noisefree + np.real(np.fft.ifft(noise_fft))

    @property
    def cone_noise_spd(self):  # spd of cone noise
        k = round((self.n_positions-1)/2)+1
        freq = np.arange(k) / self.sample_time / self.n_positions

        # compute noise spd as sum of Lorentzian component
        noise_spd = 0.48/(1+(freq/55)**2)**4 + 0.135/(1+(freq/190)**2)**2.5

        # make up the negative components
        return np.concatenate((noise_spd, noise_spd[::-1]))[:self.n_positions]

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
    def cones_per_degree(self):  # number of cones per degree
        return self.n_cols / self.fov

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

    def __init__(self, cone: ConeOuterSegmentMosaic):
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

