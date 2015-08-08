from unittest import TestCase
from Objects.Optics import Optics
from Objects.Display import Display
from Objects.Scene import Scene
from scipy.ndimage import imread
import os.path
from Utility.IO import get_data_path
import numpy as np

__author__ = 'Killua'


class TestOptics(TestCase):
    def test_constructor(self):
        self.assertRaises(Exception, Optics())

    def test_compute(self):
        d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
        img = imread(os.path.join(get_data_path(), "Image", "eagle.jpg")).astype(float) / 255
        scene = Scene.init_with_display_image(d, img)
        oi = Optics()
        self.assertRaises(Exception, oi.compute(scene))

    def test_plot(self):
        oi = Optics()
        self.assertRaises(Exception, oi.plot("otf", 550))
        self.assertRaises(Exception, oi.plot("psf", 550))

    def test_visualize(self):
        oi = Optics()
        self.assertRaises(Exception, oi.visualize())

    def test_properties(self):
        d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
        img = imread(os.path.join(get_data_path(), "Image", "eagle.jpg")).astype(float) / 255
        scene = Scene.init_with_display_image(d, img)
        oi = Optics()
        oi.compute(scene)
        self.assertRaises(Exception, oi.wave)
        self.assertRaises(Exception, oi.bin_width)
        self.assertRaises(Exception, oi.shape)
        self.assertRaises(Exception, oi.width)
        self.assertRaises(Exception, oi.height)
        self.assertRaises(Exception, oi.sample_size)
        self.assertRaises(Exception, oi.image_distance)
        self.assertRaises(Exception, oi.magnification)
        self.assertRaises(Exception, oi.pupil_diameter)
        self.assertRaises(Exception, oi.spatial_support)
        self.assertRaises(Exception, oi.n_rows)
        self.assertRaises(Exception, oi.n_cols)
        self.assertRaises(Exception, oi.meters_per_degree)
        self.assertRaises(Exception, oi.degrees_per_meter)
        self.assertRaises(Exception, oi.frequency_support)
