from unittest import TestCase
from Objects.Display import Display
import os.path
from Data.path import get_data_path
from scipy.ndimage import imread

__author__ = 'HJ'


class TestDisplay(TestCase):
    def test_init_with_isetbio_mat_file(self):
        self.assertRaises(Exception, Display.init_with_isetbio_mat_file("LCD-Apple.mat"))

    def test_compute(self):
        d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
        fn = os.path.join(get_data_path(), 'Image', 'eagle.jpg')
        img = imread(fn, mode='RGB').astype(float)/255
        self.assertRaises(Exception, d.compute(img))

    def test_plot(self):
        d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
        self.assertRaises(Exception, d.plot("gamma"))
        self.assertRaises(Exception, d.plot("spd"))
        self.assertRaises(Exception, d.plot("invert gamma"))
        self.assertRaises(Exception, d.plot("gamut"))

    def test_ls_display(self):
        self.assertRaises(Exception, Display.ls_display())

    def test_properties(self):
        d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
        self.assertRaises(Exception, d.n_bits)
        self.assertRaises(Exception, d.n_levels)
        self.assertRaises(Exception, d.bin_width)
        self.assertRaises(Exception, d.n_primaries)
        self.assertRaises(Exception, d.rgb2xyz)
        self.assertRaises(Exception, d.rgb2lms)
        self.assertRaises(Exception, d.white_xyz)
        self.assertRaises(Exception, d.white_lms)
        self.assertRaises(Exception, d.meters_per_dot)
        self.assertRaises(Exception, d.dots_per_meter)
        self.assertRaises(Exception, d.deg_per_pixel)
        self.assertRaises(Exception, d.invert_gamma)
        self.assertRaises(Exception, d.white_spd)
