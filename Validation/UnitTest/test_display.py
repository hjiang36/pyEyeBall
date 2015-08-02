from unittest import TestCase
from Objects.Display import Display
import os.path
from Data.path import get_data_path

__author__ = 'HJ'


class TestDisplay(TestCase):
    def test_init_with_isetbio_mat_file(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        self.assertRaises(Exception, Display.init_with_isetbio_mat_file(fn))

    def test_compute(self):
        self.fail()

    def test_visualize(self):
        self.fail()

    def test_plot(self):
        self.fail()

    def test_ls_display(self):
        self.assertRaises(Exception, Display.ls_display())

    def test_n_bits(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.n_bits)

    def test_n_levels(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.n_levels)

    def test_bin_width(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.bin_width)

    def test_n_primaries(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.n_primaries)

    def test_rgb2xyz(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.rgb2xyz)

    def test_rgb2lms(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.rgb2lms)

    def test_white_xyz(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.white_xyz)

    def test_white_lms(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.white_lms)

    def test_meters_per_dot(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.meters_per_dot)

    def test_dots_per_meter(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.dots_per_meter)

    def test_deg_per_pixel(self):
        fn = os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat')
        d = Display.init_with_isetbio_mat_file(fn)
        self.assertRaises(Exception, d.deg_per_pixel)
