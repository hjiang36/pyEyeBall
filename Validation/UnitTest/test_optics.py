from unittest import TestCase
from Source.Objects.Optics import Optics
from Source.Objects.Scene import Scene

__author__ = 'Killua'


class TestOptics(TestCase):
    def test_constructor(self):
        self.assertRaises(Exception, Optics())

    def test_compute(self):
        scene = Scene("macbeth")
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
        scene = Scene("macbeth")
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
