from unittest import TestCase
from ...Objects.Illuminant import Illuminant
import numpy as np

__author__ = 'Killua'


class TestIlluminant(TestCase):
    def test_constructor(self):
        self.assertRaises(Exception, Illuminant())
        self.assertRaises(Exception, Illuminant("D50.mat"))
        self.assertRaises(Exception, Illuminant("D65.mat"))
        self.assertRaises(Exception, Illuminant("D75.mat"))
        self.assertRaises(Exception, Illuminant("Fluorescent.mat"))
        self.assertRaises(Exception, Illuminant("Tungsten.mat"))

    def test_properties(self):
        il = Illuminant("D65.mat")
        self.assertRaises(Exception, il.energy)
        self.assertRaises(Exception, il.luminance)

        il.wave = np.array(range(400, 710, 10))
        self.assertTrue(il.photons.size == 31, "wavelength interpolation failed")

    def test_plot(self):
        il = Illuminant("D65.mat")
        self.assertRaises(Exception, il.plot("energy"))
        self.assertRaises(Exception, il.plot("photons"))
