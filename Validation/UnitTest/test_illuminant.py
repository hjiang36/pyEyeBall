from unittest import TestCase
from Objects.Illuminant import Illuminant
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

    def test_energy(self):
        il = Illuminant("D65.mat")
        self.assertRaises(Exception, il.energy)

    def test_luminance(self):
        il = Illuminant("D65.mat")
        self.assertRaises(Exception, il.luminance)

    def test_wave_interpolation(self):
        il = Illuminant("D65.mat")
        il.wave = np.array(range(400, 710, 10))
        self.assertTrue(il.photons.size == 31, "wavelength interpolation failed")
