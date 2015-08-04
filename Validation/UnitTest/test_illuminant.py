from unittest import TestCase
from Objects.Illuminant import Illuminant

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
