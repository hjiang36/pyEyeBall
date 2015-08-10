from unittest import TestCase
from Source.Objects.Cone import ConePhotopigmentMosaic
from Source.Objects.Scene import Scene
from Source.Objects.Optics import Optics
import numpy as np

__author__ = 'HJ'


class TestConePhotopigmentMosaic(TestCase):
    def test_compute_noisefree(self):
        fov = 1.0  # field of view in degree
        scene = Scene("macbeth", fov=fov)
        oi = Optics().compute(scene)
        cone = ConePhotopigmentMosaic()
        cone.set_fov(new_fov=fov, oi=oi)
        cone.compute_noisefree(oi)
        cone.position = np.round(np.random.normal(size=[100, 2], scale=5))
        cone.compute(oi)

    def test_properties(self):
        fov = 1.0  # field of view in degree
        scene = Scene("macbeth", fov=fov)
        oi = Optics().compute(scene)
        cone = ConePhotopigmentMosaic()
        cone.set_fov(new_fov=fov, oi=oi)
        cone.compute_noisefree(oi)
        self.assertRaises(Exception, cone.wave)
        self.assertRaises(Exception, cone.name)
        self.assertRaises(Exception, cone.bin_width)
        self.assertRaises(Exception, cone.n_cols)
        self.assertRaises(Exception, cone.n_rows)
        self.assertRaises(Exception, cone.size)
        self.assertRaises(Exception, cone.height)
        self.assertRaises(Exception, cone.width)
        self.assertRaises(Exception, cone.cone_area)
        self.assertRaises(Exception, cone.spatial_support)
        self.assertRaises(Exception, cone.degrees_per_cone)
        self.assertRaises(Exception, cone.rgb)
