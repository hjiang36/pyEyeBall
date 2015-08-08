from unittest import TestCase
from Objects.Scene import Scene
from Objects.Display import Display
from Objects.Illuminant import Illuminant
import os.path
from Utility.IO import get_data_path
from scipy.ndimage import imread

__author__ = 'Killua'


class TestScene(TestCase):

    def test_init_with_display_image(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        self.assertRaises(Exception, Scene.init_with_display_image(d, img))

    def test_properties(self):
        scene = Scene('macbeth')
        il = Illuminant('D50.mat', wave=scene.wave)
        self.assertRaises(Exception, scene.adjust_illuminant(il))
        self.assertRaises(Exception, scene.adjust_luminance(100))
        self.assertRaises(Exception, scene.luminance)
        self.assertRaises(Exception, scene.mean_luminance)
        self.assertRaises(Exception, scene.shape)
        self.assertRaises(Exception, scene.width)
        self.assertRaises(Exception, scene.height)
        self.assertRaises(Exception, scene.sample_size)
        self.assertRaises(Exception, scene.bin_width)
        self.assertRaises(Exception, scene.energy)
        self.assertRaises(Exception, scene.xyz)
        self.assertRaises(Exception, scene.srgb)
