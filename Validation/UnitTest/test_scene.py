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

    def test_adjust_illuminant(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        il = Illuminant('D50.mat', wave=scene.wave)
        self.assertRaises(Exception, scene.adjust_illuminant(il))

    def test_adjust_luminance(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.adjust_luminance(100))

    def test_mean_luminance(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.mean_luminance)

    def test_luminance(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.luminance)

    def test_shape(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.shape)

    def test_width(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.width)

    def test_height(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.height)

    def test_sample_size(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.sample_size)

    def test_bin_width(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.bin_width)

    def test_energy(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.energy)

    def test_xyz(self):
        d = Display.init_with_isetbio_mat_file(os.path.join(get_data_path(), 'Display', 'LCD-Apple.mat'))
        img = imread(os.path.join(get_data_path(), 'Image', 'eagle.jpg'), mode='RGB').astype(float)/255
        scene = Scene.init_with_display_image(d, img)
        self.assertRaises(Exception, scene.xyz)
