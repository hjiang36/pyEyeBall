__author__ = 'HJ'

from Source.Objects.Scene import Scene
from Source.Objects.Display import Display
from Source.Objects.Optics import Optics
from Source.Objects.Cone import ConeOuterSegmentMosaic
import matplotlib.pyplot as plt


def main():

    # tmp = Display.ls_display()
    # d = Display.init_with_isetbio_mat_file("OLED-Sony.mat")
    # d.visualize()

    scene = Scene("macbeth")
    scene.mean_luminance = 100
    # scene.visualize()

    oi = Optics()
    # oi.plot('psf', 550)
    oi.compute(scene)
    # oi.visualize()

    cone = ConeOuterSegmentMosaic()
    cone.set_fov(scene.fov, oi)
    cone.init_eye_movement(n_samples=200)
    # cone.plot('eyemovement')
    cone.compute(oi)
    print(cone.current)
    cone.visualize()

    # plt.show(block=True)

if __name__ == "__main__":
    main()
