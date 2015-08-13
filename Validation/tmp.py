__author__ = 'HJ'

from Source.Objects.Scene import Scene
from Source.Objects.Display import Display
from Source.Objects.Optics import Optics
from Source.Objects.Cone import ConePhotopigmentMosaic
import matplotlib.pyplot as plt


def main():

    d = Display.init_with_isetbio_mat_file("OLED-Sony.mat")
    # d.visualize()

    scene = Scene("macbeth")
    # scene.visualize()

    oi = Optics()
    # oi.plot('psf', 550)
    oi.compute(scene)
    # oi.visualize()

    cone = ConePhotopigmentMosaic()
    cone.set_fov(scene.fov, oi)
    cone.compute(oi)
    cone.visualize()

    # plt.show(block=True)

if __name__ == "__main__":
    main()
