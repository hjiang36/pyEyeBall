__author__ = 'HJ'

from Source.Objects.Scene import Scene
from Source.Objects.Display import Display


def main():
    scene = Scene("macbeth")
    scene.visualize()

    # d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
    # d.visualize()

if __name__ == "__main__":
    main()
