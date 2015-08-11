__author__ = 'HJ'

from Source.Objects.Display import Display, DisplayGUI
from PyQt4 import QtGui
import sys


def main():
    app = QtGui.QApplication(sys.argv)
    d = Display.init_with_isetbio_mat_file("LCD-Apple.mat")
    dg = DisplayGUI(d)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
