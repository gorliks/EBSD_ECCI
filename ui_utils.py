import qtdesigner_files.display_window as new_window

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (QGridLayout, QLabel, QMessageBox, QSizePolicy,
                             QVBoxLayout, QWidget)
from PyQt5.QtWidgets import *
from matplotlib.figure import Figure
import sip


import numpy as np

import matplotlib.pyplot as plt
from dataclasses import dataclass

import h5py
import hyperspy.api as hs
import kikuchipy as kp

import utils


def display_image(main_gui, image=None):
    #image = plt.imread(image) #skimage.color.gray2rgb(plt.imread(image))
    window = GUIUserWindow(parent=main_gui, display_image=image)
    return window



class GUIUserWindow(new_window.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, parent=None, display_image=None):
        super(GUIUserWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setup_connections()

        self.setWindowTitle("Test window")
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        # show text
        self.label_message.setText('plot EBSD pattern')
        self.label_image.setText("")

        # show image
        self.image = display_image
        self.wp = _WidgetPlot(self, display_image=self.image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)

        # draw crosshair
        # draw_crosshair(display_image, self.wp.canvas)

        self.show()

    def setup_connections(self):
        self.pushButton_apply_clahe.clicked.connect(lambda: self.apply_clahe())
        self.pushButton_restore.clicked.connect(lambda: self.restore())

    def apply_clahe(self):
        self.label_message.setText('apply clahe')
        clipLimit = int(self.spinBox_clip_limit.value())
        tileGridSize = int(self.spinBox_tile_grid_size.value())
        image_mod = utils.enhance_contrast(self.image,
                                           clipLimit=clipLimit,
                                           tileGridSize=tileGridSize)
        # delete the previous widget
        self.label_image.layout().removeWidget(self.wp)
        sip.delete(self.wp)
        self.wp = None
        # make a new widget
        self.wp = _WidgetPlot(self, display_image=image_mod)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)
        self.show()
        self.label_message.setText('clahe applied')

    def restore(self):
        self.label_message.setText('restoring')
        # delete the previous widget
        self.label_image.layout().removeWidget(self.wp)
        sip.delete(self.wp)
        self.wp = None
        # make a new widget
        self.wp = _WidgetPlot(self, display_image=self.image)
        self.label_image.setLayout(QtWidgets.QVBoxLayout())
        self.label_image.layout().addWidget(self.wp)
        self.show()
        self.label_message.setText('original image restored')






class _WidgetPlot(QWidget):
    def __init__(self, *args, display_image, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = _PlotCanvas(self, image=display_image)
        self.layout().addWidget(self.canvas)


class _PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, image=None):
        self.fig = Figure()
        FigureCanvasQTAgg.__init__(self, self.fig)

        # clicked point coordinates initialise
        self.x = None
        self.y = None

        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvasQTAgg.updateGeometry(self)
        self.image = image
        self.plot()
        self.createConn()

        self.figureActive = False
        self.axesActive = None
        self.cursorGUI = "arrow"
        self.cursorChanged = False

    def plot(self):
        gs0 = self.fig.add_gridspec(1, 1)

        self.ax11 = self.fig.add_subplot(gs0[0], xticks=[], yticks=[], title="")

        if self.image.ndim == 3:
            self.ax11.imshow(self.image,)
        else:
            self.ax11.imshow(self.image, cmap="gray")

    def updateCanvas(self, event=None):
        ax11_xlim = self.ax11.get_xlim()
        ax11_xvis = ax11_xlim[1] - ax11_xlim[0]

        while len(self.ax11.patches) > 0:
            [p.remove() for p in self.ax11.patches]
        while len(self.ax11.texts) > 0:
            [t.remove() for t in self.ax11.texts]

        ax11_units = ax11_xvis * 0.003
        self.fig.canvas.draw()

    def createConn(self):
        self.fig.canvas.mpl_connect("figure_enter_event", self.activeFigure)
        self.fig.canvas.mpl_connect("figure_leave_event", self.leftFigure)
        self.fig.canvas.mpl_connect("button_press_event", self.mouseClicked)
        self.ax11.callbacks.connect("xlim_changed", self.updateCanvas)

    def activeFigure(self, event):
        self.figureActive = True

    def leftFigure(self, event):
        self.figureActive = False
        if self.cursorGUI != "arrow":
            self.cursorGUI = "arrow"
            self.cursorChanged = True

    def mouseClicked(self, event):
        self.x = event.xdata
        self.y = event.ydata
        print(self.x, self.y)





@dataclass
class Crosshair:
    rectangle_horizontal: plt.Rectangle
    rectangle_vertical: plt.Rectangle


def create_crosshair(
    image: np.ndarray or AdornedImage, x=None, y=None, colour="xkcd:yellow"):
    try:
        image = image.data
    except:
        image = image

    midx = int(image.shape[1] / 2) if x is None else x
    midy = int(image.shape[0] / 2) if y is None else y

    cross_width = int(0.05 / 100 * image.shape[1])
    cross_length = int(5 / 100 * image.shape[1])

    rect_horizontal = plt.Rectangle(
        (midx - cross_length / 2, midy - cross_width / 2), cross_length, cross_width
    )
    rect_vertical = plt.Rectangle(
        (midx - cross_width, midy - cross_length / 2), cross_width * 2, cross_length
    )

    # set colours
    rect_horizontal.set_color(colour)
    rect_vertical.set_color(colour)

    return Crosshair(
        rectangle_horizontal=rect_horizontal, rectangle_vertical=rect_vertical
    )

# TODO update with Point
def draw_crosshair(image, canvas, x: float = None, y: float = None, colour: str ="yellow"):
    # draw crosshairs
    crosshair = create_crosshair(image, x, y, colour=colour)
    for patch in crosshair.__dataclass_fields__:
        canvas.ax11.add_patch(getattr(crosshair, patch))
        getattr(crosshair, patch).set_visible(True)