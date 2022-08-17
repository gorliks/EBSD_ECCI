import scipy
import scipy.ndimage as ndi
import skimage.draw
from scipy import fftpack, misc

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

import os, time
import numpy as np
import skimage.util
import skimage
import skimage.io
import skimage.transform
import skimage.color

# sub-pixel cross-correrlation precision
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from skimage.transform import AffineTransform










def open_correlation_window(main_gui, image_1=None, image_2=None, output_path=None):
    """Opens a new window to perform correlation
    Parameters
    ----------
    main_gui : PyQt5 Window
    image1 : numpy.array with shape: (rows, columns) or path
        to numpy.array with shape: (rows, columns)
    image2 : numpy.array with shape: (rows, columns) or path
        to numpy.array with shape: (rows, columns)
    output_path : path to save location
    """
    global image1
    global image2
    global image1_path
    global image2_path
    global gui
    global output

    gui = main_gui
    output = output_path

    if type(image_1) == str:
        image1 = skimage.color.gray2rgb(plt.imread(image_1))

    if type(image_2) == str:
        image2 = skimage.color.gray2rgb(plt.imread(image_2))

    image1 = skimage.transform.resize(image1, image2.shape)

    window = _CorrelationWindow(parent=gui)
    return window



class _CorrelationWindow(QMainWindow):
    """Main correlation window"""
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.create_window()
        self.create_connections()

        self.wp.canvas.fig.subplots_adjust(
            left=0.01, bottom=0.01, right=0.99, top=0.99) # widget

        q1 = QTimer(self)
        q1.setSingleShot(False)
        q1.timeout.connect(self.updateGUI)
        q1.start(10000)

    def create_window(self):
        self.setWindowTitle("Control Point Selection Tool")

        widget = QWidget(self)

        self.setCentralWidget(widget)

        hlay = QHBoxLayout(widget)
        vlay = QVBoxLayout()
        vlay2 = QVBoxLayout()
        vlay2.setSpacing(20)
        hlay_buttons = QHBoxLayout()

        hlay.addLayout(vlay)
        hlay.addLayout(vlay2)

        self.wp = _WidgetPlot(self)
        vlay.addWidget(self.wp)

        self.help = QTextEdit()
        self.help.setReadOnly(True)
        self.help.setMaximumWidth(400)
        self.help.setMinimumHeight(540)

        help_header = '<!DOCTYPE html><html lang="de" ' \
                      'id="main"><head><meta charset="UTF-8"><title>' \
                      'Description of cpselect for Python</title><style>td,' \
                      'th{font-size:14px;}p{font-size: 14px;}</style></head>'
        help_body = '<body><h1>Description of cpselect for Python&emsp;' \
                    '</h1><h2>Navigation Toolbar</h2><img src="{}" ' \
                    'alt="navbuttons"><br/><table cellspacing="20px"><tr>' \
                    '<th valign="middle" height="20px">Tool</th><th valign=' \
                    '"middle" height="20px">how to use</th></tr><tr><td>' \
                    '<img src="{}" alt="homebutton"></td>' \
                    '<td valign="middle">For all Images, reset ' \
                    'to the original view.</td></tr><tr><td>' \
                    '<img src="{}" alt="backwardforwardbutton">' \
                    '</td><td valign="middle">Go back to the last ' \
                    'or forward to the next view.</td></tr><tr><td>' \
                    '<img src="{}" alt="panzoombutton"></td>' \
                    '<td valign="middle">Activate the pan/zoom tool. ' \
                    'Pan with left mouse button, zoom with right</td></tr>' \
                    '<tr><td><img src="{}" alt="backwardforwardbutton">' \
                    '</td><td valign="middle">Zoom with drawing a rectangle' \
                    '</td></tr></table><h2>Pick Mode</h2><p>' \
                    'Change into pick mode to pick up your control points. ' \
                    'You have to pick the control points in both images ' \
                    'before you can start to pick the next point.</p><p>' \
                    'Press the red button below to start pick mode.</p><h2>' \
                    'Control Point list</h2><p>Below in the table, all ' \
                    'your control points are listed. You can delete one or ' \
                    'more selected control points with the <b>delete</b> ' \
                    'button.</p><h2>Return</h2><p>If you are finished, ' \
                    'please press the <b>return</b> button below. You will' \
                    ' come back to wherever you have been.</p></body></html>'
        help_html = help_header + help_body.format(
            os.path.join(os.path.dirname(__file__), "img/navbuttons.PNG"),
            os.path.join(os.path.dirname(__file__), "img/homebutton.png"),
            os.path.join(os.path.dirname(__file__),
                         "img/backforwardbutton.png"),
            os.path.join(os.path.dirname(__file__), "img/panzoombutton.png"),
            os.path.join(os.path.dirname(__file__), "img/zoomboxbutton.png"),
        )
        self.help.insertHtml(help_html)
        self.cpTabelModel = QStandardItemModel(self)
        self.cpTable = QTableView(self)
        self.cpTable.setModel(self.cpTabelModel)
        self.cpTable.setMaximumWidth(400)

        self.delButton = QPushButton("Delete selected Control Point")
        self.delButton.setStyleSheet("font-size: 16px")

        self.pickButton = QPushButton("pick mode")
        self.pickButton.setFixedHeight(60)
        self.pickButton.setStyleSheet("color: red; font-size: 16px;")

        self.exitButton = QPushButton("Return")
        self.exitButton.setFixedHeight(60)
        self.exitButton.setStyleSheet("font-size: 16px;")

        vlay2.addWidget(self.help)
        vlay2.addWidget(self.cpTable)
        vlay2.addWidget(self.delButton)

        vlay2.addLayout(hlay_buttons)
        hlay_buttons.addWidget(self.pickButton)
        hlay_buttons.addWidget(self.exitButton)

        self.updateCPtable()
        self.statusBar().showMessage("Ready")


    def create_connections(self):
        self.pickButton.clicked.connect(self.pickmodechange)
        self.delButton.clicked.connect(self.delCP)


    def menu_quit(self):

        matched_points_dict = self.get_dictlist()
        src, dst = point_coords(matched_points_dict)
        print(src)
        print(dst)

        # TODO: correlation fix
        # result, overlay_adorned_image, fluorescence_image_rgb, fluorescence_original = correlate_images(img1, img2, output, matched_points_dict)
        # result = correlate_images(src_image=image1, target_image=image2, output_path=None,
        #                           matched_points_dict=matched_points_dict, save=False)
        result = correlate_images_projective(src_image=image1, target_image=image2, output_path=None,
                                  matched_points_dict=matched_points_dict, save=False)
        self.close()
        return result


    def get_dictlist(self):
        dictlist = []
        for cp in self.wp.canvas.CPlist:
            dictlist.append(cp.getdict)
        return dictlist


    def pickmodechange(self):
        #if self.wp.canvas.toolbar._active in ["", None]:
        #if not self.wp.canvas.toolbar.isActiveWindow():

        #if not self.wp.canvas.manager.toolbar.mode.value in ["", None]:
        if self.wp.canvas.pickmode == True:
            self.wp.canvas.pickMode_changed = True
            self.wp.canvas.pickmode = False
            self.statusBar().showMessage("Pick Mode deactivate.")
            self.wp.canvas.cursorGUI = "arrow"
            self.wp.canvas.cursorChanged = True
        else:
            self.wp.canvas.pickMode_changed = True
            self.wp.canvas.pickmode = True
            #self.wp.canvas.toolbar._active = ""
            self.statusBar().showMessage(
                "Pick Mode activate. Select Control Points."
            )

        # else:
        #     self.statusBar().showMessage(
        #         f"Please, first deactivate the selected navigation tool"
        #         #f"{self.wp.canvas.toolbar._active}",
        #         f"{self.wp.canvas.toolbar.isActiveWindow()}",
        #         3000,
        #     )

    def delCP(self):
        rows = self.cpTable.selectionModel().selectedRows()
        for row in rows:
            try:
                idp = int(row.data())
                for cp in self.wp.canvas.CPlist:
                    if cp.idp == idp:
                        index = self.wp.canvas.CPlist.index(cp)
                        self.wp.canvas.CPlist.pop(index)
            except Exception as e:
                print("Error occured: '{}'".format(e))
                pass

        self.wp.canvas.updateCanvas()
        self.wp.canvas.cpChanged = True


    def updateGUI(self):
        #if self.wp.canvas.toolbar._active not in ["", None]:
        #if not self.wp.canvas.toolbar.isActiveWindow():
        self.wp.canvas.pickmode = False
        self.wp.canvas.pickMode_changed = True

        if self.wp.canvas.pickMode_changed:
            if not self.wp.canvas.pickmode:
                self.pickButton.setStyleSheet("color: red; font-size: 20px;")
            elif self.wp.canvas.pickmode:
                self.pickButton.setStyleSheet("color: green; font-size: 20px;")
            self.wp.canvas.pickMode_changed = False

        if self.wp.canvas.cursorChanged:
            if self.wp.canvas.cursorGUI == "cross":
                QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
            elif self.wp.canvas.cursorGUI == "arrow":
                QApplication.restoreOverrideCursor()
            self.wp.canvas.cursorChanged = False

        if self.wp.canvas.cpChanged:
            self.updateCPtable()

    def updateCPtable(self):
        self.wp.canvas.cpChanged = False
        self.cpTable.clearSelection()
        self.cpTabelModel.clear()
        self.cpTabelModel.setHorizontalHeaderLabels(
        ["Point Number", "x (Img 1)", "y (Img 1)", "x (Img 2)", "y (Img 2)"]
        )

        for cp in self.wp.canvas.CPlist:
            idp, x1, y1, x2, y2 = cp.coordText

            c1 = QStandardItem(idp)
            c2 = QStandardItem(x1)
            c3 = QStandardItem(y1)
            c4 = QStandardItem(x2)
            c5 = QStandardItem(y2)

            row = [c1, c2, c3, c4, c5]

            for c in row:
                c.setTextAlignment(Qt.AlignCenter)
                c.setFlags(Qt.ItemIsEditable)
                c.setFlags(Qt.ItemIsSelectable)

            self.cpTabelModel.appendRow(row)

        self.cpTable.resizeColumnsToContents()


class _WidgetPlot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = _PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)


class _PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        FigureCanvas.__init__(self, self.fig)

        self.setParent(parent)
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()
        self.create_connections()

        self.figureActive = False
        self.axesActive = None
        self.CPactive = None
        self.pickmode = False
        self.pickMode_changed = True
        self.cpChanged = False
        self.cursorGUI = "arrow"
        self.cursorChanged = False
        self.CPlist = []
        self.lastIDP = 0

    def plot(self):
        gs0 = self.fig.add_gridspec(1, 2)

        self.ax11 = self.fig.add_subplot(
            gs0[0], xticks=[], yticks=[], title="Image 1: Select Points")
        self.ax12 = self.fig.add_subplot(
            gs0[1], xticks=[], yticks=[], title="Image 2: Select Points")

        self.ax11.imshow(image1)
        self.ax12.imshow(image2)

    def updateCanvas(self, event=None):
        ax11_xlim = self.ax11.get_xlim()
        ax11_xvis = ax11_xlim[1] - ax11_xlim[0]
        ax12_xlim = self.ax12.get_xlim()
        ax12_xvis = ax12_xlim[1] - ax12_xlim[0]

        while len(self.ax11.patches) > 0:
            [p.remove() for p in self.ax11.patches]
        while len(self.ax12.patches) > 0:
            [p.remove() for p in self.ax12.patches]
        while len(self.ax11.texts) > 0:
            [t.remove() for t in self.ax11.texts]
        while len(self.ax12.texts) > 0:
            [t.remove() for t in self.ax12.texts]

        ax11_units = ax11_xvis * 0.003
        ax12_units = ax12_xvis * 0.003

        for cp in self.CPlist:
            x1 = cp.img1x
            y1 = cp.img1y
            x2 = cp.img2x
            y2 = cp.img2y
            idp = str(cp.idp)

            if x1:
                symb1 = plt.Circle(
                    (x1, y1), ax11_units * 8, fill=False, color="red")
                symb2 = plt.Circle(
                    (x1, y1), ax11_units * 1, fill=True, color="red")
                self.ax11.text(x1 + ax11_units * 5, y1 + ax11_units * 5, idp)
                self.ax11.add_patch(symb1)
                self.ax11.add_patch(symb2)

            if x2:
                symb1 = plt.Circle(
                    (x2, y2), ax12_units * 8, fill=False, color="red")
                symb2 = plt.Circle(
                    (x2, y2), ax12_units * 1, fill=True, color="red")
                self.ax12.text(x2 + ax12_units * 5, y2 + ax12_units * 5, idp)
                self.ax12.add_patch(symb1)
                self.ax12.add_patch(symb2)

        self.fig.canvas.draw()

    def create_connections(self):
        self.fig.canvas.mpl_connect("figure_enter_event", self.activeFigure)
        self.fig.canvas.mpl_connect("figure_leave_event", self.leftFigure)
        self.fig.canvas.mpl_connect("axes_enter_event", self.activeAxes)
        self.fig.canvas.mpl_connect("button_press_event", self.mouseClicked)
        self.ax11.callbacks.connect("xlim_changed", self.updateCanvas)
        self.ax12.callbacks.connect("xlim_changed", self.updateCanvas)

    def activeFigure(self, event):
        self.figureActive = True
        if self.pickmode and self.cursorGUI != "cross":
            self.cursorGUI = "cross"
            self.cursorChanged = True

    def leftFigure(self, event):
        self.figureActive = False
        if self.cursorGUI != "arrow":
            self.cursorGUI = "arrow"
            self.cursorChanged = True

    def activeAxes(self, event):
        self.axesActive = event.inaxes

    def mouseClicked(self, event):
        x = event.xdata
        y = event.ydata

        if self.toolbar.mode != "":
            self.pickmode = False

        if self.pickmode and (
            (event.inaxes == self.ax11) or (event.inaxes == self.ax12)
        ):

            if self.CPactive and not self.CPactive.status_complete:
                self.CPactive.appendCoord(x, y)
                self.cpChanged = True
            else:
                idp = self.lastIDP + 1
                cp = _ControlPoint(idp, x, y, self)
                self.CPlist.append(cp)
                self.cpChanged = True
                self.lastIDP += 1

            self.updateCanvas()


class _ControlPoint:
    def __init__(self, idp, x, y, other):
        self.img1x = None
        self.img1y = None
        self.img2x = None
        self.img2y = None
        self.status_complete = False
        self.idp = idp

        self.mn = other
        self.mn.CPactive = self

        self.appendCoord(x, y)

    def appendCoord(self, x, y):
        if self.mn.axesActive == self.mn.ax11 and self.img1x is None:
            self.img1x = x
            self.img1y = y
        elif self.mn.axesActive == self.mn.ax12 and self.img2x is None:
            self.img2x = x
            self.img2y = y

        else:
            raise Exception("Please, select control point in the other image.")

        if self.img1x and self.img2x:
            self.status_complete = True
            self.mn.cpActive = None

    @property
    def coord(self):
        return self.idp, self.img1x, self.img1y, self.img2x, self.img2y

    @property
    def coordText(self):
        if self.img1x and not self.img2x:
            return (
                str(round(self.idp, 2)),
                str(round(self.img1x, 2)),
                str(round(self.img1y, 2)),
                "",
                "",
            )
        elif not self.img1x and self.img2x:
            return (
                str(round(self.idp, 2)),
                "",
                "",
                str(round(self.img2x, 2)),
                str(round(self.img2y, 2)),
            )
        else:
            return (
                str(round(self.idp, 2)),
                str(round(self.img1x, 2)),
                str(round(self.img1y, 2)),
                str(round(self.img2x, 2)),
                str(round(self.img2y, 2)),
            )

    def __str__(self):
        return f"CP {self.idp}: {self.coord}"

    @property
    def getdict(self):

        dict = {
            "point_id": self.idp,
            "img1_x": self.img1x,
            "img1_y": self.img1y,
            "img2_x": self.img2x,
            "img2_y": self.img2y,
        }

        return dict





def correlate_images(src_image, target_image, output_path, matched_points_dict, save=False):
    """Correlates two images using points chosen by the user
    Parameters
    ----------
    fluorescence_image_rgb :
        numpy array with shape (cols, rows, channels)
    fibsem_image : AdornedImage.
        Expecting .data attribute of shape (cols, rows, channels)
    output : str
        Path to save location
    matched_points_dict : dict
    Dictionary of points selected in the correlation window
    """
    if matched_points_dict == []:
        print('No control points selected, exiting.')
        return

    src, dst = point_coords(matched_points_dict)
    transformation = calculate_transform(src, dst)
    src_image_aligned = apply_transform(src_image, transformation)
    result = overlay_images(src_image_aligned, target_image)
    result = skimage.util.img_as_ubyte(result)

    # plt.imsave(output_path, result)
    if save:
        pass
        #save result image in

    return result


def correlate_images_projective(src_image, target_image, output_path, matched_points_dict, save=False):

    if matched_points_dict == []:
        print('No control points selected, exiting.')
        return

    src, dst = point_coords(matched_points_dict)
    transformation = calculate_transform_projective(src, dst)
    src_image_aligned = apply_transform_projective(src_image, transformation)
    result = overlay_images(src_image_aligned, target_image)
    result = skimage.util.img_as_ubyte(result)

    # plt.imsave(output_path, result)
    if save:
        pass
        #save result image in

    return result



def point_coords(matched_points_dict):
    """Create source & destination coordinate numpy arrays from cpselect dict.
    Matched points is an array where:
    * the number of rows is equal to the number of points selected.
    * the first column is the point index label.
    * the second and third columns are the source x, y coordinates.
    * the last two columns are the destination x, y coordinates.
    Parameters
    ----------
    matched_points_dict : dict
        Dictionary returned from cpselect containing matched point coordinates.
    Returns
    -------
    (src, dst)
        Row, column coordaintes of source and destination matched points.
        Tuple contains two N x 2 ndarrays, where N is the number of points.
    """

    matched_points = np.array([list(point.values())
                               for point in matched_points_dict])
    src = np.flip(matched_points[:, 1:3], axis=1)  # flip for row, column index
    dst = np.flip(matched_points[:, 3:], axis=1)   # flip for row, column index

    return src, dst


def calculate_transform(src, dst, model=AffineTransform()):
    """Calculate transformation matrix from matched coordinate pairs.
    Parameters
    ----------
    src : ndarray
        Matched row, column coordinates from source image.
    dst : ndarray
        Matched row, column coordinates from destination image.
    model : scikit-image transformation class, optional.
        By default, model=AffineTransform()
    Returns
    -------
    ndarray
        Transformation matrix.
    """

    model.estimate(src, dst)
    print('Transformation matrix:')
    print(model.params)

    return model.params


def apply_transform(image, transformation, inverse=True, multichannel=True):
    """Apply transformation to a 2D image.
    Parameters
    ----------
    image : ndarray
        Input image array. 2D grayscale image expected, or
        2D plus color channels if multichannel kwarg is set to True.
    transformation : ndarray
        Affine transformation matrix. 3 x 3 shape.
    inverse : bool, optional
        Inverse transformation, eg: aligning source image coords to destination
        By default `inverse=True`.
    multichannel : bool, optional
        Treat the last dimension as color, transform each color separately.
        By default `multichannel=True`.
    Returns
    -------
    ndarray
        Image warped by transformation matrix.
    """

    if inverse:
        transformation = np.linalg.inv(transformation)

    if not multichannel:
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        elif image.ndim != transformation.shape[0] - 1:
            raise ValueError('Unexpected number of image dimensions for the '
                             'input transformation. Did you need to use: '
                             'multichannel=True ?')

    # move channel axis to the front for easier iteration over array
    image = np.moveaxis(image, -1, 0)
    warped_img = np.array([ndi.affine_transform((img_channel), transformation)
                           for img_channel in image])
    warped_img = np.moveaxis(warped_img, 0, -1)

    return warped_img



def calculate_transform_projective(src, dst):
    src = np.flip(src, axis=1)
    dst = np.flip(dst, axis=1)
    tform = skimage.transform.ProjectiveTransform()
    tform.estimate(src, dst)
    return tform.inverse


def apply_transform_projective(image, transformation, multichannel=True):
    if not multichannel:
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        elif image.ndim != transformation.shape[0] - 1:
            raise ValueError('Unexpected number of image dimensions for the '
                             'input transformation. Did you need to use: '
                             'multichannel=True ?')

    # move channel axis to the front for easier iteration over array
    image = np.moveaxis(image, -1, 0)
    warped_img = np.array([skimage.transform.warp((img_channel), transformation)
                           for img_channel in image])
    warped_img = np.moveaxis(warped_img, 0, -1)

    return warped_img


def overlay_images(image1, image2, transparency=0.5):
    """Blend two RGB images together.
    Parameters
    ----------
    image1 : ndarray
        2D RGB image.
    image2 : ndarray
        2D RGB image.
    transparency : float, optional
        Transparency alpha parameter between 0 - 1, by default 0.5
    Returns
    -------
    ndarray
        Blended 2D RGB image.
    """

    image1 = skimage.img_as_float(image1)
    image2 = skimage.img_as_float(image2)
    blended = transparency * image1 + (1 - transparency) * image2
    blended = np.clip(blended, 0, 1)

    return blended


def save_text(output_filename, transformation, matched_points_dict):
    """Save text summary of transformation matrix and image control points.
    Parameters
    ----------
    output_filename : str
        Filename for saving output overlay image file.
    transformation : ndarray
        Transformation matrix relating the two images.
    matched_points_dict : list of dict
        User selected matched control point pairs.
    Returns
    -------
    str
        Filename of output text file.
    """

    output_text_filename = os.path.splitext(output_filename)[0] + '.txt'
    with open(output_text_filename, 'w') as f:
        f.write(_timestamp() + '\n')
        f.write('PIEScope GUI version {}\n'.format(__version__))
        f.write('\nTRANSFORMATION MATRIX\n')
        f.write(str(transformation) + '\n')
        f.write('\nUSER SELECTED CONTROL POINTS\n')
        f.write(str(matched_points_dict) + '\n')

    return output_text_filename





def normalise(img):
    img_normalised = (img - np.mean(img)) / np.std(img)
    return img_normalised


def circ_mask(size=(128, 128), radius=32, sigma=3):
    x = size[0]
    y = size[1]
    img = Image.new('I', size)
    draw = ImageDraw.Draw(img)
    draw.ellipse((x / 2 - radius, y / 2 - radius, x / 2 + radius, y / 2 + radius), fill='white', outline='white')
    tmp = np.array(img, float) / 255
    if sigma > 0:
        mask = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        mask = tmp
    return mask


def rectangular_mask(size=(128, 128), sigma=None):
    # leave at least a 5% gap on each edge
    start = np.round(np.array(size) * 0.25)
    extent = np.round(np.array(size) * 0.5)
    rr, cc = skimage.draw.rectangle(start, extent=extent, shape=size)
    mask = np.zeros(size)
    mask[rr.astype(int), cc.astype(int)] = 1.0
    if sigma:
        mask = ndi.gaussian_filter(mask, sigma=sigma)
    return mask


def ellipse_mask(size=(128, 128), radius1=32, radius2=32, sigma=3):
    x = size[0]
    y = size[1]
    img = Image.new('I', size)
    draw = ImageDraw.Draw(img)
    draw.ellipse((x / 2 - radius1, y / 2 - radius2, x / 2 + radius1, y / 2 + radius2), fill='white', outline='white')
    tmp = np.array(img, float) / 255
    if sigma > 0:
        mask = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        mask = tmp
    return mask


def bandpass_mask(size=(128, 128), low_pass=32, high_pass=2, sigma=3):
    x = size[0]
    y = size[1]
    lowpass = circ_mask(size=(x, y), radius=low_pass, sigma=0)
    highpass = circ_mask(size=(x, y), radius=high_pass, sigma=0)
    highpass = -1 * (highpass - 1)
    bandpass = lowpass * highpass
    if sigma > 0:
        bandpass = ndi.filters.gaussian_filter(bandpass, sigma=sigma)
    return bandpass


def crosscorrelation(img1, img2, filter='no', *args, **kwargs):
    if img1.shape != img2.shape:
        print('### ERROR in xcorr2: img1 and img2 do not have the same size ###')
        return -1
    if img1.dtype != 'float64':
        img1 = np.array(img1, float)
    if img2.dtype != 'float64':
        img2 = np.array(img2, float)

    if filter == 'yes':
        low_pass = kwargs.get('low_pass', None)
        high_pass = kwargs.get('high_pass', None)
        sigma = kwargs.get('sigma', None)
        if low_pass == 'None' or high_pass == 'None' or sigma == 'None':
            print('ERROR in xcorr: check bandpass parameters')
            return

        bandpass = bandpass_mask(size=(img1.shape[1], img1.shape[0]),
                                 low_pass=low_pass,
                                 high_pass=high_pass,
                                 sigma=sigma)
        img1ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img1)))
        s = img1.shape[0] * img1.shape[1]
        tmp = img1ft * np.conj(img1ft)
        img1ft = s * img1ft / np.sqrt(tmp.sum())
        img2ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img2)))
        img2ft[0, 0] = 0
        tmp = img2ft * np.conj(img2ft)
        img2ft = s * img2ft / np.sqrt(tmp.sum())
        xcorr = np.real(fftpack.fftshift(fftpack.ifft2(img1ft * np.conj(img2ft))))
    elif filter == 'no':
        img1ft = fftpack.fft2(img1)
        img2ft = np.conj(fftpack.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(fftpack.fftshift(fftpack.ifft2(img1ft * img2ft)))
    else:
        print('ERROR in xcorr: filter value ( filter= ' + str(filter) + ' ) not recognized')
        return -1
    return xcorr


def shift_from_crosscorrelation_simple_images(img1, img2, low_pass=256, high_pass=22, sigma=2):
    img1_norm = normalise(img1)
    img2_norm = normalise(img2)
    xcorr = crosscorrelation(img1_norm, img2_norm, filter='yes',
                             low_pass=low_pass,
                             high_pass=high_pass,
                             sigma=sigma)
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    err = np.array(cen - [maxX, maxY], int)
    print('centre = ', cen)
    print("Shift between 1 and 2 is = " + str(err))
    print("img2 is X-shifted by ", err[1], '; Y-shifted by ', err[0])
    return err[1], err[0]




def subpixel_shift_from_crosscorrelation(ref_image, offset_image, upsample_factor=100):
    # pixel precision
    shift, error, diffphase = phase_cross_correlation(ref_image, offset_image)
    image_product = np.fft.fft2(ref_image) * np.fft.fft2(offset_image).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    cc_image = cc_image.real
    print(f'Detected pixel offset (y, x): {shift}')

    # subpixel precision
    shift, error, diffphase = phase_cross_correlation(ref_image, offset_image,
                                                      upsample_factor=upsample_factor)
    print(f'Detected subpixel offset (y, x): {shift}')
    offset_image_aligned = scipy.ndimage.shift(offset_image, shift,
                                               output=None, order=3, mode='constant', cval=0.0, prefilter=True)
    return shift, offset_image_aligned



def load_image(file_path):
    image = Image.open(file_path)
    image = np.array(image)
    return image





if __name__ == '__main__':
    image1 = load_image("01_tilted.tif")
    #image2 = load_image("02_flat.tif")
    #image3 = load_image("02_flat_shifted.tif")



    if 0:
        image1 = skimage.color.gray2rgb(image1)
        image2 = skimage.color.gray2rgb(image2)
        image1 = skimage.transform.resize(image1, image2.shape)

        src = [[ 445.28349348,  713.83713241],
               [ 330.83521061,  713.83713241],
               [ 414.21895956,  860.98492467],
               [ 523.76231602,  775.96620025],
               [ 442.01354254,  975.43320754],
               [ 482.88792928,  844.63516997],
               [ 476.3480274,  1425.05146167],
               [ 293.23077481,  849.54009638]]

        dst = [[ 507.41256133,  727.96926583],
               [ 296.50072575,  723.06433942],
               [ 448.55344442,  878.38700903],
               [ 646.38547624,  788.4633582 ],
               [ 507.41256133,  984.66041455],
               [ 572.81158011,  857.13232792],
               [ 579.35148199, 1409.75403664],
               [ 229.4667315,   868.57715621]]

        src = np.array(src)
        dst = np.array(dst)

        src = np.flip(src, axis=1)
        dst = np.flip(dst, axis=1)

        tform = skimage.transform.ProjectiveTransform()
        tform.estimate(src, dst)
        print(tform)

        image1_aligned = apply_transform_projective(image1, tform.inverse)

        overlay = overlay_images(image1_aligned, image2)



        # angle_x = np.deg2rad(0)
        # angle_y = np.deg2rad(2)
        # matrix = np.array([[-0.5, 0, 0],
        #                   [0, -0.5, 0],
        #                   [0, 0, 1]
        #                    ])
        # tform2 = skimage.transform.ProjectiveTransform(matrix=matrix)
        # print(tform2)
        # warped_image = skimage.transform.warp(image1, tform2.inverse)




        def rotate_about_x(points, angle):
            angle = np.deg2rad(angle)
            # matrix = np.array([
            #     [1 , 0 , 0],
            #     [0 , np.cos(angle) , -np.sin(angle)],
            #     [0 , np.sin(angle) , np.cos(angle)]
            # ])
            matrix = np.array([
                [np.cos(angle) , 0 , np.sin(angle)],
                [0 , 1 , 0],
                [-np.sin(angle) , 0 , np.cos(angle)]
            ])
            points = np.array(points)
            centre  = np.asarray((1024,1536)) / 2
            points = points - centre
            points_rot = np.zeros( (points.shape[0], points.shape[1]+1) )
            points_rot = [[point[0],point[1],0] for point in points]
            points_rot = [ np.dot(matrix,point) for point in points_rot ]
            points_rot = np.array(points_rot)

            centre = np.asarray((1024, 1536,0)) / 2
            points_rot += centre
            return points_rot

        def project_3d_point_to_xy(points):
            points_2d = np.zeros((points.shape[0], points.shape[1] - 1))
            points_2d = [[point[0], point[1]] for point in points]
            points_2d = np.array(points_2d)
            return points_2d

        points_on_flat_surface = np.array([
            [0, 0],
            [0, 256],
            [0, 768],
            [0, 1536],
            [256, 0],
            [256, 256],
            [256, 512],
            [256, 1024],
            [256, 1536],
            [512, 0],
            [512, 256],
            [512, 512],
            [512, 1024],
            [768, 0],
            [768, 256],
            [768, 512],
            [768, 768],
            [768, 1024],
            [1024, 0],
            [1024, 256],
            [1024, 512],
            [1024, 768],
            [1024, 1024],
            [1024, 1536]
            ])


        ANGLE = 27.
        points_rotate_about_x = rotate_about_x(points_on_flat_surface, ANGLE)
        points_rotated = project_3d_point_to_xy(points_rotate_about_x)

        plt.figure(31)
        [plt.scatter(point[0],point[1], c='b') for point in points_on_flat_surface]
        [plt.scatter(point[0],point[1], c='r') for point in points_rotated]
        plt.title('matrix transform')

        tform2 = skimage.transform.ProjectiveTransform()
        tform2.estimate(points_on_flat_surface, points_rotated)
        print(tform2)
        image1_transform = apply_transform_projective(image1, tform2.inverse)
        ####################
        plt.figure(32)
        plt.subplot(3, 1, 1)
        plt.imshow(image1, cmap='gray')
        plt.title('image_1')
        plt.subplot(3, 1, 2)
        plt.imshow(image1_transform, cmap='gray')
        plt.title('image_1_transform')
        plt.subplot(3, 1, 3)
        plt.imshow(overlay_images(image1_transform, image2), cmap='gray')
        plt.title('overlay')



        plt.figure(30)
        plt.subplot(2, 2, 1)
        plt.imshow(image1, cmap='gray')
        [plt.scatter(point[0],point[1]) for point in src]
        plt.title('image_1')

        plt.subplot(2, 2, 2)
        plt.imshow(image2, cmap='gray')
        [plt.scatter(point[0],point[1]) for point in dst]
        plt.title('image_2')

        plt.subplot(2, 2, 3)
        plt.imshow(image1_aligned, cmap='gray')
        plt.title('image_1 aligned')

        plt.subplot(2, 2, 4)
        plt.imshow(overlay, cmap='gray')
        plt.title('overlay')

        plt.show()








