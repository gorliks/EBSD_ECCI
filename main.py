import qtdesigner_files.main_gui as gui_main
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QObject, QThread, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
import qimage2ndarray

from importlib import reload  # Python 3.4+

from dataclasses import dataclass

import sys, time, os, glob
import numpy as np
import SEM
import correlation
import movement
import detection as detection

import ui_utils

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as _NavigationToolbar,
)

import utils
from utils import BeamType

import h5py
import hyperspy.api as hs
import kikuchipy as kp

@dataclass
class StackSettings:
    x_range: float = 0
    dx: float = 0
    Nx: int = 0
    y_range: float = 0
    dy: float = 0
    Ny: int = 0
    z_range: float = 0
    dz: float = 0
    Nz: int = 0
    r_range: float = 0
    dr: float = 0
    Nr: int = 0
    t_range: float = 0
    dt: float = 0
    Nt: int = 0

    def update_numbers_in_stack(self, numbers: list):
        self.Nx = numbers[0]
        self.Ny = numbers[1]
        self.Nz = numbers[2]
        self.Nt = numbers[3]
        self.Nr = numbers[4]


class GUIMainWindow(gui_main.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, demo):
        super(GUIMainWindow, self).__init__()
        self.setupUi(self)
        self.demo = demo
        self.time_counter = 0
        print('mode demo is ', demo)
        self.setStyleSheet("""QPushButton {
        border: 1px solid lightgray;
        border-radius: 5px;
        background-color: #e3e3e3;
        }""")

        self.DIR = os.getcwd()
        self.coords = [0, 0]

        self.setup_connections()
        self.initialise_image_frames()
        self.initialise_hardware()
        self.stack_settings = StackSettings()
        self.stack_settings.x_range = 100

        self.pushButton_abort_stack_collection.setEnabled(False)
        self.pushButton_abort_stack_collection_detector.setEnabled(False)
        self._abort_clicked_status = False
        self._blanked = False

        self.image = None
        self.image_mod = None
        self.current_image = None

        self.corr_image_1_path = None
        self.corr_image_2_path = None

        # timer on a separate thread
        # self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        # self.timer = QTimer()
        # self.timer.setInterval(1000) #1 s intervals
        # self.timer.timeout.connect(self._time_counter)
        # self.timer.start()

    def setup_connections(self):
        self.label_demo_mode.setText('demo mode: ' + str(self.demo))
        self.pushButton_acquire.clicked.connect(lambda: self.acquire_image())
        self.pushButton_select_directory.clicked.connect(lambda: self.select_directory())
        self.pushButton_initialise_microscope.clicked.connect(lambda: self.initialise_hardware())
        self.pushButton_last_image.clicked.connect(lambda: self.last_image())
        self.pushButton_update_stage_position.clicked.connect(lambda: self.update_stage_position())
        self.pushButton_move_stage.clicked.connect(lambda: self.move_stage())
        self.pushButton_set_scan_rotation.clicked.connect(lambda: self.set_scan_rotation())
        self.pushButton_estimate_stack_time.clicked.connect(lambda: self.estimate_stack_time())
        self.pushButton_abort_stack_collection.clicked.connect(lambda: self._abort_clicked())
        self.pushButton_collect_stack.clicked.connect(lambda: self.collect_stack())
        self.comboBox_move_type.currentTextChanged.connect(lambda: self._change_of_move_type())
        self.pushButton_update_SEM_state.clicked.connect(lambda: self.update_SEM_state())
        self.pushButton_set_beam_shift.clicked.connect(lambda: self.set_beam_shift())
        self.pushButton_reset_beam_shift.clicked.connect(lambda: self.reset_beam_shift())
        self.pushButton_set_beam_point.clicked.connect(lambda: self.set_beam_point())
        self.pushButton_beam_blank.clicked.connect(lambda: self.beam_blank())
        self.pushButton_set_full_frame.clicked.connect(lambda: self.set_full_frame())

        #
        self.pushButton_correlation.clicked.connect(lambda: self.test_correlation())
        self.pushButton_open_window.clicked.connect(lambda: self.test_open_window())
        self.pushButton_open_file.clicked.connect(lambda: self._open_file())
        self.pushButton_save_file.clicked.connect(lambda: self._save_SEM_image())
        #
        self.pushButton_set_EBSD_detector.clicked.connect(lambda: self.setup_EBSD_detector())
        self.pushButton_open_EBSD_file.clicked.connect(lambda: self._open_EBSD_file())
        self.pushButton_open_EBSD_stack.clicked.connect(lambda: self._open_EBSD_stack())
        #
        self.pushButton_apply_clahe.clicked.connect(lambda: self._apply_clahe())
        self.pushButton_restore.clicked.connect(lambda: self._restore_image())
        #
        self.pushButton_select_image_1.clicked.connect(lambda: self._select_corr_image(num=1))
        self.pushButton_select_image_2.clicked.connect(lambda: self._select_corr_image(num=2))
        self.pushButton_run_correlation.clicked.connect(lambda: self.run_correlation())
        # Detector
        self.pushButton_initialise_detector.clicked.connect(lambda: self.initialise_detector())
        self.pushButton_close_detector.clicked.connect(lambda: self.close_detector())
        self.pushButton_setup_acquisition.clicked.connect(lambda: self.setup_acquisition())
        self.pushButton_acquire_detector.clicked.connect(lambda: self.single_acquisition())
        self.pushButton_select_directory.clicked.connect(lambda: self.select_directory())
        self.pushButton_check_temperature.clicked.connect(lambda: self.update_temperature())
        self.pushButton_collect_stack_detector.clicked.connect(lambda: self.collect_stack_detector())
        self.pushButton_abort_stack_collection_detector.clicked.connect(lambda: self._abort_clicked())



    def collect_stack_detector(self):
        self.setup_acquisition()  # update the acquisition parameters
        stamp = utils.current_timestamp()
        self.sample_id = self.plainTextEdit_sample_name_detector.toPlainText()
        self.pushButton_abort_stack_collection_detector.setEnabled(True)
        self.pushButton_acquire_detector.setEnabled(False)
        self.pushButton_setup_acquisition.setEnabled(False)
        self.pushButton_check_temperature.setEnabled(False)

        stack_i = self.spinBox_scan_pixels_i.value()  # scan over sample, no. pixels along X
        stack_j = self.spinBox_scan_pixels_j.value()  # scan over sample, no. pixels along Y
        x0 = self.spinBox_x0.value()  # start scan from pixel x0
        y0 = self.spinBox_y0.value()  # start scan from pixel y0

        # initialise the data storage
        # TODO detector image Nx,Ny settings more generic
        Nx = 256
        Ny = 256
        self.storage_EVENT = hs.signals.Signal2D(np.zeros((stack_i, stack_j,
                                                           Nx,Ny))) #i,j pixels in the image [Nx,Ny] spectra for each i,j pixel
        self.storage_iTOT  = hs.signals.Signal2D(np.zeros((stack_i, stack_j,
                                                           Nx,
                                                           Ny)))  # i,j pixels in the image [Nx,Ny] spectra for each i,j pixel


        if self.DIR:
            self.stack_dir = self.DIR + '/stack_' + self.sample_id + '_' + stamp
        else:
            self.stack_dir = os.getcwd() + '/stack_' + self.sample_id + '_' + stamp

        # the loop will check if abort button is clicked by checking QtWidgets.QApplication.processEvents()
        # if the abort button was clicked, _return_ will break the _run_loop
        # TODO use threads for more elegant solution
        def _run_loop():
            pixel_counter = 0
            for ii in range(stack_i):
                for jj in range(stack_j):
                    print(x0 + ii, y0 + jj)
                    self.label_current_i.setText(f'{ii + 1} of {stack_i}')
                    self.label_current_j.setText(f'{jj + 1} of {stack_j}')

                    file_name = '%06d_' % pixel_counter + str(ii) + '_' + str(jj)

                    """ set point coordinate, unblank beam, colelct data, blank beam """
                    self.microscope.set_beam_point(beam_x=x0 + ii, beam_y=y0 + jj)
                    self.microscope.unblank()
                    # get_data, get_data also does update_image
                    self.data = self.get_data(save_dir=self.stack_dir, file_name=file_name)
                    self.microscope.blank()

                    """ store the data in hyperspy stacks """
                    self.storage_EVENT.data[ii][jj] = self.data['EVENT']
                    self.storage_iTOT.data[ii][jj]  = self.data['iTOT']

                    """ plot the new data """
                    self.repaint()  # update the GUI to show the progress
                    QtWidgets.QApplication.processEvents()

                    if self._abort_clicked_status == True:
                        print('Abort clicked')
                        self._abort_clicked_status = False  # reinitialise back to False
                        return
                    pixel_counter += 1

        _run_loop()

        # save the stacks
        if self.checkBox_save_in_h5_format.isChecked():
            print('saving h5 format')
            file_name = 'EVENT' + '_stack_' + str(stack_i) + '_' + str(stack_j) + '.h5'
            file_name = os.path.join(self.stack_dir, file_name)
            self.storage_EVENT.save(file_name)
            #####
            file_name = 'iTOT' + '_stack_' + str(stack_i) + '_' + str(stack_j) + '.h5'
            file_name = os.path.join(self.stack_dir, file_name)
            self.storage_iTOT.save(file_name)

        if self.checkBox_save_in_hspy_format.isChecked():
            print('saving h5 format')
            file_name = "EVENT" + '_stack_' + str(stack_i) + '_' + str(stack_j) + '.hspy'
            file_name = os.path.join(self.stack_dir, file_name)
            self.storage_EVENT.save(file_name)
            ####
            file_name = "iTOT" + '_stack_' + str(stack_i) + '_' + str(stack_j) + '.hspy'
            file_name = os.path.join(self.stack_dir, file_name)
            self.storage_iTOT.save(file_name)


        # Plot the stacks
        self.storage_EVENT.plot()
        plt.title('EVENT')
        self.storage_iTOT.plot()
        plt.title('iTOT')

        plt.show()

        self.pushButton_abort_stack_collection_detector.setEnabled(False)
        self.pushButton_acquire_detector.setEnabled(True)
        self.pushButton_setup_acquisition.setEnabled(True)
        self.pushButton_check_temperature.setEnabled(True)





    def initialise_detector(self):
        path_to_api = self.lineEdit_path_to_detector_API.text()
        print('****************' , path_to_api, '****************')
        self.device = detection.Detector(path_to_api=path_to_api)
        self.device.initialise()
        if self.device.initialised == True:
            self.pushButton_initialise_detector.setStyleSheet("background-color: green")

    def close_detector(self):
        print('closing detector...')
        self.device.close()



    def update_temperature(self):
        if not self.demo:
            temperature = self.device.get_temperature()
        else:
            temperature = 'demo mode: ' + str((np.random.rand() + 0.05) * 100)
        self.label_temperature.setText(str(temperature))


    def setup_acquisition(self):
        """
        Retrieve the settings fromt the main GUI window,
        Prepare the detector for acquisition
        ******** Currently only uses the EVENT_iTOT mode (Minipix code is able to work with different modalities)
        Returns
        -------
        None
        """
        print('setting the acquisition parameters')
        type = self.comboBox_type_of_measurement.currentText()
        mode = 'EVENT_iTOT'     #self.comboBox_mode_of_measurement.currentText()
        number_of_frames = self.spinBox_number_of_frames.value()
        integration_time = self.spinBox_integration_time.value()
        energy_threshold_keV = self.spinBox_energy_threshold.value()
        self.device.setup_acquisition(type=type,
                                      mode=mode,
                                      number_of_frames=number_of_frames,
                                      integration_time=integration_time,
                                      energy_threshold_keV=energy_threshold_keV)



    def single_acquisition(self):
        self.setup_acquisition()
        self.get_data()


    def get_data(self, save_dir=None, file_name=None, update_display=True):
        self.update_temperature()
        stamp = utils.current_timestamp()  # make a timestamp for new file

        if save_dir is not None:
            save_dir = save_dir
        elif save_dir==None and self.DIR == None:
            save_dir = os.getcwd()
        else:
            save_dir = self.DIR


        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        print(self.DIR, save_dir)

        if file_name:
            file_name = save_dir + '/' + file_name + '_' + stamp
        else:
            file_name = save_dir + '/' + stamp
        print(file_name)

        self.integration_time = self.device.integration_time    # TODO update device state in settings self.device.settings['integration_time']


        self.pushButton_acquire_detector.setEnabled(False)
        self.pushButton_setup_acquisition.setEnabled(False)
        self.pushButton_check_temperature.setEnabled(False)
        self.repaint()  # update the GUI to show the progress

        self.data = \
            self.device.acquire(file_name=file_name,
                                type=self.comboBox_type_of_measurement.currentText(),
                                mode='EVENT_iTOT')

        self.pushButton_acquire_detector.setEnabled(True)
        self.pushButton_setup_acquisition.setEnabled(True)
        self.pushButton_check_temperature.setEnabled(True)
        self.repaint()  # update the GUI to show the progress

        if update_display==True:
            image3 = self.data['EVENT']
            if type(image3) == np.ndarray:
                self.update_image(quadrant=3, image=image3)
            image4 = self.data['iTOT']
            if type(image4) == np.ndarray:
                self.update_image(quadrant=4, image=image4)

        return self.data


    def update_image(self, quadrant, image, update_current_image=True):
        self.update_temperature()
        # _convention_ = self.comboBox_image_convention.currentText()
        # if _convention_ == 'TEM convention':
        #     image  = np.flipud(image)
        # elif _convention_ == 'EBSD convention':
        #     image = np.flipud(image)
        #     image = np.fliplr(image)
        # else:
        #     pass
        if update_current_image:
            # self.data_in_quadrant[quadrant] = image
            pass
        image_to_display = qimage2ndarray.array2qimage(image.copy())
        if quadrant==3:
            self.label_image_frame3.setPixmap(QtGui.QPixmap(image_to_display))
        elif quadrant==4:
            self.label_image_frame4.setPixmap(QtGui.QPixmap(image_to_display))
        else:
            self.label_image_frame3.setText('No image acquired')
            self.label_image_frame4.setText('No image acquired')




    def create_settings_dict(self) -> dict:
        resolution = self.comboBox_resolution.currentText()
        dwell_time = self.spinBox_dwell_time.value() * 1e-6
        horizontal_field_width = self.spinBox_horizontal_field_width.value() * 1e-6
        detector = self.comboBox_detector.currentText()
        autocontrast = self.checkBox_autocontrast.isChecked()
        save = self.checkBox_save_image.isChecked()
        beam_type = self.comboBox_beam_type.currentText()
        if beam_type == "ELECTRON":
            beam_type = BeamType.ELECTRON
        elif beam_type == "ION":
            beam_type = BeamType.ION
        else:
            beam_type = BeamType.ELECTRON
        quadrant = int(self.comboBox_quadrant.currentText())
        path = self.DIR
        sample_name = self.plainTextEdit_sample_name.toPlainText()
        bit_depth = int(self.comboBox_bit_depth.currentText())


        self.gui_settings = {
            "imaging": {
                "resolution": resolution,
                "horizontal_field_width": horizontal_field_width,
                "dwell_time": dwell_time,
                "detector": detector,
                "autocontrast": autocontrast,
                "save": save,
                "beam_type": beam_type,
                "quadrant": quadrant,
                "path": path,
                "sample_name": sample_name,
                "bit_depth": bit_depth
            }
        }
        return self.gui_settings

    ##############################################  HARDWARE ###########################################################

    def initialise_hardware(self):
        gui_settings = self.create_settings_dict()
        self.microscope = SEM.Microscope(settings=gui_settings, log_path=None, demo=self.demo)
        self.microscope.establish_connection()
        self.label_messages.setText(str(self.microscope.microscope_state))


    def beam_blank(self):
        status = \
            self.microscope.beam_blank()
        self.label_messages.setText("beam blanked/unblanked: ", status)


    def set_beam_point(self):
        beam_x = int( self.spinBox_beam_x.value() )
        beam_y = int( self.spinBox_beam_y.value() )
        self.microscope.set_beam_point(beam_x=beam_x, beam_y=beam_y)
        self.label_messages.setText(f"beam position set to ({beam_x}, {beam_y})")


    def set_full_frame(self):
        self.microscope.set_full_frame()
        self.label_messages.setText(f"set scan mode to full frame")




    def acquire_image(self,
                      hfw = None):
        gui_settings = self.create_settings_dict()
        self.image = \
            self.microscope.acquire_image(gui_settings=gui_settings,
                                          hfw=hfw)
        try:
            self.pixelsize_x = self.image.metadata.binary_result.pixel_size.x
        except Exception as e:
            self.pixelsize_x = 1
            print(f'Cannot extract pixel size from the image metadata, error {e}')

        self.doubleSpinBox_pixel_size.setValue(self.pixelsize_x / 1e-9)
        self.update_display(image=self.image)
        return self.image


    def last_image(self):
        quadrant = int(self.comboBox_quadrant.currentText())
        self.image = \
            self.microscope.last_image(quadrant=quadrant)
        try:
            image = self.image.data
        except:
            image = self.image
        self.update_display(image=image)


    def move_stage(self):
        compucentric = self.checkBox_compucentric.isChecked()
        move_type = self.comboBox_move_type.currentText()
        #print('compucentric = ', compucentric, '; movw_type = ', move_type)
        x = self.doubleSpinBox_stage_x.value() * 1e-6
        y = self.doubleSpinBox_stage_y.value() * 1e-6
        z = self.doubleSpinBox_stage_z.value() * 1e-6
        r = self.doubleSpinBox_stage_r.value()
        t = self.doubleSpinBox_stage_t.value()
        r = np.deg2rad(r)
        t = np.deg2rad(t)
        self.microscope.move_stage(x=x, y=y, z=z,
                                   move_type=move_type)
        self.microscope.rotate_stage(r=r,
                                     move_type=move_type)
        self.microscope.tilt_stage(t=t,
                                   move_type=move_type)
        self.microscope.update_stage_position()


    def update_stage_position(self):
        self.comboBox_move_type.setCurrentText("Absolute")
        self.microscope.update_stage_position()
        self.doubleSpinBox_stage_x.setValue(self.microscope.microscope_state.x / 1e-6)
        self.doubleSpinBox_stage_y.setValue(self.microscope.microscope_state.y / 1e-6)
        self.doubleSpinBox_stage_z.setValue(self.microscope.microscope_state.z / 1e-6)
        r = np.rad2deg(self.microscope.microscope_state.r)
        self.doubleSpinBox_stage_r.setValue(r)
        t = np.rad2deg(self.microscope.microscope_state.t)
        self.doubleSpinBox_stage_t.setValue(t)


    def update_SEM_state(self):
        self.microscope._get_current_microscope_state()
        self.update_stage_position()
        self.doubleSpinBox_working_distance.setValue(self.microscope.microscope_state.working_distance / 1e-3)

        self.doubleSpinBox_high_voltage.setValue(self.microscope.microscope_state.hv)
        self.doubleSpinBox_beam_current.setValue(self.microscope.microscope_state.beam_current / 1e-9)
        self.doubleSpinBox_brightness.setValue(self.microscope.microscope_state.brightness)
        self.doubleSpinBox_contrast.setValue(self.microscope.microscope_state.contrast)

        self.spinBox_horizontal_field_width.setValue(
            self.microscope.microscope_state.horizontal_field_width / 1e-6 )
        self.doubleSpinBox_scan_rotation.setValue(
            np.rad2deg(self.microscope.microscope_state.scan_rotation_angle)
        )
        self.doubleSpinBox_beam_shift_x.setValue(self.microscope.microscope_state.beam_shift_x / 1e-6)
        self.doubleSpinBox_beam_shift_y.setValue(self.microscope.microscope_state.beam_shift_y / 1e-6)



    def set_scan_rotation(self):
        rotation_angle = self.doubleSpinBox_scan_rotation.value()
        rotation_angle = np.deg2rad(rotation_angle)
        self.microscope.set_scan_rotation(rotation_angle=rotation_angle)


    def set_beam_shift(self):
        beam_shift_x = self.doubleSpinBox_beam_shift_x.value() * 1e-6
        beam_shift_y = self.doubleSpinBox_beam_shift_y.value() * 1e-6
        self.microscope.set_beam_shift(beam_shift_x=beam_shift_x,
                                       beam_shift_y=beam_shift_y)

    def reset_beam_shift(self):
        self.microscope.reset_beam_shifts()

    ##########################################################################################

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, caption='Select a folder')
        print(directory)
        self.label_messages.setText(directory)
        self.DIR = directory

    # def update_image(self, quadrant, image, update_current_image=True):
    #     self.update_temperature()
    #     _convention_ = self.comboBox_image_convention.currentText()
    #     if _convention_ == 'TEM convention':
    #         image  = np.flipud(image)
    #     elif _convention_ == 'EBSD convention':
    #         image = np.flipud(image)
    #         image = np.fliplr(image)
    #     else:
    #         pass
    #     if update_current_image:
    #         self.data_in_quadrant[quadrant] = image
    #     image_to_display = qimage2ndarray.array2qimage(image.copy())
    #     if quadrant in range(0, 4):
    #         self.label_image_frames[quadrant].setPixmap(QtGui.QPixmap(image_to_display))
    #     else:
    #         self.label_image_frames[0].setText('No image acquired')

    # TODO fix pop-up plot bugs
    def initialise_image_frames(self):
        self.figure_SEM = plt.figure(10)
        plt.axis("off")
        plt.tight_layout()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.01)
        self.canvas_SEM = _FigureCanvas(self.figure_SEM)
        self.toolbar_SEM = _NavigationToolbar(self.canvas_SEM, self)
        #
        self.label_image_frame1.setLayout(QtWidgets.QVBoxLayout())
        self.label_image_frame1.layout().addWidget(self.toolbar_SEM)
        self.label_image_frame1.layout().addWidget(self.canvas_SEM)


    """TODO fix bugs with plotting data and using pop-up plots"""
    def update_display(self, image):
        ###### added ######
        # plt.axis("off")
        # self.label_image_frame1.layout().removeWidget(self.canvas_SEM)
        # self.label_image_frame1.layout().removeWidget(self.toolbar_SEM)
        # self.canvas_SEM.deleteLater()
        # self.toolbar_SEM.deleteLater()
        # self.canvas_SEM = _FigureCanvas(self.figure_SEM)
        ###### end added ######

        self.current_image = image

        try:
            image = image.data
        except:
            image = image

        self.figure_SEM.clear()
        self.figure_SEM.patch.set_facecolor(
            (240 / 255, 240 / 255, 240 / 255))
        self.ax = self.figure_SEM.add_subplot(111)
        # ax.set_title("test")

        #### added ######
        # self.toolbar_SEM = _NavigationToolbar(self.canvas_SEM, self)
        # self.label_image_frame1.layout().addWidget(self.toolbar_SEM)
        ##### end added ######

        # self.label_image_frames[quadrant].layout().addWidget(self.canvases[quadrant])
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.imshow(image, cmap='gray')
        self.canvas_SEM.draw()

        def on_click(event):
            coords = []
            coords.append(event.ydata)
            coords.append(event.xdata)
            coords = np.flip(coords[-2:], axis=0)
            coords = movement.pixel_to_realspace_coordinate(coord=coords, image=image)
            self.doubleSpinBox_point_x.setValue(coords[0])
            self.doubleSpinBox_point_y.setValue(coords[1])

            if event.dblclick:
                self.doubleSpinBox_double_click_point_x.setValue(coords[0])
                self.doubleSpinBox_double_click_point_y.setValue(coords[1])

        self.figure_SEM.canvas.mpl_connect("button_press_event", on_click)


    def estimate_stack_time(self):
        gui_settings = self.create_settings_dict()
        resolution = gui_settings["imaging"]["resolution"]
        print("resolution = ", resolution)
        [width, height] = np.array(resolution.split("x")).astype(int)
        pixels_in_frame = width * height

        self._update_stack_settings()

        total_frames = self.stack_settings.Nx * self.stack_settings.Ny * self.stack_settings.Nz * \
                       self.stack_settings.Nt * self.stack_settings.Nr
        overhead = 1.25  # to move the stage etc
        time_to_acquire = total_frames * self.gui_settings["imaging"]["dwell_time"] * \
                          pixels_in_frame * overhead
        self.label_messages.setText(f"Time to collect {total_frames} frames at {resolution} is {time_to_acquire:.1f} s "
                                    f"or {time_to_acquire / 60:.1f} min")


    def collect_stack(self):
        """ Update all the settings, store the current microscope state and
            particularly the current position. After the stack acquisition
            it is possible to return to the original position using move_absolute
            stored state: self.microscope.stored_state : MicroscopeState
            stack setting are stored in self.stack_settings : StackSettings
            If N elements for an axis is smaller than 3 then this axis stays stationary
            If N elements are > 3, the x,y,z axis will move relative by -range and move towards
            +range in delta steps for N times
            For t and r axis, if N>3, the movement will be from the current t or r value in
            +delta steps towards (current_position + range) position
        """
        self._update_stack_settings()
        gui_settings = self.create_settings_dict()
        timestamp = utils.current_timestamp()
        sample_name = self.plainTextEdit_sample_name.toPlainText()

        self.pushButton_acquire.setEnabled(False)
        self.pushButton_collect_stack.setEnabled(False)
        self.pushButton_estimate_stack_time.setEnabled(False)
        self.pushButton_abort_stack_collection.setEnabled(True)

        """Store the current microscope state, including the current position"""
        stored_microscope_state = self.microscope._get_current_microscope_state()
        x0 = stored_microscope_state.x
        y0 = stored_microscope_state.y
        z0 = stored_microscope_state.z
        t0 = stored_microscope_state.t
        r0 = stored_microscope_state.r
        scan_rot0 = stored_microscope_state.scan_rotation_angle

        """Create directory for saving the stack"""
        if self.DIR:
            self.stack_dir = os.path.join(self.DIR, 'stack_' + sample_name + '_' + timestamp)
        else:
            self.stack_dir = os.path.join(os.getcwd(), 'stack_' + sample_name + '_' + timestamp)
        if not os.path.isdir(self.stack_dir):
            os.mkdir(self.stack_dir)

        self.label_messages.setText(f"stack save dir {self.stack_dir}")

        Nx, Ny, Nz, Nt, Nr = self.stack_settings.Nx, self.stack_settings.Ny, self.stack_settings.Nz, \
                             self.stack_settings.Nt, self.stack_settings.Nr

        """ create pandas dataframe for collected data and metadata
            use microscope state to populate the dictionary
            use the same keys to access the dictionary in MicroscopeState.__to__dict__() 
        """
        keys = ('x', 'y', 'z', 't', 'r',
                'horizontal_field_width', 'scan_rotation_angle',
                'brightness', 'contrast',
                'beam_shift_x', 'beam_shift_y')
        self.experiment_data = {element: [] for element in keys}
        """add other data keys"""
        self.experiment_data['file_name'] = []
        self.experiment_data['timestamp'] = []

        """move to the position from where to start the long scan
           move negative x,y,z by half-range (range = 2*x_range etc)
           do not move the tilt or stage rotation, the current position are the start point
        """
        if Nx >= 3:
            self.microscope.move_stage(x=-1 * self.stack_settings.x_range - self.stack_settings.dx,
                                       move_type="Relative")
        if Ny >= 3:
            self.microscope.move_stage(y=-1 * self.stack_settings.y_range - self.stack_settings.dy,
                                       move_type="Relative")
        if Nz >= 3:
            self.microscope.move_stage(z=-1 * self.stack_settings.z_range - self.stack_settings.dz,
                                       move_type="Relative")

        def _run_loop(gui_settings):
            previous_image = None
            counter = 0

            """ ---------------- X Y Z scans ---------------- """
            for ii in range(Nx):
                self.label_x.setText(f"{ii + 1} of")
                if Nx>=3: self.microscope.move_stage(x=self.stack_settings.dx, move_type="Relative")
                for jj in range(Ny):
                    self.label_y.setText(f"{jj + 1} of")
                    if Ny>=3: self.microscope.move_stage(y=self.stack_settings.dy, move_type="Relative")
                    for kk in range(Nz):
                        self.label_z.setText(f"{kk + 1} of")
                        if Nz>=3: self.microscope.move_stage(z=self.stack_settings.dz, move_type="Relative")

                        """ ---------------- tilt scan ---------------- """
                        for oo in range(Nt):
                            self.label_t.setText(f"{oo + 1} of")
                            if Nt>=2:
                                new_tilt = t0 + oo * self.stack_settings.dt
                                self.microscope.tilt_stage(t=new_tilt, move_type="Absolute")

                            """ ---------------- rotation scan ---------------- """
                            for qq in range(Nr):
                                self.label_r.setText(f"{qq + 1} of")
                                if Nr>=2:
                                    new_rot_angle = r0 + qq * self.stack_settings.dr
                                    self.microscope.rotate_stage(r=new_rot_angle, move_type="Absolute")

                                    ''' correct stage rotation with scan rotation for image alignment
                                        stage rotated total by dr,
                                        image rotated by dr, scanning must be rotated relative by -dr
                                        scan rotation = initial_scan_rotation - dr 
                                    '''
                                    if self.checkBox_scan_rotation_correction.isChecked():
                                        stage_rotated_by = qq * self.stack_settings.dr
                                        correction_scan_rotation = scan_rot0 - stage_rotated_by
                                        self.microscope.set_scan_rotation(rotation_angle=correction_scan_rotation,
                                                                          type="Absolute")

                                # After all the movements are done, take images, save images, etc
                                self.microscope._get_current_microscope_state()

                                file_name = '%06d_' % counter + sample_name + '_' + \
                                            utils.current_timestamp() + '.tif'

                                ''' In the beginning of the loop take two reference images
                                    "highres": at the same field width as the current scan for beamshift alignment
                                    "lowres": lower magnification image for stage shift alignment, currently x2 larger hfw
                                '''
                                if qq==0: # take the reference images in the beginning
                                    # TODO better decision for the lowres field width (currently x2)
                                    hfw_highres = gui_settings["imaging"]["horizontal_field_width"]
                                    hfw_lowres  = 2 * hfw_highres
                                    #ref_image = utils.make_copy_of_Adorned_image(image) #copy the aligned image for the next alignment
                                    ref_image_lowres, ref_image_highres = \
                                        self.microscope.acquire_low_and_high_res_reference_images(gui_settings=gui_settings,
                                                                                                  hfw_highres=hfw_highres,
                                                                                                  hfw_lowres=hfw_lowres)

                                """ Correct the drift by stage movement. No need to correct the first image """
                                if self.checkBox_stage_drift_correction.isChecked() and counter>0:
                                    # grab a low res image
                                    image_low_res = self.acquire_image(hfw=hfw_lowres)
                                    # find shift and execute stage correction
                                    _ = self.microscope.stage_shift_alignment(ref_image=ref_image_lowres,
                                                                              image=image_low_res,
                                                                              mode='crosscorrelation')
                                    image_low_res = self.acquire_image() # take coarsely aligned image


                                """ Correct the drift by beam shift. No need to correct the first image """
                                if self.checkBox_beam_shift_drift_correction.isChecked() and counter>0:
                                    # grab an image at the "highres" field width setting, default setting in the gui_setting
                                    image = self.acquire_image(hfw=hfw_highres)
                                    # find shift and execute beam shift correction
                                    _ = self.microscope.beam_shift_alignment(ref_image=ref_image_highres,
                                                                             image=image,
                                                                             mode='crosscorrelation')

                                # take finely aligned image or if the alignment option is disabled, simply take an image
                                image = self.acquire_image()

                                utils.save_image(image, path=self.stack_dir, file_name=file_name)

                                self.experiment_data = utils.populate_experiment_data_frame(
                                    data_frame=self.experiment_data,
                                    microscope_state=self.microscope.microscope_state,
                                    file_name=file_name,
                                    timestamp=utils.current_timestamp(),
                                    keys=keys
                                )

                                self.repaint()  # update the GUI to show the progress
                                QtWidgets.QApplication.processEvents()

                                if self._abort_clicked_status == True:
                                    print('Abort clicked')
                                    self._abort_clicked_status = False  # reinitialise back to False
                                    return

                                """update gui settings manually if the scan is long"""
                                gui_settings = self.create_settings_dict()
                                """update the autocontrast setting for stack"""
                                gui_settings["imaging"]["autocontrast"] = self.checkBox_autocontrast_stack.isChecked()

                                counter += 1

        _run_loop(gui_settings)
        self.pushButton_acquire.setEnabled(True)
        self.pushButton_collect_stack.setEnabled(True)
        self.pushButton_estimate_stack_time.setEnabled(True)
        self.pushButton_abort_stack_collection.setEnabled(False)

        print('End of long scan, returning to the stored microscope state', stored_microscope_state)
        utils.save_data_frame(data_frame=self.experiment_data,
                              path=self.stack_dir,
                              file_name='summary'
                              )
        #self.microscope._restore_microscope_state(state=stored_microscope_state)


    def _abort_clicked(self):
        print('------------ abort clicked --------------')
        self.pushButton_abort_stack_collection.setEnabled(False)
        self._abort_clicked_status = True




    def _update_stack_settings(self):
        print('uploading stack settings...')
        self.stack_settings.x_range = self.doubleSpinBox_x_range.value() * 1e-6
        self.stack_settings.dx = self.doubleSpinBox_x_step.value() * 1e-6
        self.stack_settings.y_range = self.doubleSpinBox_y_range.value() * 1e-6
        self.stack_settings.dy = self.doubleSpinBox_y_step.value() * 1e-6
        self.stack_settings.z_range = self.doubleSpinBox_z_range.value() * 1e-6
        self.stack_settings.dz = self.doubleSpinBox_z_step.value() * 1e-6
        self.stack_settings.t_range = np.deg2rad(self.doubleSpinBox_t_range.value())
        self.stack_settings.dt = np.deg2rad(self.doubleSpinBox_t_step.value())
        self.stack_settings.r_range = np.deg2rad(self.doubleSpinBox_r_range.value())
        self.stack_settings.dr = np.deg2rad(self.doubleSpinBox_r_step.value())

        ranges = [self.stack_settings.x_range, self.stack_settings.y_range, self.stack_settings.z_range,
                  self.stack_settings.t_range, self.stack_settings.r_range]
        deltas = [self.stack_settings.dx, self.stack_settings.dy, self.stack_settings.dz,
                  self.stack_settings.dt, self.stack_settings.dr]
        NN = [self.stack_settings.Nx, self.stack_settings.Ny, self.stack_settings.Nz,
              self.stack_settings.Nt, self.stack_settings.Nr]

        for i in range(len(ranges)):
            if ranges[i] == 0 or deltas[i] == 0:
                NN[i] = 1
            elif i <= 2:
                NN[i] = int(2 * ranges[i] / deltas[i]) + 1
            else:
                NN[i] = int(ranges[i] / deltas[i]) + 1

        StackSettings.update_numbers_in_stack(self.stack_settings, NN)

        self.spinBox_Nx.setValue(self.stack_settings.Nx)
        self.spinBox_Ny.setValue(self.stack_settings.Ny)
        self.spinBox_Nz.setValue(self.stack_settings.Nz)
        self.spinBox_Nt.setValue(self.stack_settings.Nt)
        self.spinBox_Nr.setValue(self.stack_settings.Nr)


    def _change_of_move_type(self):
        move_type = self.comboBox_move_type.currentText()
        if move_type == "Relative":
            coords = [0, 0, 0, 0, 0]
        elif move_type == "Absolute":
            coords = self.microscope.update_stage_position()
        else:
            return

        self.doubleSpinBox_stage_x.setValue(coords[0] / 1e-6)
        self.doubleSpinBox_stage_y.setValue(coords[1] / 1e-6)
        self.doubleSpinBox_stage_z.setValue(coords[2] / 1e-6)
        self.doubleSpinBox_stage_t.setValue(np.rad2deg(coords[3]))
        self.doubleSpinBox_stage_r.setValue(np.rad2deg(coords[4]))


    def _open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "QFileDialog.getOpenFileName()",
                                                   "", "TIF files (*.tif);;TIFF files (*.tiff);;All Files (*)",
                                                   options=options)
        if file_name:
            print(file_name)
            # data format files saved by PiXet detector
            if file_name.lower().endswith('.tif') or file_name.lower().endswith('.tiff'):
                self.image = correlation.load_image(file_name)
                self.update_display(image=self.image)

            # other file format, not tiff, for example numpy array data, or txt format
            else:
                try:
                    self.image = np.loadtxt(file_name)
                    self.update_image(image=self.image)
                except:
                    self.label_messages.setText('File or mode not supported')


    def _save_SEM_image(self):
        if self.current_image is not None:
            utils.save_image(self.current_image)


    def setup_EBSD_detector(self):
        convention = self.comboBox_convention.currentText()
        Nx = self.spinBox_chip_pixels_x.value()
        Ny = self.spinBox_chip_pixels_y.value()
        shape = (Nx, Ny)
        pc_x = self.doubleSpinBox_pc_x.value()
        pc_y = self.doubleSpinBox_pc_y.value()
        pc_z = self.doubleSpinBox_pc_x.value()
        px_size = self.doubleSpinBox_detector_pixel_size.value()
        binning = self.spinBox_binning.value()
        tilt = self.doubleSpinBox_detector_tilt.value()
        sample_tilt = self.doubleSpinBox_sample_tilt.value()
        print(convention, shape, (pc_x,pc_y,pc_z), px_size, binning, tilt, sample_tilt)
        self.detectorEBSD = kp.detectors.EBSDDetector(
            shape=shape,
            pc=[pc_x, pc_y, pc_z],
            convention=convention,
            px_size=px_size,  # microns
            binning=binning,
            tilt=tilt,
            sample_tilt=sample_tilt
        )
        print(self.detectorEBSD)
        self.pushButton_set_EBSD_detector.setStyleSheet("background-color: green")



    def _open_EBSD_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "QFileDialog.getOpenFileName()",
                                                   "", "PMF files (*.pmf);;h5 files (*.h5);;All Files (*)",
                                                   options=options)
        if file_name:
            print(file_name)
            #################################################
            ################ data format pmf ################
            #################################################
            # data format files saved by PiXet detector
            if file_name.lower().endswith('.pmf'):
                metadata_file_name = file_name + '.dsc'
                ebsd_data = np.loadtxt(file_name)
                # self.update_display(image=ebsd_data)
                window = ui_utils.display_image(self, ebsd_data)
                window.show()
                # try:
                #     self.detectorEBSD.plot(pattern=ebsd_data)
                #     plt.show()
                # except:
                #     self.label_messages.setText('EBSD detector is not set')


            ################################################
            ################ data format h5 ################
            ################################################
            elif file_name.lower().endswith('.h5'):
                with h5py.File(file_name, 'r') as f:
                    acqTime = list(f['Frame_0']['MetaData']['Acq time'])
                    threshold = list(f['Frame_0']['MetaData']['Threshold'])
                    ebsd_data = f['Frame_0']['Data']
                    ebsd_data = np.reshape(ebsd_data, (256, 256))
                    print('acq time = ', acqTime, 'threshold = ', threshold)
                # self.update_display(image=ebsd_data)
                window = ui_utils.display_image(self, ebsd_data)
                window.show()


            #####################################################
            ################ data format unknown ################
            #####################################################
            # for example numpy array data, or txt format
            else:
                try:
                    ebsd_data = np.loadtxt(file_name)
                    # self.update_display(image=ebsd_data)
                    window = ui_utils.display_image(self, image=ebsd_data)
                    window.show()
                except:
                    self.label_messages.setText('File or mode not supported')

                # try:
                #     self.detectorEBSD.plot(pattern=ebsd_data)
                #     plt.show()
                # except:
                #     self.label_messages.setText('EBSD detector is not set')

            # except:
            #     print('Could not read the file, or something else is wrong')



    def _open_EBSD_stack(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "QFileDialog.getOpenFileName()",
                                                   "", "PMF files (*.pmf);;h5 files (*.h5);;hspy files (*.hspy);;All Files (*)",
                                                   options=options)
        if file_name:
            print(file_name)
            file_dir = os.path.dirname(file_name)

            #################################################
            ################ data format pmf ################
            #################################################
            # data format files saved by PiXet detector
            if file_name.lower().endswith('.pmf'):
                try:
                    if '_ToA' in file_name:
                        selected_mode = 'TOA'
                        files = sorted(glob.glob(file_dir + '/' + '*_ToA.pmf'))
                    elif '_ToT' in file_name:
                        selected_mode = 'TOT'
                        files = sorted(glob.glob(file_dir + '/' + '*_ToT.pmf'))
                    elif '_Event' in file_name:
                        selected_mode = 'EVENT'
                        files = sorted(glob.glob(file_dir + '/' + '*_Event.pmf'))
                    elif '_iToT' in file_name:
                        selected_mode = 'iTOT'
                        files = sorted(glob.glob(file_dir + '/' + '*_iTOT.pmf'))
                    else:
                        print('File or mode not supported')

                    Nx, Ny = utils.get_Nx_Ny_from_indices(file_dir, files)

                    self.ebsd_stack = hs.signals.Signal2D(np.zeros((Nx,Ny, 256, 256)))

                    for file_name in files:
                        ebsd_data = np.loadtxt(file_name)
                        ii, jj = utils.get_indices_from_file_name(file_dir, file_name)
                        self.ebsd_stack.data[ii][jj] = ebsd_data

                    # convert hs.signals.Signal2D BaseSignal to EBSD (EBSDMasterPattern or VirtualBSEImage)
                    self.ebsd_stack.set_signal_type("EBSD")

                    # TODO static background file or metadata needed to perform this operation
                    # static_bg (Union[None, ndarray, Array]) – Static background pattern. If None is passed (default) we try to read it from the signal metadata
                    # if self.checkBox_remove_static_background.isChecked():
                    #     print('removing the static background')
                    #     self.storage_dict[selected_mode].stack.remove_static_background(operation="subtract",
                    #                                                                     static_bg= ???
                    #                                                                     relative=True)
                    if self.checkBox_remove_dynamic_background.isChecked():
                        print('removing the dynamic background')
                        self.ebsd_stack.remove_dynamic_background(operation="subtract",  # Default
                                                                  filter_domain="frequency",  # Default
                                                                  std=8,  # Default is 1/8 of the pattern
                                                                  truncate=4)
                    #self.ebsd_stack.plot()
                    #plt.show()
                    window = ui_utils.display_stack(self, ebsd_stack=self.ebsd_stack)
                    window.show()


                except:
                    print('Could not read the file, or something else is wrong')

            ################################################
            ################ data format h5 ################
            ################################################
            if file_name.lower().endswith('.h5'):
                try:
                    stack = hs.load(file_name)
                    stack.plot()
                    plt.show()
                except:
                    print('Could not read the file, or something else is wrong')


            ################################################
            ################ data format hspy ################
            ################################################
            if file_name.lower().endswith('.hspy'):
                try:
                    stack = hs.load(file_name)
                    stack.plot()
                    plt.show()
                except:
                    print('Could not read the file, or something else is wrong')


    def _apply_clahe(self):
        if self.image is not None:
            clipLimit = int(self.spinBox_clip_limit.value())
            tileGridSize = int(self.spinBox_tile_grid_size.value())
            self.image_mod = utils.enhance_contrast(self.image,
                                                    clipLimit=clipLimit,
                                                    tileGridSize=tileGridSize)
            self.update_display(self.image_mod)


    def _restore_image(self):
        if self.image is not None:
            self.update_display(self.image)


    def disconnect(self):
        # logging.info("Running cleanup/teardown")
        # logging.debug("Running cleanup/teardown")
        print('closing down, cleaning...')
        if self.microscope:
            self.microscope.disconnect()











    def run_correlation(self):
        print('transformation_type = ', self.comboBox_tranformation_type.currentText())
        if self.corr_image_1_path is not None:
            if self.corr_image_2_path is not None:
                window = correlation.open_correlation_window(self,
                                                             self.corr_image_1_path,
                                                             self.corr_image_2_path,
                                                             output_path=None,
                                                             transformation_type=self.comboBox_tranformation_type.currentText())
                window.showMaximized()
                window.show()
                window.exitButton.clicked.connect(lambda: self.correlation_complete(window))


    def correlation_complete(self, window):
        overlayed_image, transformation = window.menu_quit()
        print("Correlation complete, overlayed image dims = ", overlayed_image.shape)
        print(f'transformation = {transformation}')

        theta00 = +np.arccos(transformation.params[0, 0])
        theta01 = -np.arcsin(transformation.params[0, 1])
        theta10 = +np.arccos(transformation.params[1, 0])
        theta11 = +np.arcsin(transformation.params[0, 1])

        shift_x = transformation.params[0, 2]
        shift_y = transformation.params[1, 2]
        print(f'rotation angle = {np.rad2deg(theta00)}, shift_x = {shift_x}, shift_y = {shift_y}')
        self.label_messages.setText(str(f'rotation angle = {np.rad2deg(theta00)}, shift_x = {shift_x}, shift_y = {shift_y}'))


        self.update_display(image=overlayed_image)
        self.label_transformation.setText(str( str(transformation) ) )


    def _select_corr_image(self, num=1):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "QFileDialog.getOpenFileName()",
                                                   "", "TIF files (*.tif);;TIFF files (*.tiff);;PNG files (*.png);;All Files (*)",
                                                   options=options)
        if file_name:
            print(file_name)
            if num==1:
                self.corr_image_1_path = file_name
                self.label_corr_image_1_path.setText(str(self.corr_image_1_path))
                image_to_preview = correlation.load_image(self.corr_image_1_path)
                image_to_preview = utils.resize(image_to_preview, size=(200,200))
                image_to_preview = qimage2ndarray.array2qimage(image_to_preview.copy())
                self.label_image_corr_1.setPixmap(QtGui.QPixmap(image_to_preview))
            elif num==2:
                self.corr_image_2_path = file_name
                self.label_corr_image_2_path.setText(str(self.corr_image_2_path))
                image_to_preview = correlation.load_image(self.corr_image_2_path)
                image_to_preview = utils.resize(image_to_preview, size=(200,200))
                image_to_preview = qimage2ndarray.array2qimage(image_to_preview.copy())
                self.label_image_corr_2.setPixmap(QtGui.QPixmap(image_to_preview))





    def test_correlation(self):
        image_1 = "01_tilted.tif"
        image_2 = "02_flat_shifted.tif"
        #image_2 = "02_flat.tif"

        window = correlation.open_correlation_window(
            self, image_1, image_2, output_path=None
        )
        window.showMaximized()
        window.show()
        window.exitButton.clicked.connect(lambda: self.correlation_complete(window))

    def test_open_window(self):
        image_1 = "01_tilted.tif"
        image = plt.imread(image_1)
        window = ui_utils.display_image(self, image)
        window.show()










def main(demo):
    app = QtWidgets.QApplication([])
    qt_app = GUIMainWindow(demo)
    app.aboutToQuit.connect(qt_app.disconnect)  # cleanup & teardown
    qt_app.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(demo=False)
    stack_settings = StackSettings()
    # stack_settings.x_range = 100
