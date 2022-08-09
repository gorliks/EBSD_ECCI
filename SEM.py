import os

try:
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import (AdornedImage,
                                                             GrabFrameSettings,
                                                             Rectangle,
                                                             RunAutoCbSettings)
except:
    print('Autoscript module not found')

import numpy as np
from dataclasses import dataclass
import movement
import utils
from main import BeamType


@dataclass
class MicroscopeState:
    hv : float = 20
    beam_current : float = 0
    x : float = 0
    y : float = 0
    z : float = 0
    t : float = 0
    r : float = 0
    horizontal_field_width : float = 0
    resolution : str = '768x512'
    detector : str = 'ETD'
    rotate_compucentric : bool = True
    scan_rotation_angle : float = 0
    brighness : float = 0
    contrast : float = 0

    def __to__dict__(self) -> dict:
        state_dict = {
            "x" : self.x,
            "y" : self.y,
            "z" : self.z,
            "r" : self.r,
            "t" : self.t,
            "rotate_compucentric" : self.rotate_compucentric,
            "horizontal_field_width" : self.horizontal_field_width,
            "detector" : self.detector,
            "scan_rotation_angle" : self.scan_rotation_angle,
            "brighness" : self.brighness,
            "contrast" : self.contrast
        }
        return state_dict

    def get_stage_position(self):
        coords_dict = {
            "x" : self.x,
            "y" : self.y,
            "z" : self.z,
            "r" : self.r,
            "t" : self.t,
            "rotate_compucentric" : self.rotate_compucentric
        }
        return coords_dict

    def update_stage_position(self, x=0, y=0, z=0, r=0, t=0):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.t = t


@dataclass
class ImageSettings:
    resolution: str
    dwell_time: float
    horizontal_field_width: float
    autocontrast: bool
    beam_type: BeamType
    save: bool
    sample_name: str
    path : str
    quadrant: int
    bit_depth: int


class Microscope():
    def __init__(self, settings: dict = None, log_path: str = None,
                 ip_address: str = "10.0.0.1", demo: bool=True):
        self.settings = settings
        self.demo = demo
        self.ip_address = ip_address
        self.log_path = log_path
        self.microscope_state = MicroscopeState()
        self.stored_state = MicroscopeState()

        print('image settings: ', self.settings)

        try:
            print('initialising microscope')
            self.microscope = SdbMicroscopeClient()
            return self.microscope
        except:
            print('Autoscript not installed on the computer, using demo mode')
            self.demo = True
            self.microscope = ['no autoscript found']


    def establish_connection(self):
        """Connect to the SEM microscope."""
        try:
            # TODO: get the port
            print('connecting to microscope...')
            #logging.info(f"Microscope client connecting to [{ip_address}]")
            self.microscope = SdbMicroscopeClient()
            self.microscope.connect(self.ip_address)
            #logging.info(f"Microscope client connected to [{ip_address}]")
        except Exception as e:
            print(f"AutoLiftout is unavailable. Unable to connect to microscope: {e}")
            self.microscope = ['could not connect to microscope']

        return self.microscope


    def autocontrast(self, quadrant: int = 1) -> None:
        """Automatically adjust the microscope image contrast."""
        if not self.demo:
            self.microscope.imaging.set_active_view(quadrant)
            RunAutoCbSettings(
                method="MaxContrast",
                resolution="768x512",  # low resolution, so as not to damage the sample
                number_of_frames=5,
            )
            #logging.info("automatically adjusting contrast...")
            self.microscope.auto_functions.run_auto_cb()
        else:
            print('demo: automatically adjusting contrast...')


    def acquire_image(self, gui_settings: dict):
        """Take new electron or ion beam image.
        Returns
        -------
        AdornedImage
            If the returned AdornedImage is named 'image', then:
            image.data = a numpy array of the image pixels
            image.metadata.binary_result.pixel_size.x = image pixel size in x
            image.metadata.binary_result.pixel_size.y = image pixel size in y
        """
        print('acquiring image...')
        # logging.info(f"acquiring new {beam_type.name} image.")

        if not self.demo:
            if gui_settings is not None:
                settings = self.update_image_settings(gui_settings)
                self.microscope.imaging.set_active_view(settings.quadrant)
                self.microscope.beams.electron_beam.horizontal_field_width.value = \
                    settings.horizontal_field_width
                if settings["autocontrast"]==True:
                    self.autocontrast(quadrant=settings.quadrant)

                grab_frame_settings = GrabFrameSettings(resolution=settings.resolution,
                                                        dwell_time=settings.dwell_time,
                                                        bit_depth=settings.bit_depth
                                                        )
                image = self.microscope.imaging.grab_frame(grab_frame_settings)
            else:
                image = self.microscope.imaging.grab_frame()
            return image

        else:
            print('demo mode')
            if gui_settings is not None:
                settings = self.update_image_settings(gui_settings)
                resolution = settings.resolution
                print("resolution = ", resolution)
                [width, height] = np.array(resolution.split("x")).astype(int)
            else:
                height, width = 768, 512
            simulated_image = np.random.randint(0, 255, [height,width])
            return simulated_image


    def last_image(self,  quadrant : int, save : bool = False):
        """Get the last previously acquired ion or electron beam image.
        Parameters
        ----------
        microscope : Autoscript microscope object.
        beam_type :

        Returns
        -------
        AdornedImage
            If the returned AdornedImage is named 'image', then:
            image.data = a numpy array of the image pixels
            image.metadata.binary_result.pixel_size.x = image pixel size in x
            image.metadata.binary_result.pixel_size.y = image pixel size in y
        """
        if not self.demo:
            self.microscope.imaging.set_active_view(quadrant)
            image = microscope.imaging.get_image()
            return image
        else:
            simulated_image = np.random.randint(0, 255, [512,768])
            print(simulated_image.shape)
            return simulated_image


    def update_stage_position(self):
        try:
            current_position = \
                self.microscope.specimen.stage.current_position
            return current_position
        except:
            print('demo mode, simulated coordinates are:')
            #[x, y, z, t, r] = np.random.randint( 0,5, [5,1] ).astype(float)
            [x, y, z, t, r] = np.random.rand(5, 1)
            MicroscopeState.update_stage_position(self.microscope_state,
                                                  x=x[0], y=y[0], z=z[0], t=t[0], r=r[0])
            print(MicroscopeState.get_stage_position(self.microscope_state))
            return (x[0], y[0], z[0], t[0], r[0])


    def move_stage(self, x=0, y=0, z=0, r=0, t=0,
                   compucentric=True,
                   move_type='Relative'):
        """ Two types of movement: relative and absolute
            Two coordinate systems: StagePosition(t=position.t, coordinate_system=position.coordinate_system)
            SPECIMEN: based on position of the specimen (link between Z and working distance)
            RAW: coordinate system based solely on stage hardware encoders
        """
        if move_type=='Relative':
            print('SEM: moving relative')
            new_position = \
                movement.move_relative(self.microscope,
                                   dx=x, dy=y, dz=z,
                                   dr=r, dt=t)
            x, y, z, r, t = new_position[0], new_position[1], new_position[2], new_position[3], new_position[4]
            MicroscopeState.update_stage_position(self.microscope_state,
                                                  x=x, y=y, z=z, r=r, t=t)
            print('simulated coords = ', MicroscopeState.get_stage_position(self.microscope_state))


        if move_type=="Absolute":
            print("moving SEM stage: absolute")
            new_position = \
                movement.move_absolute(self.microscope,
                                       x=x, y=y, z=z,
                                       t=t, r=r)
            x, y, z, r, t = new_position[0], new_position[1], new_position[2], new_position[3], new_position[4]
            MicroscopeState.update_stage_position(self.microscope_state,
                                                  x=x, y=y, z=z, r=r, t=t)
            print('simulated coords = ', MicroscopeState.get_stage_position(self.microscope_state))



    def set_scan_rotation(self, rotation_angle : float = 0, type="Absolute") -> float:
        """Set scan rotation angle
        Args:
            rotation_angle (float): angle of scan rotation in degrees,
            needs conversion to rad
        Returns
        -------
        float: system-level scan rotation angle in degrees
        """
        rotation_angle_rad = np.deg2rad(rotation_angle)

        if type=="Relative":
            """Change the current scan rotation by the specified value"""
            self.microscope.beams.ion_beam.scanning.rotation.value += rotation_angle_rad
        else:
            """Absolute value of the scan rotation"""
            self.microscope.beams.ion_beam.scanning.rotation.value = rotation_angle_rad

        return np.rad2deg(self.microscope.beams.ion_beam.scanning.rotation.value)


    def _store_current_microscope_state(self):
        """Acquires the current microscope state to store if necessary to return to this initial state
         Stores the state in MicroscopeState dataclass variable
        Args:
            None
        Returns
        -------
        None
        """
        try:
            position = self.update_stage_position()
            self.stored_state.x = position.x
            self.stored_state.y = position.y
            self.stored_state.z = position.z
            self.stored_state.t = position.t
            self.stored_state.r = position.r
            self.stored_state.horizontal_field_width = \
                self.microscope.beams.electron_beam.horizontal_field_width.value
            self.stored_state.resolution = self.microscope.beams.electron_beam.scanning.resolution
            self.stored_state.hv = self.microscope.beams.electron_beam.high_voltage.value
            self.stored_state.scan_rotation_angle = \
                self.microscope.beams.electron_beam.scanning.rotation.value
            self.stored_state.brightness = self.microscope.detector.brightness.value
            self.stored_state.contrast = self.microscope.detector.contrast.value
        except:
            self.stored_state.x = 2
            self.stored_state.y = 1
            self.stored_state.z = 0
            self.stored_state.t = 0
            self.stored_state.r = 0
            self.stored_state.scan_rotation_angle = 0



    def update_image_settings(self,
                              gui_settings: dict,
                              resolution=None,
                              dwell_time=None,
                              horizontal_field_width=None,
                              autocontrast=None,
                              beam_type=None,
                              quadrant=None,
                              save=None,
                              sample_name=None,
                              path=None,
                              bit_depth=None,
                              ):
        """Update image settings. Uses default values if not supplied
        Args:
            settings (dict): the settings dictionary from GUI
            resolution (str, optional): image resolution. Defaults to None.
            dwell_time (float, optional): image dwell time. Defaults to None.
            hfw (float, optional): image horizontal field width. Defaults to None.
            autocontrast (bool, optional): use autocontrast. Defaults to None.
            beam_type (BeamType, optional): beam type to image with (Electron, Ion). Defaults to None.
            gamma (GammaSettings, optional): gamma correction settings. Defaults to None.
            save (bool, optional): save the image. Defaults to None.
            label (str, optional): image filename . Defaults to None.
            save_path (Path, optional): directory to save image. Defaults to None.
        """

        # new image_settings
        if resolution:
            self.resolution = resolution
        else:
            self.resolution = gui_settings["imaging"]["resolution"]

        if dwell_time:
            self.dwell_time = dwell_time
        else:
            self.dwell_time = gui_settings["imaging"]["dwell_time"]

        if horizontal_field_width:
            self.horizontal_field_width = horizontal_field_width
        else:
            self.horizontal_field_width = gui_settings["imaging"]["horizontal_field_width"]

        if autocontrast:
            self.autocontrast = autocontrast
        else:
            self.autocontrast = gui_settings["imaging"]["autocontrast"]

        if autocontrast:
            self.autocontrast = autocontrast
        else:
            self.autocontrast = gui_settings["imaging"]["autocontrast"]

        if beam_type:
            self.beam_type = beam_type
        else:
            self.beam_type = gui_settings["imaging"]["beam_type"]

        if quadrant:
            self.quadrant = quadrant
        else:
            self.quadrant = gui_settings["imaging"]["quadrant"]

        if save:
            self.save = save
        else:
            self.save = gui_settings["imaging"]["save"]

        if path:
            print("<-------->", path)
            self.path = path
        else:
            self.path = gui_settings["imaging"]["path"]
            print("-------->", self.path)

        if bit_depth:
            self.bit_depth = bit_depth
        else:
            self.bit_depth = gui_settings["imaging"]["bit_depth"]

        if sample_name:
            self.sample_name = sample_name
        else:
            self.sample_name = gui_settings["imaging"]["sample_name"]

        self.image_settings = ImageSettings(
            resolution=self.resolution,
            dwell_time=self.dwell_time,
            horizontal_field_width=self.horizontal_field_width,
            quadrant=self.quadrant,
            autocontrast=self.autocontrast,
            beam_type=BeamType.ELECTRON if beam_type is None else beam_type,
            save=self.save,
            path=self.path,
            sample_name=self.sample_name,
            bit_depth=self.bit_depth,
        )

        return self.image_settings


if __name__ == '__main__':
    microscope = Microscope()
    microscope.establish_connection()
    print('current state: ', microscope.microscope_state)
    print('update stage position')
    microscope.update_stage_position()
    print('current state: ', microscope.microscope_state)
    print('x = ', microscope.microscope_state.x)

    gui_settings = {
        "imaging": {
            "resolution": "758x512",
            "horizontal_field_width": 100e-6,
            "dwell_time": 1e-6,
            "detector": "ABS",
            "autocontrast": True,
            "save": True,
            "beam_type": BeamType.ELECTRON,
            "quadrant": 1,
            "label": "sample_name",
            "path": "No path found",
            "bit_depth": 16,
            "sample_name" : "silicon001"
        }
    }
    image_settings = microscope.update_image_settings(gui_settings=gui_settings)
    print(image_settings)

    microscope.save_image(image=1)


