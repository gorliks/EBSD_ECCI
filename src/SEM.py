try:
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import (AdornedImage,
                                                             GrabFrameSettings,
                                                             Rectangle,
                                                             RunAutoCbSettings,
                                                             Point,
                                                             MoveSettings,
                                                             StagePosition)
    from autoscript_sdb_microscope_client.enumerations import (
                                                        CoordinateSystem)
except:
    print('Autoscript module not found')

import numpy as np
import movement
import utils
import correlation
from utils import MicroscopeState
from utils import ImageSettings

from importlib import reload  # Python 3.4+
reload(movement)
reload(correlation)
reload(utils)

from utils import BeamType

class Microscope():
    def __init__(self, settings: dict = None, log_path: str = None,
                 ip_address: str = "192.168.0.1", demo: bool=True):
        self.settings = settings
        self.demo = demo
        self.ip_address = ip_address
        self.log_path = log_path
        self.microscope_state = MicroscopeState()

        try:
            print('initialising microscope')
            self.microscope = SdbMicroscopeClient()
        except:
            print('Autoscript not installed on the computer, using demo mode')
            self.demo = True
            self.microscope = ['no Autoscript found']


    def establish_connection(self):
        """Connect to the SEM microscope."""
        try:
            # TODO: get the port
            print('connecting to microscope...')
            #logging.info(f"Microscope client connecting to [{ip_address}]")
            self.microscope = SdbMicroscopeClient()
            self.microscope.connect(self.ip_address)
            self.microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
            #logging.info(f"Microscope client connected to [{ip_address}]")
        except Exception as e:
            print(f"AutoLiftout is unavailable. Unable to connect to microscope: {e}")
            self.microscope = ['could not connect to microscope']

        return self.microscope


    def autocontrast(self, quadrant: int = 1) -> None:
        """Automatically adjust the microscope image contrast."""
        if not self.demo:
            self.microscope.imaging.set_active_view(quadrant)
            settings = RunAutoCbSettings(
                            method="MaxContrast",
                            resolution="768x512",  # low resolution, so as not to damage the sample
                            number_of_frames=5)
            #logging.info("automatically adjusting contrast...")
            self.microscope.auto_functions.run_auto_cb(settings)
        else:
            print('demo: automatically adjusting contrast...')


    def set_beam_point(self,
                       beam_x : int = 0,
                       beam_y : int = 0):
        """
        API requires:
        x : float,  X coordinate of the spot. The valid range of the coordinate is [0, 1].
        y : float,  Y coordinate of the spot. The valid range of the coordinate is [0, 1].
        ----------
        Parameters
        ----------
        beam_x : int coordinate in pixels
        beam_y : int coordinate in pixels
        convert to
        x, y = beam_x / pixels_in_x, beam_y / pixels_in_y

        Returns None
        -------
        """
        if not self.demo:
            """ current screen resolution """
            resolution = self.microscope.beams.electron_beam.scanning.resolution.value
            [width, height] = np.array(resolution.split("x")).astype(int)
            if beam_x > width  : beam_x = width
            if beam_y > height : beam_y = height
            x = float(beam_x) / float(width)
            y = float(beam_y) / float(height)
            self.microscope.beams.electron_beam.scanning.mode.set_spot(x, y)
        else:
            print(f'demo: setting beam spot coordinates to ({beam_x}, {beam_y})')


    def set_full_frame(self):
        if not self.demo:
            self.microscope.beams.electron_beam.scanning.mode.set_full_frame()
        elif self.demo:
            print('setting scanning mode to full frame...   ')


    def beam_blank(self):
        if not self.demo:
            _state = self.microscope.beams.electron_beam.is_blanked
            if _state == True:
                """ beam state is blanked, unblank it """
                self.microscope.beams.electron_beam.unblank()
            elif _state == False:
                """beam state is not blanked, blank it"""
                self.microscope.beams.electron_beam.blank()
            return self.microscope.beams.electron_beam.is_blanked

        elif self.demo:
            print('demo mode: beam blank/unblank function called...   ')
            return 'demo blank'


    def blank(self):
        if not self.demo:
            self.microscope.beams.electron_beam.blank()
            return self.microscope.beams.electron_beam.is_blanked

        elif self.demo:
            print('demo mode: beam blank...   ')
            return 'demo blank'

    def unblank(self):
        if not self.demo:
            self.microscope.beams.electron_beam.unblank()
            return self.microscope.beams.electron_beam.is_blanked

        elif self.demo:
            print('demo mode: beam unblank...   ')
            return 'demo blank'


    def acquire_image(self, gui_settings: dict,
                      hfw = None):
        """Take new electron or ion beam image.
        Returns
        -------
        AdornedImage
            If the returned AdornedImage is named 'image', then:
            image.data = a numpy array of the image pixels
            image.metadata.binary_result.pixel_size.x = image pixel size in x
            image.metadata.binary_result.pixel_size.y = image pixel size in y
            dwell_time in microseconds! x1e-6 coversion to seconds
        """
        print('acquiring image...')
        # logging.info(f"acquiring new {beam_type.name} image.")

        if not self.demo:
            if gui_settings is not None:
                settings = self.update_image_settings(gui_settings)
                self.microscope.imaging.set_active_view(settings.quadrant)

                if hfw is not None:
                    hfw = hfw
                else:
                    hfw = settings.horizontal_field_width

                if hfw > self.microscope.beams.electron_beam.horizontal_field_width.limits.max:
                    hfw = self.microscope.beams.electron_beam.horizontal_field_width.limits.max
                self.microscope.beams.electron_beam.horizontal_field_width.value = hfw

                if settings.autocontrast==True:
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
            print('demo mode   ')
            if gui_settings is not None:
                settings = self.update_image_settings(gui_settings)
                print('settings = ', settings)
                resolution = settings.resolution
                [width, height] = np.array(resolution.split("x")).astype(int)
            else:
                height, width = 768, 512
            simulated_image = np.random.randint(0, 255, [height,width])
            return simulated_image



    def acquire_low_and_high_res_reference_images(self,
                                                  gui_settings: dict,
                                                  image_highres=None,
                                                  hfw_lowres = 100e-6,
                                                  hfw_highres = 50e-6):
        if not self.demo:
            if gui_settings is not None:
                settings = self.update_image_settings(gui_settings)
                self.microscope.imaging.set_active_view(settings.quadrant)
                grab_frame_settings = GrabFrameSettings(resolution=settings.resolution,
                                                        dwell_time=settings.dwell_time,
                                                        bit_depth=settings.bit_depth)
            else:
                grab_frame_settings = GrabFrameSettings()

            if image_highres is not None:
                try:
                    pixel_size = image_highres.metadata.binary_result.pixel_size.x
                    hfw_highres = pixel_size * image_highres.width
                except Exception as e :
                    print(f'Cannot estimate the field width from the image supplied, error {e}')
                    hfw_highres = 50e-6

            if hfw_lowres > self.microscope.beams.electron_beam.horizontal_field_width.limits.max:
                hfw_lowres = self.microscope.beams.electron_beam.horizontal_field_width.limits.max
            if hfw_highres > self.microscope.beams.electron_beam.horizontal_field_width.limits.max:
                hfw_highres = self.microscope.beams.electron_beam.horizontal_field_width.limits.max

            self.microscope.beams.electron_beam.horizontal_field_width.value = hfw_lowres
            ref_image_lowres = self.microscope.imaging.grab_frame(grab_frame_settings)

            self.microscope.beams.electron_beam.horizontal_field_width.value = hfw_highres
            ref_image_highres = self.microscope.imaging.grab_frame(grab_frame_settings)


        elif self.demo:
            ref_image_lowres = np.random.randint(0, 255, [512,768])
            ref_image_highres = np.random.randint(0, 255, [512,768])

        return ref_image_lowres, ref_image_highres





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
            image = self.microscope.imaging.get_image()
            return image

        else:
            simulated_image = np.random.randint(0, 255, [512,768])
            print(simulated_image.shape)
            return simulated_image


    def update_stage_position(self):
        try:
            position = \
                self.microscope.specimen.stage.current_position
            MicroscopeState.update_stage_position(self.microscope_state,
                                                  x=position.x, y=position.y, z=position.z,
                                                  t=position.t, r=position.r)
            return (position.x, position.y, position.z,
                    position.t, position.r)

        except Exception as e:
            print(f'stage position, error {e}, simulated coordinates are:')
            #[x, y, z, t, r] = np.random.randint( 0,5, [5,1] ).astype(float)
            [x, y, z, t, r] = np.random.rand(5, 1)
            MicroscopeState.update_stage_position(self.microscope_state,
                                                  x=x[0]*1e-3, y=y[0]*1e-3, z=z[0]*1e-3,
                                                  t=t[0], r=r[0])
            print(MicroscopeState.get_stage_position(self.microscope_state))
            return (x[0]*1e-3, y[0]*1e-3, z[0]*1e-3, t[0], r[0])


    def move_stage(self, x=0, y=0, z=0,
                   move_type='Relative'):
        """ Two types of movement: relative and absolute
            Two coordinate systems: StagePosition(t=position.t, coordinate_system=position.coordinate_system)
            SPECIMEN: based on position of the specimen (link between Z and working distance)
            RAW: coordinate system based solely on stage hardware encoders
        """
        print(f"moving SEM stage: {move_type}")
        new_position = \
            movement.move_stage(self.microscope,
                                x=x, y=y, z=z,
                                move_type=move_type)

        x, y, z, r, t = new_position[0], new_position[1], new_position[2], \
                        new_position[3], new_position[4]

        MicroscopeState.update_stage_position(self.microscope_state,
                                              x=x, y=y, z=z, r=r, t=t)
        print('coords = ', MicroscopeState.get_stage_position(self.microscope_state))


    def move_stage_corrected(self,
                             expected_x : float = 0,
                             expected_y : float = 0) -> None:
        """Stage movement in X, Y, corrected for stage_tilt and scan_rotation
        ----------
        expected_x : in meters
        expected_y : in meters
        Returns
        -------
            None
        """
        self._get_current_microscope_state()  # update the microscope state to refresh the current stage tilt
        stage_tilt = self.microscope_state.t # tilt in radians
        scan_rotation = self.microscope_state.scan_rotation_angle # scan rotation in radians

        """ First correct the movement for the stage tilt """
        x_corrected = expected_x #x-axis does not need correction for stage tilt """
        y_corrected = +np.cos(stage_tilt) * expected_y
        z_corrected = 0#-np.sin(stage_tilt) * expected_y # TODO verify that z movements are safe!

        """ Correct the (x,y) movement for the scan rotation """
        (x_corrected, y_corrected) = utils.rotate_coordinate_in_xy(coord=(x_corrected, y_corrected),
                                                                   angle=-scan_rotation)
        """ perform movement """
        stage_settings = MoveSettings(rotate_compucentric=True)
        self.microscope.specimen.stage.relative_move(StagePosition(x=x_corrected, y=y_corrected),
                                                stage_settings)
        # self.move_stage(x=x_corrected,
        #                 y=y_corrected,
        #                 move_type="Relative")



    def rotate_stage(self, r=0,  move_type='Relative'):
        """ Two types of movement: relative and absolute
            Two coordinate systems: StagePosition(t=position.t, coordinate_system=position.coordinate_system)
            SPECIMEN: based on position of the specimen (link between Z and working distance)
            RAW: coordinate system based solely on stage hardware encoders
        """
        print(f'SEM: rotating stage {move_type}')
        new_position = \
            movement.rotate_stage(self.microscope,
                                  r=r,
                                  move_type=move_type)
        x, y, z, r, t = new_position[0], new_position[1], new_position[2], \
                        new_position[3], new_position[4]
        MicroscopeState.update_stage_position(self.microscope_state,
                                              x=x, y=y, z=z, r=r, t=t)
        print('coords = ', MicroscopeState.get_stage_position(self.microscope_state))


    def tilt_stage(self, t=0,  move_type='Relative'):
        """ Two types of movement: relative and absolute
            Two coordinate systems: StagePosition(t=position.t, coordinate_system=position.coordinate_system)
            SPECIMEN: based on position of the specimen (link between Z and working distance)
            RAW: coordinate system based solely on stage hardware encoders
        """
        print(f'SEM: tilting stage {move_type}')
        new_position = \
            movement.tilt_stage(self.microscope,
                                t=t,
                                move_type=move_type)
        x, y, z, r, t = new_position[0], new_position[1], new_position[2], \
                        new_position[3], new_position[4]
        MicroscopeState.update_stage_position(self.microscope_state,
                                              x=x, y=y, z=z, r=r, t=t)
        print('coords = ', MicroscopeState.get_stage_position(self.microscope_state))


    def set_scan_rotation(self, rotation_angle : float = 0, type="Absolute") -> float:
        """Set scan rotation angle
        Args:
            rotation_angle (float): angle of scan rotation in degrees,
            needs conversion to rad
        Returns
        -------
        float: system-level scan rotation angle in degrees
        """
        try:
            rot_min = self.microscope.beams.electron_beam.scanning.rotation.limits.min
            rot_max = self.microscope.beams.electron_beam.scanning.rotation.limits.max

            if type=="Relative":
                """
                    Change the current scan rotation by the specified value
                    Check that targety scan_rot does not exceed (-2pi, +2pi)
                    Otherwise divide module to stay within the (-2pi, +2pi) range
                """
                current_scan_rot = \
                    self.microscope.beams.electron_beam.scanning.rotation.value

                target_rot_angle = current_scan_rot + rotation_angle

                if target_rot_angle >=rot_max:
                    target_rot_angle = target_rot_angle % (2*np.pi)
                elif target_rot_angle <=rot_min:
                    target_rot_angle = target_rot_angle % (2*np.pi)

                print(f"setting scan rotation {type} to {np.rad2deg(target_rot_angle)}")
                # TODO backend from from frontend separation
                self.microscope.beams.electron_beam.scanning.rotation.value = target_rot_angle

            elif type=="Absolute":
                """Absolute value of the scan rotation"""
                if rotation_angle >=rot_max:
                    rotation_angle = rotation_angle % (2*np.pi)
                elif rotation_angle <=rot_min:
                    rotation_angle = rotation_angle % (2*np.pi)

                print(f"setting scan rotation {type} to {np.rad2deg(rotation_angle)}")
                # TODO backend from from frontend separation
                self.microscope.beams.electron_beam.scanning.rotation.value = rotation_angle

            self._get_current_microscope_state()
            return self.microscope.beams.electron_beam.scanning.rotation.value

        except Exception as e:
            print(f'Failed to set scan rotation {type} by {np.rad2deg(rotation_angle)} deg, error {e}')
            return rotation_angle


    def set_beam_shift(self,
                       beam_shift_x : float = 0.0,
                       beam_shift_y : float = 0.0) -> None:
        """Adjusting the beam shift
        Args:
            beam_shift_x: in metres, shift along x-axis
            beam_shift_y: in metres, shift along y-axis
        Returns
        -------
        None. Update the microscope state with the new beam shift values
        """
        # adjust beamshift
        try:
            self.microscope.beams.electron_beam.beam_shift.value = Point(beam_shift_x, beam_shift_y)
            self._get_current_microscope_state()
        except Exception as e:
            print(f"Could not apply beam shift, error {e}")


    def reset_beam_shifts(self):
        """Set the beam shift to zero for the electron beam
        Args:
            None
        """
        # logging.info(
        #     f"reseting ebeam shift to (0, 0) from: {microscope.beams.electron_beam.beam_shift.value} "
        # )
        try:
            print(f"reseting e-beam shift to (0, 0) from: {self.microscope.beams.electron_beam.beam_shift.value}")
            self.microscope.beams.electron_beam.beam_shift.value = Point(0, 0)
            print(f"reset beam shifts to zero complete")
        except Exception as e:
            print(f"Could not reset the beam shift, error {e}")
        self._get_current_microscope_state()
        # logging.info(f"reset beam shifts to zero complete")


    def stage_shift_alignment(self,
                              ref_image,
                              image,
                              mode='crosscorrelation') -> list:
        """Align the images by moving the stage
        Args:
            ref_image (AdornedImage): reference image to align to
            image: shifted image
            reduced_area (Rectangle): The reduced area to image with.
            mode: type of alignment (crosscorrelation, phase, phase_subpixel)
            crosscorrelation seems to work with noisy images, where phase correlation fails
            but phase correlation give sub-pixel precision
        Returns:
            shift: list of offset coordinates
        """

        rect_mask = correlation.rectangular_mask(size=ref_image.data.shape, sigma=20)
        low_pass  = int(max(ref_image.data.shape) / 6)   # =128 worked for 768x512
        high_pass = int(max(ref_image.data.shape) / 64) # =12 worked for 768x512
        sigma     = int(max(ref_image.data.shape) / 256)    # =3 worked for 768x512

        shift = correlation.shift_from_crosscorrelation_simple_images(ref_image=ref_image.data,
                                                                      offset_image=image.data,
                                                                      filter='yes',
                                                                      low_pass=low_pass,
                                                                      high_pass=high_pass,
                                                                      sigma=sigma)
        """ check that the shift is not too large e.g. due to the cross-correlation failure
            set the shift values to zero if the shift is larger than the quarter of the 
            corresponding (hor,ver) field width
        """
        if (abs(shift) > np.array(ref_image.data.shape) / 4).any():
            print('Something went wrong with the cross-correlation, the shift is larger than quarter of field width\n'
                  'Setting the stage shift to zero')
            shift = shift * 0

        """ convert pixels to metres """
        shift = movement.pixels_to_metres(coord=shift, image=ref_image)
        dx = shift[1]
        dy = shift[0]

        """ move the stage """
        try:
            print(f"Trying to move the stage by shift correction {-dx}, {+dy}")
            self.move_stage_corrected(expected_x=-dx,
                                      expected_y=+dy)
            self._get_current_microscope_state()
        except Exception as e:
            print(f"Could not apply stage shift, error {e}")

        return shift


    def beam_shift_alignment(self,
                             ref_image,
                             image,
                             reduced_area=None,
                             mode='crosscorrelation'
    ) -> list:
        """Align the images by adjusting the beam shift, instead of moving the stage
                (increased precision, lower range)
        Args:
            ref_image (AdornedImage): reference image to align to
            image: shifted image
            reduced_area (Rectangle): The reduced area to image with.
            mode: type of alignment (crosscorrelation, phase, phase_subpixel)
            crosscorrelation seems to work with noisy images, where phase correlation fails
            but phase correlation give sub-pixel precision
        Returns:
            shift: list of offset coordinates
        """

        rect_mask = correlation.rectangular_mask(size=ref_image.data.shape, sigma=20)
        low_pass = int(max(ref_image.data.shape) / 6)   # =128 worked for 768x512
        high_pass = int(max(ref_image.data.shape) / 64) # =12 worked for 768x512
        sigma = int(max(ref_image.data.shape) / 256)    # =3 worked for 768x512

        shift = correlation.shift_from_crosscorrelation_simple_images(ref_image=ref_image.data * rect_mask,
                                                                      offset_image=image.data * rect_mask,
                                                                      filter='yes',
                                                                      low_pass=low_pass,
                                                                      high_pass=high_pass,
                                                                      sigma=sigma)
        """ check that the shift is not too large e.g. due to the cross-correlation failure
            set the shift values to zero if the shift is larger than the quarter of the 
            corresponding (hor,ver) field width
        """
        if (abs(shift) > np.array(ref_image.data.shape) / 4).any():
            print('Something went wrong with the cross-correlation, the shift is larger than quarter of field width\n'
                  'Setting the beam shifts to zero')
            shift = shift * 0

        """ convert pixels to metres """
        shift = movement.pixels_to_metres(coord=shift, image=ref_image)
        dx = shift[1]
        dy = shift[0]

        """ adjust the beamshift """
        # TODO check the syntax +=(-dy, dx) or +=Point(-dx,dy)
        try:
            print(f"trying to apply beam shift correction {-dx}, {dy}")
            # TODO backend from frontend separation
            self.microscope.beams.electron_beam.beam_shift.value += (-dx, dy)
            self._get_current_microscope_state()
        except Exception as e:
            print(f"Could not apply beam shift, error {e}")

        return shift






    def _get_current_microscope_state(self) -> MicroscopeState:
        """Acquires the current microscope state to store
         if necessary it is possible to return to this stored state later
         Returns the state in MicroscopeState dataclass variable
        Args:
            None
        Returns
        -------
        MicroscopeState
        """
        try:
            (x,y,z,t,r) = self.update_stage_position()
            self.microscope_state.x = x
            self.microscope_state.y = y
            self.microscope_state.z = z
            self.microscope_state.t = t
            self.microscope_state.r = r
            self.microscope_state.working_distance = \
                self.microscope.beams.electron_beam.working_distance.value

            self.microscope_state.horizontal_field_width = \
                self.microscope.beams.electron_beam.horizontal_field_width.value
            self.microscope_state.resolution = self.microscope.beams.electron_beam.scanning.resolution.value

            self.microscope_state.hv = self.microscope.beams.electron_beam.high_voltage.value
            self.microscope_state.beam_current = self.microscope.beams.electron_beam.beam_current.value

            self.microscope_state.scan_rotation_angle = \
                self.microscope.beams.electron_beam.scanning.rotation.value
            self.microscope_state.brightness = self.microscope.detector.brightness.value
            self.microscope_state.contrast = self.microscope.detector.contrast.value

            beam_shift = self.microscope.beams.electron_beam.beam_shift.value # returns Point()
            self.microscope_state.beam_shift_x = beam_shift.x
            self.microscope_state.beam_shift_y = beam_shift.y

        except Exception as e:
            print(f"Could not get the microscope state, error {e}")
            self.microscope_state.x = 2
            self.microscope_state.y = 1
            self.microscope_state.z = 0
            self.microscope_state.t = 0
            self.microscope_state.r = 0
            self.microscope_state.scan_rotation_angle = 0

        return self.microscope_state


    def _restore_microscope_state(self, state : MicroscopeState) -> None:
        """Restores the microscope state from the stored MicroscopeState variable
        Args:
            state : MicroscopeState
            TODO basic restore (state position, imaging conditions),
            full restore (volage, beam current etc)
        Returns
        -------
        None
        """
        try:
            """move function take x,y,z in micrometres and r,t in degrees
            the stored values are straight from the microscope in metres and rad
            """
            self.move_stage(x=state.x, y=state.y, z=state.z,
                            t=state.t, r=state.r,
                            move_type='Absolute')
            self.microscope.beams.electron_beam.horizontal_field_width.value = \
                state.horizontal_field_width
            self.microscope.beams.electron_beam.scanning.resolution = state.resolution
            self.microscope.beams.electron_beam.scanning.rotation.value =\
                state.scan_rotation_angle
            self.microscope.detector.brightness.value = state.brighness
            self.microscope.detector.contrast.value = state.contrast
            self.microscope.beams.electron_beam.beam_shift.value = Point(state.beam_shift_x,
                                                                         state.beam_shift_y)
        except:
            print('Could not restore the microscope state')


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
            self.__autocontrast = autocontrast
        else:
            self.__autocontrast = gui_settings["imaging"]["autocontrast"]

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
            self.path = path
        else:
            self.path = gui_settings["imaging"]["path"]

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
            autocontrast=self.__autocontrast,
            beam_type=BeamType.ELECTRON if beam_type is None else beam_type,
            save=self.save,
            path=self.path,
            sample_name=self.sample_name,
            bit_depth=self.bit_depth,
        )

        return self.image_settings


    def disconnect(self):
        try:
            self.microscope.disconnect()
        except:
            pass # probably demo mode, no microscope connected, continue shutting down





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

    microscope._get_current_microscope_state()



