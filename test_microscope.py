import os
import correlation
import movement

try:
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import (AdornedImage,
                                                             GrabFrameSettings,
                                                             Rectangle,
                                                             RunAutoCbSettings,
                                                             Point)
except Exception as e:
    print(f"AutoLiftout is unavailable. Unable to connect to microscope: {e}")


def open_microscope():
    try:
        microscope = SdbMicroscopeClient()
        print(microscope)
        return microscope
    except Exception as e:
        print(f"SdbMicroscopeClientis unavailable. Unable to create microscope variable: {e}")

def initialise(microscope, ip_address="192.168.0.1"):
    try:
        microscope.connect(ip_address)
    except Exception as e:
        print(f"Unable to connect to microscope: {e}")
    return microscope

def autocontrast(microscope, quadrant: int = 1) -> None:
    microscope.imaging.set_active_view(quadrant)
    RunAutoCbSettings(
        method="MaxContrast",
        resolution="768x512",  # low resolution, so as not to damage the sample
        number_of_frames=5,
    )
    microscope.auto_functions.run_auto_cb()

def update_stage_position(microscope):
    position = microscope.specimen.stage.current_position
    print(position)
    return position


if __name__ == '__main__':

    """
        Basic tests
    """
    microscope = open_microscope()
    microscope = initialise(microscope)

    #autocontrast(microscope, quadrant=1)
    position = update_stage_position(microscope)

    print('hfw limits = ', microscope.beams.electron_beam.horizontal_field_width.limits)

    resolution = microscope.beams.electron_beam.scanning.resolution.value
    print("resolution = ",
          microscope.beams.electron_beam.scanning.resolution.value, 'type = ', type(resolution))


    """
        reset the beam shift
        grab a referece image
        manually apply beam shift
        take an offset image
        find the correction shift from cross-correlation
    """
    microscope.beams.electron_beam.beam_shift.value = Point(0, 0)

    dwell_time = 2e-6
    grab_frame_settings = GrabFrameSettings(dwell_time=dwell_time)
    ref_image = microscope.imaging.grab_frame(grab_frame_settings)

    """ manually shift the image using beam shift"""
    shift_x = +2.3e-6
    shift_y = -1.8e-6
    microscope.beams.ion_beam.beam_shift.value += (shift_x, shift_y)

    """grab a shifted image"""
    offset_image = microscope.imaging.grab_frame(grab_frame_settings)

    """ find the shift using the reference image """
    low_pass = int(max(ref_image.data.shape) / 6)  # =128 worked for @768x512
    high_pass = int(max(ref_image.data.shape) / 64)  # =12 worked for @768x512
    sigma = int(max(ref_image.data.shape) / 256)
    rect_mask = correlation.rectangular_mask(size=ref_image.data.shape, sigma=20)
    _shift_ = correlation.shift_from_crosscorrelation_simple_images(ref_image=ref_image.data,
                                                                  offset_image=offset_image.data,
                                                                  filter='yes',
                                                                  low_pass=low_pass, high_pass=high_pass,
                                                                  sigma=sigma)

    shift = movement.pixel_to_realspace_coordinate(coord=_shift_, image=ref_image)
    dx = shift[1]
    dy = shift[0]

    print(f'manually set shift x={shift_x/1e-6}, y={shift_y/1e-6} um')
    print(f'shift found from crosscorrelation x={dx/1e-6}, y={dy/1e-6} um')


    yy = input('Do the correction yes/(no)?')

    if yy=='yes':
        print(' - - - - - performing correction  - - - - - ')
        """ correct the shift using beam shift, grab new image"""
        # TODO check the syntax (dx,dy) or Point(dx,dy)
        microscope.beams.electron_beam.beam_shift.value += (-dx, dy) # ?

        aligned_image = microscope.imaging.grab_frame(grab_frame_settings)

        fig = plt.figure(figsize=(16, 6))
        fig.suptitle('beam shift correction test', fontsize=16)
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

        ax1.imshow(ref_image.data, cmap='gray')
        ax1.set_axis_off()
        ax1.set_title('ref image ref')

        ax2.imshow(ref_image.data, cmap='Blues_r', alpha=1)
        ax2.imshow(offset_image.data, cmap='Oranges_r', alpha=0.5)
        ax2.set_axis_off()
        ax2.set_title('offset image overlayed with reference')

        ax3.imshow(ref_image.data, cmap='Blues_r', alpha=1)
        ax3.imshow(aligned_image.data, cmap='Oranges_r', alpha=0.5)
        ax3.set_axis_off()
        ax3.set_title("aligned overlayed with reference")





