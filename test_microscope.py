import os

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

microscope = open_microscope()
microscope = initialise(microscope)

#autocontrast(microscope, quadrant=1)
position = update_stage_position(microscope)

print('hfw limits = ', microscope.beams.electron_beam.horizontal_field_width.limits)

resolution = microscope.beams.electron_beam.scanning.resolution.value
print("resolution = ", microscope.beams.electron_beam.scanning.resolution.value, 'type = ', type(resolution))