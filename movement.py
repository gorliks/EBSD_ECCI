import time
import logging
import numpy as np

try:
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.enumerations import (
        CoordinateSystem,
    )
    from autoscript_sdb_microscope_client.structures import (
        MoveSettings,
        StagePosition,
        AdornedImage,
    )
except Exception as e:
    demo = True
    print(f'demo mode, no Autoscript detected, exception {e}')

from utils import BeamType


def pixel_to_realspace_coordinate(coord: list, image) -> list:
    """Convert pixel image coordniate to real space coordinate.
    This conversion deliberately ignores the nominal pixel size in y,
    as this can lead to inaccuracies if the sample is not flat in y.
    Parameters
    ----------
    coord : listlike, float
        In x, y format & pixel units. Origin is at the top left.
    image : AdornedImage
        Image the coordinate came from.
    Returns
    -------
    realspace_coord
        xy coordinate in real space. Origin is at the image center.
        Output is in (x, y) format.
    """
    coord = np.array(coord).astype(np.float64)

    try: # Is it Adorned image format?
        if len(image.data.shape) > 2:
            y_shape, x_shape = image.data.shape[0:2]
        else:
            y_shape, x_shape = image.data.shape
        pixelsize_x = image.metadata.binary_result.pixel_size.x
        pixelsize_y = image.metadata.binary_result.pixel_size.y
        # deliberately don't use the y pixel size, any tilt will throw this off

    except AttributeError:
        # not Adorned image, no pixel size in the metadata, pixelsize is just one pixel
        # image is presumably np.array
        y_shape, x_shape = image.shape[0:2]
        (pixelsize_x, pixelsize_y) = (1, 1)

    coord[1] = y_shape - coord[1]  # flip y-axis for relative coordinate system
    # reset origin to center
    coord -= np.array([x_shape / 2, y_shape / 2]).astype(np.int64)
    realspace_coord = list(np.array(coord) * pixelsize_x)  # to real space
    return realspace_coord


def pixels_to_metres(coord: list, image) -> list:
    pixelsize_x = image.metadata.binary_result.pixel_size.x
    pixelsize_y = image.metadata.binary_result.pixel_size.y
    coord_metres = list(np.array(coord) * pixelsize_x)  # to real space
    return coord_metres


def move_stage(microscope,
               x : float = 0,
               y : float = 0,
               z : float = 0,
               move_type='Relative',
               stage_settings : dict = None
               ):
    """Move the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system
        The units of movement vector are in metres
        The units of supplied angles are radians
    """
    try:
        """TODO !Compucentric tilt positioning option is not supported on this microscope!"""
        stage = microscope.specimen.stage
        stage_settings = MoveSettings(rotate_compucentric=True)
        #                              tilt_compucentric=True <-- Not supported
        if move_type=="Absolute":
            stage.absolute_move(StagePosition(x=x, y=y, z=z),
                                stage_settings)
        else:
            stage.relative_move(StagePosition(x=x, y=y, z=z),
                                stage_settings)

        position = stage.current_position

        return (position.x, position.y, position.z,
                position.t, position.r)

    except Exception as e:
        print(f'error {e}: movement abs: Could not execute movement, demo mode')
        return (5,5,5,5,5)



def rotate_stage(microscope,
                 r: float = 0,
                 move_type : str = "Relative",
                 stage_settings : dict = None
                 ):
    """Rotate the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system
        The units of supplied angles are radians
    """
    try:
        """TODO !Compucentric tilt positioning option is not supported on this microscope!"""
        stage = microscope.specimen.stage
        stage_settings = MoveSettings(rotate_compucentric=True)
        new_stage_position = StagePosition(r=r)

        if move_type=="Absolute":
            stage.absolute_move(new_stage_position, stage_settings)
        else:
            stage.relative_move(new_stage_position, stage_settings)

        position = stage.current_position

        return (position.x, position.y, position.z,
                position.t, position.r
                )
    except Exception as e:
        print(f'error {e}: movement rel: Could not execute movement, demo mode')
        return (5,5,5,5,5)



def tilt_stage(microscope,
                 t: float = 0,
                 move_type : str = "Relative",
                 stage_settings : dict = None
                 ):
    """ Tilt the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system
        The units of supplied angles are radians
    """
    try:
        """TODO !Compucentric tilt positioning option is not supported on this microscope!"""
        stage = microscope.specimen.stage
        #stage_settings = MoveSettings(tilt_compucentric=True)
        stage_settings = MoveSettings(rotate_compucentric=True)

        new_stage_position = StagePosition(t=t)

        if move_type=="Absolute":
            stage.absolute_move(new_stage_position, stage_settings)
        else:
            stage.relative_move(new_stage_position, stage_settings)

        position = stage.current_position

        return (position.x, position.y, position.z,
                position.t, position.r
                )
    except Exception as e:
        print(f'error {e}: movement rel: Could not execute movement, demo mode')
        return (5,5,5,5,5)




if __name__ == '__main__':
    image = np.random.randint(0, 255, [512, 768])
    coord = (100,200)
    realspace_coord = pixel_to_realspace_coordinate(coord=coord, image=image)
    print(realspace_coord)

    new_coor = move_relative(None, 10, 20)
    print(new_coor)