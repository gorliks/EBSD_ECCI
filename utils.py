import datetime
import time
import os
import pandas as pd

from importlib import reload  # Python 3.4+
from dataclasses import dataclass
from enum import Enum

class BeamType(Enum):
    ION = 'ION'
    ELECTRON = 'ELECTRON'

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
    brightness : float = 0
    contrast : float = 0
    beam_shift_x : float = 0
    beam_shift_y : float = 0

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
            "brightness" : self.brightness,
            "contrast" : self.contrast,
            "beam_shift_x" : self.beam_shift_x,
            "beam_shift_y" : self.beam_shift_y
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





def current_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%y%m%d.%H%M%S")


def save_image(image, path=None, file_name=None):
    if not path:
        path = os.getcwd()
    if not file_name:
        timestamp = current_timestamp()
        file_name = "no_name_" + timestamp + '.tif'

    file_name = os.path.join(path, file_name)
    print(file_name)


def populate_experiment_data_frame(data_frame : dict,
                                   keys : list,
                                   microscope_state : MicroscopeState,
                                   file_name : str = "None",
                                   timestamp: str = "None") -> dict:
    microscope_state = microscope_state.__to__dict__()
    for key in keys:
        data_frame[key].append( microscope_state[key] )

    data_frame['file_name'].append(file_name)
    data_frame['timestamp'].append(timestamp)

    return data_frame


def save_data_frame(data_frame : dict,
                    path : str,
                    file_name : str) -> None:
    """convert the accumulated dictionary to pandas DataFrame
        save .cvs file in the directory as a form of metadata
    Parameters
    ----------
    data_frame : dict
        contains microscope state snippet for each acquired image and filenames.
    path : str
        Path of the stack where the images are stored.
    file_name : str
        Filename of the .cvs file
    Returns
    -------
    None
    """
    d = pd.DataFrame(data=data_frame)
    file_name = os.path.join(path, file_name + '.csv')
    d.to_csv(file_name)


if __name__ == '__main__':

    keys = ('x', 'y', 'z', 't', 'r',
            'horizontal_field_width', 'scan_rotation_angle',
            'brightness', 'contrast',
            'beam_shift_x', 'beam_shift_y')
    experiment_data = {element: [] for element in keys}
    """add other data keys"""
    experiment_data['file_name'] = []
    experiment_data['timestamp'] = []
    print(experiment_data)
    experiment_data = populate_experiment_data_frame(data_frame=experiment_data,
                                                     microscope_state=microscope.microscope_state,
                                                     file_name='lalala', timestamp=current_timestamp(),
                                                     keys=keys)
    print(experiment_data)






