import datetime
import time
import os
import pandas as pd
import numpy as np
import re
from PIL import Image
import cv2

from importlib import reload  # Python 3.4+
from dataclasses import dataclass
from enum import Enum

try:
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import (AdornedImage,
                                                             GrabFrameSettings,
                                                             Rectangle,
                                                             RunAutoCbSettings,
                                                             Point)
    from autoscript_sdb_microscope_client.enumerations import (CoordinateSystem)
except:
    print('Autoscript module not found')


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
    working_distance : float = 0

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
            "beam_shift_y" : self.beam_shift_y,
            "hv" : self.hv,
            "beam_current" : self.beam_current,
            "working_distance" : self.working_distance
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




@dataclass
class DetectorState:
    type : str = 'TPX3'
    path_to_api : str = ''
    acquisition_type : str = 'Frames'
    acquisition_mode : str = 'EVENT_iTOT'
    number_of_frames : int = 1
    integration_time : float = 0.25
    energy_threshold : float = 2.0
    path_to_data : str = ''
    detector_info : str = ''
    detector_full_name : str = ''
    width : int = 256
    pixel_count : int = 0
    chip_id : str = ''
    chip_count : str = ''
    temperature : float = 25.5

    def __to__dict__(self) -> dict:
        state_dict = {
            "acquisition_type" : self.acquisition_type,
            "acquisition_mode" : self.acquisition_mode,
            "number_of_frames" : self.number_of_frames,
            "integration_time" : self.integration_time,
            "energy_threshold" : self.energy_threshold,
            "path_to_data" : self.path_to_data
        }
        return state_dict




def current_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%y%m%d.%H%M%S")



def save_image(image, path=None, file_name=None):
    if not path:
        path = os.getcwd()
    if not file_name:
        timestamp = current_timestamp()
        file_name = "no_name_" + timestamp + '.tif'

    file_name = os.path.join(path, file_name)
    try:
        """Adorned image needs only path"""
        image.save(file_name)
    except Exception as e:
        print('error {e}, Image is not Adorned, trying to save numpy array to tiff')
        try:
            _im = Image.fromarray(image)
            _im.save(file_name)
        except Exception as e:
            print('error {e}, Could not save the image')



def enhance_contrast(image, clipLimit=1.0, tileGridSize=8):
    # if image.dtype == 'float64' or 'uint64':
    #     pass

    # the images needs to be 8bit, otherwise the algorithm does not work TODO fix 8-bit to any-bit
    image = image / image.max()
    image = (image * 2 ** 8).astype('uint8')

    tileGridSize = int(tileGridSize)

    clahe = cv2.createCLAHE(clipLimit=clipLimit,
                            tileGridSize=(tileGridSize, tileGridSize))
    image = clahe.apply(image)
    return image


def equalise_histogram(image, bitdepth=8):
    try:
        _image = image.data
    except:
        _image = image

    # _image = _image / _image.max()
    # _image = (_image * 2 ** bitdepth).astype("uint8")

    _image = cv2.equalizeHist(_image)
    return _image


def resize(image, size=(200,200)):
    return cv2.resize(image, size)


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


def extract_metadata_from_tif(path_to_file):
    from PIL import Image
    from PIL.TiffTags import TAGS

    with Image.open(path_to_file) as img:
        #[print(key, img.tag[key]) for key in img.tag.keys()]
        metadata_dict = {key: img.tag[key] for key in img.tag.keys()}

        return metadata_dict


def parse_metadata(path_to_file):
    # SEM meta data key is 34682, comes as a string
    img = Image.open(path_to_file)
    img_metadata = img.tag[34682][0]

    # parse metadata
    parsed_metadata = img_metadata.split("\r\n")

    metadata_dict = {}
    for item in parsed_metadata:
        if item == "":
            # skip blank lines
            pass
        elif re.match("\[(.*?)\]", item):
            # find category, dont add to dict
            category = item
        else:
            # meta data point
            datum = item.split("=")
            # save to dictionary
            metadata_dict[category + "." + datum[0]] = datum[1]

    # add filename to metadata
    metadata_dict["filename"] = path_to_file

    # convert to pandas df
    df = pd.DataFrame.from_dict(metadata_dict, orient="index").T

    return df


def make_copy_of_Adorned_image(image):
    copy = AdornedImage()
    AdornedImage._AdornedImage__construct_from_data(copy,
                                                    data=image.data)
    copy.metadata = image.metadata
    return copy


def read_data_file(file_name):
    file_name = file_name.replace('\\', '/')
    if os.path.isfile(file_name):
        data = np.loadtxt(  file_name  )
    else:
        print('file not found')
        data = None
    return data


def rotate_coordinate_in_xy(coord : list = (0,0),
                            angle : float = 0) -> list:
    matrix = np.array([
        [+np.cos(angle), -np.sin(angle)],
        [+np.sin(angle), +np.cos(angle)]
    ])
    coord = np.array(coord)
    coord_rot = np.dot(matrix, coord)
    return list(coord_rot)



def get_indices_from_file_name(file_dir, file_name):
    file_name = file_name.replace(file_dir, '')
    temp = file_name.split('_')
    i = temp[1]
    j = temp[2]
    return int(i), int(j)

def get_Nx_Ny_from_indices(files_dir, files):
    N = len(files)
    print('Number of files = ', N)
    files = [file_name.replace(files_dir, '') for file_name in files ]
    X = []
    Y = []
    for file_name in files:
        temp = file_name.split('_')
        x = temp[1]
        y = temp[2]
        X.append(int(x))
        Y.append(int(y))
    X = np.array(X)
    Y = np.array(Y)
    Nx = X.max()
    Ny = Y.max()
    print(Nx, Ny)
    return Nx+1, Ny+1


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
    # experiment_data = populate_experiment_data_frame(data_frame=experiment_data,
    #                                                  microscope_state=microscope.microscope_state,
    #                                                  file_name='lalala', timestamp=current_timestamp(),
    #                                                  keys=keys)
    print(experiment_data)




    metadata_dict = extract_metadata_from_tif(path_to_file='01_tilted.tif')
    metadata = metadata_dict[34682][0]
    metadata = metadata.split('\r\n')
    pairs = len(metadata)//2
    # for ii in range(pairs):
    #     print(metadata[2*ii], metadata[2*ii+1]  )

    for ii in range(len(metadata)):
        if 'Dwell' in metadata[ii]:
            print(metadata[ii] )
        if 'PixelWidth' in metadata[ii]:
            print(metadata[ii])
        if 'HorFieldsize' in metadata[ii]:
            print(metadata[ii])


    df_metadata = parse_metadata(path_to_file='01_tilted.tif')
    print(f"\nDwelltime = {df_metadata['[Scan].Dwelltime'][0]} s")
    print(f"Stage tilt = {  np.rad2deg(float(df_metadata['[Stage].StageT'][0])) } deg")
    print(f"Hor field w = {  float(df_metadata['[Scan].HorFieldsize'][0])/1e-6 } um")
