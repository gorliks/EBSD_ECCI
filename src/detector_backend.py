import sys
import numpy as np
import utils

# global, need pixet for e.g. mode definitions, pixet.PX_TPX3_OPM_TOATOT
# sys.path.append(path_to_pixet) is implemented in the frontend detection.py

path_to_pixet = r'C:\Program Files\PIXet Pro'
sys.path.append(path_to_pixet)
try:
    import pypixet
    pixet = pypixet.pixet
except Exception as e:
    print(f"pypixet is unavailable. Unable to load the detector module: {e}")



def initialise(path_to_pixet):
    sys.path.append(path_to_pixet)
    pypixet.start()
    pixet = pypixet.pixet
    print(' - - - - - - - Pixet version: ', pixet.pixetVersion() )
    devices = pixet.devicesByType(pixet.PX_DEVTYPE_TPX3)
    print('low level: devices = ', devices)
    if devices == []:
        print('No device connected')
        devices = ['no device connected']

    return devices


def get_detector_info(device):
    full_name   = device.fullName()
    width       = device.width()
    pixel_count = device.pixelCount()
    chip_count  = device.chipCount()
    chip_id     = device.chipIDs()
    detector_info = {'full_name': full_name, 'width': width,
                     'pixel_count': pixel_count, 'chip_count': chip_count,
                     'chip_id': chip_id,
                     'demo mode' : False}
    return detector_info


def set_acquisition_type(device, type='Frames'):
    if type=='Frames':
        acquisition_type = pixet.PX_ACQTYPE_FRAMES
    if type=='Test pulses':
        acquisition_type = pixet.PX_ACQTYPE_TESTPULSES
    else:
        acquisition_type = pixet.PX_ACQTYPE_FRAMES
    print('setting acquisition type to ', type, 'pixet = ', acquisition_type)
    return acquisition_type


def set_acquisition_mode(device, mode='TOATOT'):
    if mode == 'TOATOT' or mode == 'TOA & TOT':
        print('Setting detector mode to (1) ', mode)
        device.setOperationMode(pixet.PX_TPX3_OPM_TOATOT)
    elif mode == 'TOA':
        print('Setting detector mode to (2)', mode)
        device.setOperationMode(pixet.PX_TPX3_OPM_TOA)
    elif mode == 'EVENT_iTOT':
        print('Setting detector mode to (3)', mode)
        device.setOperationMode(pixet.PX_TPX3_OPM_EVENT_ITOT)
    elif mode == 'TOT_not_OA':
        print('Setting detector mode to (4)', mode)
        device.setOperationMode(pixet.PX_TPX3_OPM_TOT_NOTOA)
    else:
        print('Setting detector mode to (1) ', mode)
        device.setOperationMode(pixet.PX_TPX3_OPM_TOATOT)
    response = mode + ' set'

    return response


def set_number_of_frames(device, number_of_frames=1):
    pass # send settings[] library with all the setting when needed


def set_integration_time(device, integration_time=0.1):
    pass # send settings[] library with all the setting when needed


def set_threshold_energy(device, energy_threshold_keV=2.0):
    device.setThreshold(0, energy_threshold_keV, 2)
    print('Energy threshold = ', device.threshold(0, 2), ' keV')


def get_temperature(device):
    temperature = device.temperature()
    return temperature


def acquire(device,
            number_of_frames=1,
            integration_time=0.1,
            file_name=''):
    file_name_template = file_name
    file_name = file_name + '.pmf'

    print('hardware file name = ', file_name)
    rc = device.doSimpleAcquisition(number_of_frames,
                                        integration_time,
                                        pixet.PX_FTYPE_AUTODETECT,
                                        file_name)

    #  frames are saved into a file after acquisition
    acqCount = device.acqFrameCount()  # number of measured acquisitions (frames)
    integrated_frame = np.zeros( (256, 256) )

    for index in range(acqCount):
        frame = device.acqFrameRefInc(index)  # get frame with index from last acquisition series
        # get frame data to python array/list:
        data = frame.data()
        data = np.array(data)
        data = data.reshape((256, 256))
        integrated_frame = integrated_frame + data # average frames, integrate frames

    # load data into modes and slots for plotting
    modes = ['TOA',        'TOT',       'EVENT',       'iTOT']
    DATA  = {'TOA' : None, 'TOT': None, 'EVENT': None, 'iTOT': None }
    for mode in modes:
        # the client running on the detector will acquire only images for the selected mode, e.g. TOA
        # and save the data into a file name with the key name of the mode e.g. qwert_TOA.pmf
        # the read_data_file will try to read all the time possible files *TOA.pmf, TOT.pmf etc
        # if the file is not found, the image data will be None in the returned dictionary
        # when trying to plot the data, if None, the gui will skip the plotting
        print(file_name_template + '_' + mode + '.pmf')
        data = utils.read_data_file(file_name_template + '_' + mode + '.pmf')
        DATA[mode] = data # either np.array or None

    return DATA



def close():
    try:
        pypixet.exit()
        return True
    except Exception as e:
        print(f"Could not close pypixet: {e}")
        return False








if __name__ == '__main__':
    path_to_pixet = r'C:\Program Files\PIXet Pro'
    sys.path.append(path_to_pixet)
    import pypixet

    pixet = pypixet.pixet
    pypixet.start()

    print(pixet.pixetVersion())
    devices = pixet.devicesByType(pixet.PX_DEVTYPE_TPX3)

    device = devices[0]
    # device.setOperationMode(pixet.PX_TPX3_OPM_EVENT_ITOT)
    device.setOperationMode(pixet.PX_TPX3_OPM_TOATOT)

    # make integral acquisition 100 frames, 0.1 s and save to file
    # device.doSimpleIntegralAcquisition(100, 0.1, pixet.PX_FTYPE_AUTODETECT, "test2.pmf")
    # make data driven acquisition for 5 seconds and save to file:
    # device.doAdvancedAcquisition(1, 1, pixet.PX_ACQTYPE_DATADRIVEN, pixet.PX_ACQMODE_NORMAL, pixet.PX_FTYPE_AUTODETECT, 0, "test.t3pa")



    # make data driven acquisition and process the pixels in the script. Note: if you want to process
    # the data online you cannot save the data in the acquisition function. You can save them later by calling
    # pixels.save()
    # acqCount = 10
    # acqTime = 0.1 # in seconds, 0.1 s
    # acqType = pixet.PX_ACQTYPE_FRAMES # pixet.PX_ACQTYPE_DATADRIVEN, pixet.PX_ACQTYPE_TESTPULSES
    # acqMode = pixet.PX_ACQMODE_NORMAL # pixet.PX_ACQMODE_TRG_HWSTART, pixet.PX_ACQMODE_TDI, ...
    # fileType = pixet.PX_FTYPE_AUTODETECT
    # fileFlags = 0
    # outputFile = "test.pmf"
    # device.doAdvancedAcquisition(acqCount, acqTime, acqType, acqMode, fileType, fileFlags, outputFile)
    #

    device.doAdvancedAcquisition(10, 0.1, pixet.PX_ACQTYPE_DATADRIVEN, pixet.PX_ACQMODE_NORMAL, pixet.PX_FTYPE_AUTODETECT, 0, "")

    TPX3_INDEX = 0
    TPX3_TOT = 1
    TPX3_TOA = 2

    # get tpx3 pixels:
    pixels = device.lastAcqPixelsRefInc()
    pixelCount = pixels.totalPixelCount()
    pixelData = pixels.pixels()
    print("PixelCount: %d " % pixelCount)

    # get first pixel values:
    matrixIndex = pixelData[TPX3_INDEX][0]
    event = pixelData[TPX3_TOT][0]
    itot = pixelData[TPX3_TOA][0]

    # get second pixel values:
    matrixIndex = pixelData[TPX3_INDEX][1]
    event = pixelData[TPX3_TOT][1]
    itot = pixelData[TPX3_TOA][1]

    # save data to a file
    pixels.save("/tmp/test2.t3pa", pixet.PX_FTYPE_AUTODETECT, 0)

    pixels.destroy()