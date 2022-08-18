import scipy
import scipy.ndimage as ndi
import os, glob
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

from importlib import reload
import correlation
reload(correlation)


if 0:
    image1 = correlation.load_image("01_tilted.tif")
    image2 = correlation.load_image("02_flat.tif")
    image3 = correlation.load_image("02_flat_shifted.tif")


    ref_image = image2
    offset_image = image3

    shift = correlation.subpixel_shift_from_crosscorrelation(ref_image=ref_image,
                                                          offset_image=offset_image,
                                                          upsample_factor=1000)
    aligned = correlation.correct_image_by_shift(offset_image, shift)

    overlayed = correlation.overlay_images(aligned, ref_image)

    fig = plt.figure(figsize=(8, 3))
    fig.suptitle('Subpixel cross-correlation', fontsize=16)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(ref_image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(aligned, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('aligned')

    ax3.imshow(overlayed)
    ax3.set_axis_off()
    ax3.set_title("overlayed")


#########################################################################################
#### unpack the black box, access to more parameters in the cross-correlation parameters
#########################################################################################
if 0:
    DIR_shifts = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_zscan_220817.092630'
    DIR_rotations = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_220817.094443'
    DIR_rotations_scanRotCorrection = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_SRcorrected_220817.095914'

    TYPE = 'SHIFTS'

    if TYPE=='SHIFTS':
        DIR = DIR_shifts
        images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
        i = 12
        ref_image = correlation.load_image(images[i])
        offset_image = correlation.load_image(images[i+3])
        ref_image = correlation.normalise(ref_image)
        offset_image = correlation.normalise(offset_image)
        lowpass_pixels = int(max(offset_image.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
        highpass_pixels = int(max(offset_image.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
        sigma = int(2 * max(offset_image.data.shape) / 1536)  # =2   @ 1536x1024, good for e-beam images
        lowpass_pixels = 100
        highpass_pixels = 4
        sigma = 5

    else:
        DIR = DIR_rotations_scanRotCorrection
        images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
        i = 11
        ref_image = correlation.load_image(images[i])
        offset_image = correlation.load_image(images[i+1])
        lowpass_pixels = 100
        highpass_pixels = 4
        sigma = 5



    xcorr = correlation.crosscorrelation(ref_image, offset_image, filter="yes",
                                         low_pass=lowpass_pixels,
                                         high_pass=highpass_pixels,
                                         sigma=sigma)

    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    shift = -1 * np.array(cen - [maxX, maxY], int)
    print("Shift between 1 and 2 is = " + str(shift))


    aligned = correlation.correct_image_by_shift(offset_image, shift)


    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('rotations (scan rot corrected) cross-correlation', fontsize=16)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(ref_image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('previous image ref')

    ax2.imshow(ref_image, cmap='Blues_r', alpha=1)
    ax2.imshow(offset_image, cmap='Oranges_r', alpha=0.5)
    ax2.set_axis_off()
    ax2.set_title('next image overlay')

    ax3.imshow(ref_image, cmap='Blues_r', alpha=1)
    ax3.imshow(aligned, cmap='Oranges_r', alpha=0.5)
    ax3.set_axis_off()
    ax3.set_title("aligned overlayed")



####################################################################################################
####################################################################################################
####################################################################################################
if 0:
    DIR_shifts = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_zscan_220817.092630'
    DIR_rotations = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_220817.094443'
    DIR = DIR_shifts
    images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
    i = 12


    ref_image = correlation.load_image(images[i])
    offset_image = correlation.load_image(images[i+1])

    shift = correlation.subpixel_shift_from_crosscorrelation(ref_image=ref_image,
                                                          offset_image=offset_image,
                                                          upsample_factor=1000)
    aligned = correlation.correct_image_by_shift(offset_image, shift)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('subpixel cross-correlation, z-scan', fontsize=16)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(ref_image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('previous image ref')

    ax2.imshow(ref_image, cmap='Blues_r', alpha=1)
    ax2.imshow(offset_image, cmap='Oranges_r', alpha=0.5)
    ax2.set_axis_off()
    ax2.set_title('next image overlay')

    ax3.imshow(ref_image, cmap='Blues_r', alpha=1)
    ax3.imshow(aligned, cmap='Oranges_r', alpha=0.5)
    ax3.set_axis_off()
    ax3.set_title("aligned overlayed")




####################################################################################################
####################################################################################################
####################################################################################################
if 0:

    DIR_shifts = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_zscan_220817.092630'
    DIR_rotations = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_220817.094443'
    DIR = DIR_rotations
    images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
    i = 21


    ref_image = correlation.load_image(images[i])
    offset_image = correlation.load_image(images[i+1])

    shift = correlation.subpixel_shift_from_crosscorrelation(ref_image=ref_image,
                                                          offset_image=offset_image,
                                                          upsample_factor=1000)
    aligned = correlation.correct_image_by_shift(offset_image, shift)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('rotations cross-correlation', fontsize=16)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(ref_image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('previous image ref')

    ax2.imshow(ref_image, cmap='Blues_r', alpha=1)
    ax2.imshow(offset_image, cmap='Oranges_r', alpha=0.5)
    ax2.set_axis_off()
    ax2.set_title('next image overlay')

    ax3.imshow(ref_image, cmap='Blues_r', alpha=1)
    ax3.imshow(aligned, cmap='Oranges_r', alpha=0.5)
    ax3.set_axis_off()
    ax3.set_title("aligned overlayed")




####################################################################################################
####################################################################################################
####################################################################################################

if 1:
    DIR_shifts = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_zscan_220817.092630'
    DIR_rotations = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_220817.094443'
    DIR_rotations_scanRotCorrection = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_SRcorrected_220817.095914'
    # DIR = DIR_rotations_scanRotCorrection
    # images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
    # i = 11
    # ref_image = correlation.load_image(images[i])
    # offset_image = correlation.load_image(images[i+1])


    DIR = DIR_shifts
    images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
    i = 12
    ref_image = correlation.load_image(images[i])
    offset_image = correlation.load_image(images[i+3])

    # shift = correlation.subpixel_shift_from_crosscorrelation(ref_image=ref_image,
    #                                                       offset_image=offset_image,
    #                                                       upsample_factor=1000)
    # shift_2 = correlation.shift_from_crosscorrelation_simple_images(img1=ref_image,
    #                                                                 img2=offset_image,
    #                                                                 low_pass=128,
    #                                                                 high_pass=11,
    #                                                                 sigma=2)

    ref_image = correlation.normalise(ref_image)
    offset_image = correlation.normalise(offset_image)

    lowpass_pixels = int(max(offset_image.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
    highpass_pixels = int(max(offset_image.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
    sigma = int(2 * max(offset_image.data.shape) / 1536)  # =2   @ 1536x1024, good for e-beam images

    xcorr = correlation.crosscorrelation(ref_image, offset_image, filter="no",
                                         low_pass=lowpass_pixels,
                                         high_pass=highpass_pixels,
                                         sigma=sigma)

    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print('\n', maxX, maxY)
    cen = np.asarray(xcorr.shape) / 2
    shift = -1 * np.array(cen - [maxX, maxY], int)
    print("Shift between 1 and 2 is = " + str(shift))


    aligned = correlation.correct_image_by_shift(offset_image, shift)


    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('rotations (scan rot corrected) cross-correlation', fontsize=16)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(ref_image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('previous image ref')

    ax2.imshow(ref_image, cmap='Blues_r', alpha=1)
    ax2.imshow(offset_image, cmap='Oranges_r', alpha=0.5)
    ax2.set_axis_off()
    ax2.set_title('next image overlay')

    ax3.imshow(ref_image, cmap='Blues_r', alpha=1)
    ax3.imshow(aligned, cmap='Oranges_r', alpha=0.5)
    ax3.set_axis_off()
    ax3.set_title("aligned overlayed")