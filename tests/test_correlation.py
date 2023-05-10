import glob

import matplotlib.pyplot as plt

import os
import numpy as np

# sub-pixel cross-correrlation precision

from importlib import reload
from src import correlation

reload(correlation)


def test_alignment_from_crosscorrelation(ref_image, offset_image, type='phase_subpixel',
                                         filter='yes', low_pass = 128, high_pass=5, sigma=3,
                                         rect_mask=None,
                                         plot=False, title=''):
    rect_mask = np.array(rect_mask)
    if rect_mask.any():
        ref_image = ref_image * rect_mask
        offset_image = offset_image * rect_mask

    if type=='crosscorrelation':
        shift = correlation.shift_from_crosscorrelation_simple_images(ref_image=ref_image,
                                                                      offset_image=offset_image,
                                                                      filter=filter,
                                                                      low_pass=low_pass, high_pass=high_pass, sigma=sigma)
    elif type=='phase':
        shift = correlation.pixel_shift_from_phase_cross_correlation(ref_image=ref_image,
                                                                     offset_image=offset_image)
    elif type=='phase_subpixel':
        shift = correlation.subpixel_shift_from_crosscorrelation(ref_image=ref_image, offset_image=offset_image,
                                                                 upsample_factor=100)
    else:
        shift = correlation.shift_from_crosscorrelation_simple_images(ref_image=ref_image,
                                                                      offset_image=offset_image,
                                                                      filter=filter,
                                                                      low_pass=low_pass, high_pass=high_pass,
                                                                      sigma=sigma)

    aligned = correlation.correct_image_by_shift(offset_image, shift=shift)


    if plot:
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(title + f"\n lowpass {low_pass}, highpass {high_pass}, sigma {sigma}"
                     f"\n {shift}", fontsize=16)
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

        ax1.imshow(ref_image, cmap='gray')
        ax1.set_axis_off()
        ax1.set_title('previous image ref')

        ax2.imshow(ref_image, cmap='Blues_r', alpha=1)
        ax2.imshow(offset_image, cmap='Oranges_r', alpha=0.5)
        ax2.set_axis_off()
        ax2.set_title('next image overlayed with previous')

        ax3.imshow(ref_image, cmap='Blues_r', alpha=1)
        ax3.imshow(aligned, cmap='Oranges_r', alpha=0.5)
        ax3.set_axis_off()
        ax3.set_title("aligned overlayed")

    return shift, aligned



#########################################################################################
#########################################################################################
#########################################################################################

if 0:
    DIR_shifts = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_zscan_220817.092630'
    DIR_rotations = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_220817.094443'
    DIR_rotations_scanRotCorrection = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_SRcorrected_220817.095914'


    DIR = DIR_shifts
    images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
    i = 0
    di = 1

    ref_image = correlation.load_image(images[i])
    offset_image = correlation.load_image(images[i+di])

    ref_image = correlation.normalise(ref_image)
    offset_image = correlation.normalise(offset_image)

    # low_pass = int(max(offset_image.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
    # high_pass = int(max(offset_image.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
    # sigma = int(3 * max(offset_image.data.shape) / 1536)  # ~2-3   @ 1536x1024, good for e-beam images

    # values that worked for rotated images and z-scan images
    low_pass = int(max(offset_image.data.shape) / 6) # =128 worked for @768x512
    high_pass = int(max(offset_image.data.shape) / 64) # =12 worked for @768x512
    sigma = 3
    print(f'lowpass = {low_pass}, highpass = {high_pass}, sigma = {sigma}')


    rect_mask = correlation.rectangular_mask(size=offset_image.shape, sigma=20)

    ref_image_filtered, _, _ = correlation.filter_image_in_fourier_space(image=ref_image * rect_mask,
                                                                         low_pass=low_pass,
                                                                         high_pass=high_pass,
                                                                         sigma=sigma,
                                                                         plot=False)
    offset_image_filtered, _, _ = correlation.filter_image_in_fourier_space(image=offset_image * rect_mask,
                                                                            low_pass=low_pass,
                                                                            high_pass=high_pass,
                                                                            sigma=sigma,
                                                                            plot=True,
                                                                            title=f"{DIR} \n"
                                                                                  f"i1={i}, i2={i+di} \n"
                                                                                  f"lowpass {low_pass}, highpass {high_pass}, sigma {sigma}")


    shift, aligned = test_alignment_from_crosscorrelation(ref_image, offset_image, type='phase_subpixel',
                                                          filter='yes', low_pass = low_pass, high_pass=high_pass, sigma=sigma,
                                                          rect_mask=rect_mask,
                                                          plot=True,
                                                          title=f'Phase subpixel:')

    shift, aligned = test_alignment_from_crosscorrelation(ref_image, offset_image, type='phase',
                                                          filter='yes', low_pass = low_pass, high_pass=high_pass, sigma=sigma,
                                                          rect_mask=rect_mask,
                                                          plot=True,
                                                          title=f'Phase:')

    shift, aligned = test_alignment_from_crosscorrelation(ref_image, offset_image, type='crosscorrelation',
                                                          filter='yes', low_pass = low_pass, high_pass=high_pass, sigma=3,
                                                          rect_mask=rect_mask,
                                                          plot=True,
                                                          title=f'Simple corr:')







#########################################################################################
#########################################################################################
#########################################################################################

if 0:
    DIR_shifts = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_zscan_220817.092630'
    DIR_rotations = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_220817.094443'
    DIR_rotations_scanRotCorrection = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rscan_SRcorrected_220817.095914'


    DIR = DIR_rotations_scanRotCorrection
    images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
    i = 5
    di = 1

    ref_image = correlation.load_image(images[i])
    offset_image = correlation.load_image(images[i+di])

    ref_image = correlation.normalise(ref_image)
    offset_image = correlation.normalise(offset_image)

    # low_pass = int(max(offset_image.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
    # high_pass = int(max(offset_image.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
    # sigma = int(3 * max(offset_image.data.shape) / 1536)  # ~2-3   @ 1536x1024, good for e-beam images

    low_pass = int(max(offset_image.data.shape) / 6) # =128 worked for @768x512
    high_pass = int(max(offset_image.data.shape) / 64) # =12 worked for @768x512
    sigma = 3
    print(f'lowpass = {low_pass}, highpass = {high_pass}, sigma = {sigma}')


    rect_mask = correlation.rectangular_mask(size=offset_image.shape, sigma=20)

    ref_image_filtered, _, _ = correlation.filter_image_in_fourier_space(image=ref_image * rect_mask,
                                                                         low_pass=low_pass,
                                                                         high_pass=high_pass,
                                                                         sigma=sigma,
                                                                         plot=False)
    offset_image_filtered, _, _ = correlation.filter_image_in_fourier_space(image=offset_image * rect_mask,
                                                                            low_pass=low_pass,
                                                                            high_pass=high_pass,
                                                                            sigma=sigma,
                                                                            plot=True,
                                                                            title=f"{DIR} \n"
                                                                                  f"i1={i}, i2={i+di} \n"
                                                                                  f"lowpass {low_pass}, highpass {high_pass}, sigma {sigma}")


    shift, aligned = test_alignment_from_crosscorrelation(ref_image, offset_image, type='phase_subpixel',
                                                          filter='yes', low_pass = low_pass, high_pass=high_pass, sigma=sigma,
                                                          rect_mask=rect_mask,
                                                          plot=True,
                                                          title=f'Phase subpixel:')

    shift, aligned = test_alignment_from_crosscorrelation(ref_image, offset_image, type='phase',
                                                          filter='yes', low_pass = low_pass, high_pass=high_pass, sigma=3,
                                                          rect_mask=rect_mask,
                                                          plot=True,
                                                          title=f'Phase corr:')

    shift, aligned = test_alignment_from_crosscorrelation(ref_image, offset_image, type='simple',
                                                          filter='yes', low_pass = low_pass, high_pass=high_pass, sigma=3,
                                                          rect_mask=rect_mask,
                                                          plot=True,
                                                          title=f'Simple corr:')






#########################################################################################
#########################################################################################
#########################################################################################

if 1:
    DIR_tilts = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_Si_tilt__220824.143216'
    DIR = DIR_tilts
    images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
    i = 0
    di = 40
    tilt = (i+di)*0.5

    DIR_rot = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_Si_rot__220824.141835'
    DIR = DIR_rot
    images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
    i = 0
    di = 8
    rotation = (i+di)*5

    DIR_rot_tilt = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rot_tilt__220824.153205'
    DIR = DIR_rot_tilt
    images = sorted(glob.glob( os.path.join(DIR, '*.tif')))
    i = 0
    di = 133

    ref_image = correlation.load_image(images[i])
    offset_image = correlation.load_image(images[i + di])

    ref_image = correlation.normalise(ref_image)
    offset_image = correlation.normalise(offset_image)

    # low_pass = int(max(offset_image.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
    # high_pass = int(max(offset_image.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
    # sigma = int(3 * max(offset_image.data.shape) / 1536)  # ~2-3   @ 1536x1024, good for e-beam images

    low_pass = int(max(offset_image.data.shape) / 6) # =128 worked for @768x512
    high_pass = int(max(offset_image.data.shape) / 64) # =12 worked for @768x512
    sigma = int(max(offset_image.data.shape) / 256)
    print(f'lowpass = {low_pass}, highpass = {high_pass}, sigma = {sigma}')


    rect_mask = correlation.rectangular_mask(size=offset_image.shape, sigma=20)

    ref_image_filtered, _, _ = correlation.filter_image_in_fourier_space(image=ref_image * rect_mask,
                                                                         low_pass=low_pass,
                                                                         high_pass=high_pass,
                                                                         sigma=sigma,
                                                                         plot=True)
    offset_image_filtered, _, _ = correlation.filter_image_in_fourier_space(image=offset_image * rect_mask,
                                                                            low_pass=low_pass,
                                                                            high_pass=high_pass,
                                                                            sigma=sigma,
                                                                            plot=True)


    shift, aligned = test_alignment_from_crosscorrelation(ref_image, offset_image, type='phase_subpixel',
                                                          filter='yes', low_pass = low_pass, high_pass=high_pass, sigma=sigma,
                                                          rect_mask=rect_mask,
                                                          plot=True,
                                                          title=f'Phase subpixel:')

    shift, aligned = test_alignment_from_crosscorrelation(ref_image, offset_image, type='phase',
                                                          filter='yes', low_pass = low_pass, high_pass=high_pass, sigma=3,
                                                          rect_mask=rect_mask,
                                                          plot=True,
                                                          title=f'Phase corr:')

    shift, aligned = test_alignment_from_crosscorrelation(ref_image, offset_image, type='simple',
                                                          filter='yes', low_pass = low_pass, high_pass=high_pass, sigma=3,
                                                          rect_mask=rect_mask,
                                                          plot=True,
                                                          title=f'Simple corr:')


