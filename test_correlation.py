import scipy
import scipy.ndimage as ndi
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


import correlation


image1 = correlation.load_image("01_tilted.tif")
image2 = correlation.load_image("02_flat.tif")
image3 = correlation.load_image("02_flat_shifted.tif")


ref_image = image2
offset_image = image3

shift, aligned = correlation.subpixel_shift_from_crosscorrelation(ref_image=ref_image,
                                                      offset_image=offset_image,
                                                      upsample_factor=1000)
overlayed = correlation.overlay_images(aligned, ref_image)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

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

ref_image = correlation.normalise(ref_image)
offset_image = correlation.normalise(offset_image)

lowpass_pixels = int(max(offset_image.data.shape) / 12)  # =128 @ 1536x1024, good for e-beam images
highpass_pixels = int(max(offset_image.data.shape) / 256)  # =6   @ 1536x1024, good for e-beam images
sigma = int(2 * max(offset_image.data.shape) / 1536)  # =2   @ 1536x1024, good for e-beam images

bandpass = correlation.bandpass_mask(ref_image.shape,
                                     low_pass=lowpass_pixels,
                                     high_pass=highpass_pixels,
                                     sigma=sigma)
xcorr = correlation.crosscorrelation(ref_image, offset_image, filter="yes",
                         low_pass=lowpass_pixels,
                         high_pass=highpass_pixels,
                         sigma=sigma)

maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
print('\n', maxX, maxY)
cen = np.asarray(xcorr.shape) / 2
err = np.array(cen - [maxX, maxY], int)
print("Shift between 1 and 2 is = " + str(err))

dy, dx = correlation.shift_from_crosscorrelation_simple_images(ref_image, offset_image,
                                                   low_pass=lowpass_pixels,
                                                   high_pass=highpass_pixels,
                                                   sigma=sigma)

aligned = scipy.ndimage.shift(offset_image, -1*err,
                              output=None, order=3, mode='constant', cval=0.0, prefilter=True)

overlayed = correlation.overlay_images(aligned, ref_image)


fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(ref_image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(aligned, cmap='gray')
ax2.set_axis_off()
ax2.set_title('aligned')

ax3.imshow(overlayed)
ax3.set_axis_off()
ax3.set_title("overlayed")