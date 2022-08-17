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
image     = image1


matrix = np.array([[1, -0.5, 100],
                   [0.1, 0.9, 50],
                   [0.0015, 0.0015, 1]])

''' create transformation
    transform.ProjectiveTransform(matrix=matrix) A homography, also called projective transformation, preserves lines but not necessarily parallelism.
    transform.AffineTransform(shear=np.pi/6) An affine transformation preserves lines (hence the alignment of objects), as well as parallelism between lines
    transform.SimilarityTransform(scale=0.5, rotation=np.pi/12, translation=(100, 50)) A similarity transformation preserves the shape of objects. It combines scaling, translation and rotation.
    transform.EuclideanTransform(rotation=np.pi / 12.,translation = (100, -20)) preserves the Euclidean distance between pairs of points. It can be described as a rotation about the origin followed by a translation
'''

tform = skimage.transform.ProjectiveTransform(matrix=matrix)
print(tform)
tilted_image = skimage.transform.warp(ref_image, tform.inverse)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

ax1.imshow(ref_image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(tilted_image, cmap='gray')
ax2.set_axis_off()
ax2.set_title('transformed image')


