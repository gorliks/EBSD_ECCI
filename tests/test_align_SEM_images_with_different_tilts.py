import skimage.draw

import matplotlib.pyplot as plt

import os
import numpy as np
import skimage.util
import skimage
import skimage.io
import skimage.transform
import skimage.color

from src import correlation, utils

from importlib import reload  # Python 3.4+
reload(utils)


def projective_transform_from_matrix(image):
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
    tilted_image = skimage.transform.warp(image, tform.inverse)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax1.imshow(image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')
    ax2.imshow(tilted_image, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('transformed image')


def projective_transform_from_points(points_src, points_dst, image, inverse=True):
    src = np.array(points_src)
    dst = np.array(points_dst)

    src = np.flip(src, axis=1)
    dst = np.flip(dst, axis=1)

    tform = skimage.transform.ProjectiveTransform()
    tform.estimate(src, dst)
    if inverse:
        tform = tform.inverse
    print(tform)

    warped_image = np.array( skimage.transform.warp(image, tform) )

    return warped_image


def project_3d_point_to_xy(points):
    points_2d = np.zeros((points.shape[0], points.shape[1] - 1))
    points_2d = [[point[0], point[1]] for point in points]
    points_2d = np.array(points_2d)
    return points_2d


def rotate_about_x(points, angle, image):
    shape = image.shape
    angle = np.deg2rad(angle)
    matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    points = np.array(points)
    centre = np.asarray(shape) / 2
    print(f'centre = {centre}')
    points = points - centre
    points_rot = np.zeros((points.shape[0], points.shape[1] + 1))
    points_rot = [[point[0], point[1], 0] for point in points]
    points_rot = [np.dot(matrix, point) for point in points_rot]
    points_rot = np.array(points_rot)

    points_rot = project_3d_point_to_xy(points_rot)
    points_rot += centre

    return points_rot



def test_alignment_from_crosscorrelation(ref_image, offset_image, type='phase_subpixel',
                                         filter='yes', low_pass = 128, high_pass=5, sigma=3,
                                         rect_mask=None,
                                         plot=False, title=''):
    rect_mask = np.array(rect_mask)
    if rect_mask.any():
        pass
    else:
        rect_mask = 1

    if type=='crosscorrelation':
        shift = correlation.shift_from_crosscorrelation_simple_images(ref_image=ref_image * rect_mask,
                                                                      offset_image=offset_image * rect_mask,
                                                                      filter=filter,
                                                                      low_pass=low_pass, high_pass=high_pass, sigma=sigma)
    elif type=='phase':
        shift = correlation.pixel_shift_from_phase_cross_correlation(ref_image=ref_image * rect_mask,
                                                                     offset_image=offset_image * rect_mask)
    elif type=='phase_subpixel':
        shift = correlation.subpixel_shift_from_crosscorrelation(ref_image=ref_image * rect_mask,
                                                                 offset_image=offset_image * rect_mask,
                                                                 upsample_factor=100)
    else:
        shift = correlation.shift_from_crosscorrelation_simple_images(ref_image=ref_image * rect_mask,
                                                                      offset_image=offset_image * rect_mask,
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






data_dir = r'C:\Users\sergeyg\Github\DATA.2022.10.11.ecci_Fe_fcc\for_alignment'

files = {'0a' : '01_20kV_0.4nA_WD4.4_100kx_0tilt.tif',
         '0b' : '01_20kV_0.4nA_WD4.4_100kx_0tilt_001.tif',
         '-1': '01_20kV_0.4nA_WD4.4_100kx_-1tilt.tif',
         '-2': '01_20kV_0.4nA_WD4.1_100kx_-2tilt.tif'}

tilt_0      = correlation.load_image(os.path.join(data_dir, files['0a']))
tilt_minus1 = correlation.load_image(os.path.join(data_dir, files['-1']))
tilt_minus2 = correlation.load_image(os.path.join(data_dir, files['-2']))

tilt_0 = tilt_0[0:1024, :]
tilt_minus1 = tilt_minus1[0:1024, :]
tilt_minus2 = tilt_minus2[0:1024, :]

rect_mask = correlation.rectangular_mask(size=tilt_0.shape, sigma=20)


##################################################################################################################
''' Untilt transformation of the tilted images for alignment with the flat image'''

shape = tilt_0.shape
xmin = 0
xmax = shape[0]
ymin = 0
ymax = shape[1]

X, Y = np.mgrid[xmin:xmax:17j, ymin:ymax:25j]
src = np.vstack([X.ravel(), Y.ravel()])
xx = X.ravel()
yy = Y.ravel()
src = [[i, j] for i, j in zip(xx, yy)]
src = np.array(src)

dst = rotate_about_x(points=src,
                     angle=-1,
                     image=tilt_0)
tilt_minus1_untilted = projective_transform_from_points(points_src=src,
                                                        points_dst=dst,
                                                        image=tilt_minus1,
                                                        inverse=False)

dst = rotate_about_x(points=src,
                     angle=-2,
                     image=tilt_0)
tilt_minus2_untilted = projective_transform_from_points(points_src=src,
                                                        points_dst=dst,
                                                        image=tilt_minus2,
                                                        inverse=False)

fig = plt.figure(figsize=(16, 6))
fig.suptitle("define points for projective transform", fontsize=16)
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(2, 2, 3, sharex=ax2, sharey=ax2)
ax4 = plt.subplot(2, 2, 4, sharex=ax3, sharey=ax3)

ax1.imshow(tilt_0, cmap='gray')
# [ax1.scatter(point[1], point[0], c='r') for point in src]
ax1.set_axis_off()
ax1.set_title('flat image')
#
ax2.imshow(tilt_minus1, cmap='gray')
# [ax2.scatter(point[1], point[0], c='b') for point in dst]
ax2.set_axis_off()
ax2.set_title('tilted image')
#
ax3.imshow(tilt_minus1_untilted, cmap='gray')
ax3.set_axis_off()
ax3.set_title("untilted_image")
#
ax4.imshow(tilt_0, cmap='Blues_r', alpha=1)
ax4.imshow(tilt_minus1_untilted, cmap='Oranges_r', alpha=0.5)
ax4.set_axis_off()
ax4.set_title("flat+untilted overlayed")

##################################################################################################################


##################################################################################################################

# values that worked for rotated images and z-scan images
low_pass = int(max(tilt_0.shape) / 6)  # =128 worked for @768x512
high_pass = int(max(tilt_0.shape) / 64)  # =12 worked for @768x512
sigma = int(max(tilt_0.shape) / 256)
print(f'lowpass = {low_pass}, highpass = {high_pass}, sigma = {sigma}')

shift, minus1_aligned = test_alignment_from_crosscorrelation(tilt_0, tilt_minus1_untilted, type='crosscorrelation',
                                                      filter='yes', low_pass=low_pass, high_pass=high_pass, sigma=3,
                                                      rect_mask=rect_mask,
                                                      plot=True,
                                                      title=f'Simple corr:')

shift, minus2_aligned = test_alignment_from_crosscorrelation(tilt_0, tilt_minus2_untilted, type='crosscorrelation',
                                                      filter='yes', low_pass=low_pass, high_pass=high_pass, sigma=3,
                                                      rect_mask=rect_mask,
                                                      plot=True,
                                                      title=f'Simple corr:')






fig = plt.figure(figsize=(16, 6))
fig.suptitle("define points for projective transform", fontsize=16)
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(2, 2, 3, sharex=ax2, sharey=ax2)
ax4 = plt.subplot(2, 2, 4, sharex=ax3, sharey=ax3)

ax1.imshow(tilt_0, cmap='gray')
# [ax1.scatter(point[1], point[0], c='r') for point in src]
ax1.set_axis_off()
ax1.set_title('flat image')
#
ax2.imshow(utils.enhance_contrast(tilt_0, clipLimit=15, tileGridSize=64, bitdepth=16), cmap='gray')
# [ax2.scatter(point[1], point[0], c='b') for point in dst]
ax2.set_axis_off()
ax2.set_title('tilted image')
#
ax3.imshow(tilt_minus1_untilted, cmap='gray')
ax3.set_axis_off()
ax3.set_title("untilted_image")
#
ax4.imshow(utils.enhance_contrast(tilt_minus1_untilted, clipLimit=15, tileGridSize=64, bitdepth=16), cmap='gray', alpha=1)
ax4.set_axis_off()
ax4.set_title("flat+untilted overlayed")