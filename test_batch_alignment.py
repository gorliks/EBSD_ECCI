import scipy
import scipy.ndimage as ndi
import skimage.draw
from scipy import fftpack, misc

import numpy as np
import matplotlib.pyplot as plt

import os, glob

import numpy as np
import skimage.util
import skimage
import skimage.io
import skimage.transform
import skimage.color
import pandas as pd

from importlib import reload
import correlation
reload(correlation)


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
    points = points - centre
    points_rot = np.zeros((points.shape[0], points.shape[1] + 1))
    points_rot = [[point[0], point[1], 0] for point in points]
    points_rot = [np.dot(matrix, point) for point in points_rot]
    points_rot = np.array(points_rot)

    points_rot = project_3d_point_to_xy(points_rot)
    points_rot += centre

    return points_rot




#########################################################################################
#########################################################################################
#########################################################################################

# DIR_rot_tilt = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rot_tilt__220824.153205'
DIR_rot_tilt = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_tilt_rot__220901.110316'
DIR = DIR_rot_tilt
images = sorted(glob.glob( os.path.join(DIR, '*.tif')))

data = pd.read_csv(os.path.join(DIR, "summary.csv"))

tilts = data['t']
rotations = data['r']
scan_rotations = data['scan_rotation_angle']
file_names = data['file_name']

# first image, untilted, unrotated is the reference
ref_image = correlation.load_image( os.path.join(DIR, file_names[0]) )

# source points on grid for projective transform
shape = ref_image.shape
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


COMBINED = np.zeros(ref_image.shape, dtype=np.float)

# rotations 35 steps, tilts 21 steps
map = np.zeros((21, 35))
rot_offset = (np.rad2deg(rotations-rotations[0])//10).min()

# values that worked for rotated images and z-scan images
low_pass = int(max(ref_image.shape) / 6)  # =128 worked for @768x512
high_pass = int(max(ref_image.shape) / 64)  # =12 worked for @768x512
sigma = int(max(ref_image.shape) / 256)
rect_mask = correlation.rectangular_mask(size=ref_image.shape, sigma=20)
print(f'lowpass = {low_pass}, highpass = {high_pass}, sigma = {sigma}')


for i, file_name in enumerate(file_names):
    print(i, file_name)

    tilted_rot_image = correlation.load_image( os.path.join(DIR, file_names[i]) )
    tilt = tilts[i]
    dst = rotate_about_x(angle=tilt,
                         points=src,
                         image=ref_image)
    untilted_image = projective_transform_from_points(points_src=src,
                                                      points_dst=dst,
                                                      image=tilted_rot_image,
                                                      inverse=False)
    shift = correlation.shift_from_crosscorrelation_simple_images(ref_image=ref_image * rect_mask,
                                                                  offset_image=untilted_image * rect_mask,
                                                                  filter='yes',
                                                                  low_pass=low_pass, high_pass=high_pass, sigma=sigma)
    aligned = correlation.correct_image_by_shift(untilted_image, shift=shift)

    COMBINED += aligned

    index_rot = int( np.rad2deg(rotations[i]-rotations[0])//10 - rot_offset-1)
    index_tilt= int(np.rad2deg(tilts[i]-tilts[0]) )
    map[index_tilt, index_rot] = np.sum(aligned[280-2:280+3,350-2:350+3])

plt.figure()
plt.imshow(COMBINED, cmap='gray')
plt.show()

