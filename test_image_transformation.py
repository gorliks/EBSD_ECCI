import scipy
import scipy.ndimage as ndi
import skimage.draw
from scipy import fftpack, misc

import numpy as np
import matplotlib.pyplot as plt

import os, time
import numpy as np
import skimage.util
import skimage
import skimage.io
import skimage.transform
import skimage.color

import correlation


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
        shift = correlation.subpixel_shift_from_crosscorrelation(ref_image=ref_image,offset_image=offset_image,
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



#########################################################################################
#########################################################################################
#########################################################################################
if 1:
    tilted_image = correlation.load_image("01_tilted.tif")
    flat_image = correlation.load_image("02_flat.tif")
    image3 = correlation.load_image("02_flat_shifted.tif")

    shape = tilted_image.shape
    xmin = 0
    xmax = shape[0]
    ymin = 0
    ymax = shape[1]

    X, Y = np.mgrid[xmin:xmax:17j, ymin:ymax:25j]
    src = np.vstack([X.ravel(), Y.ravel()])
    xx = X.ravel()
    yy = Y.ravel()
    src = [ [i,j] for i,j in zip(xx,yy)  ]
    src = np.array(src)

    dst = rotate_about_x(points=src,
                         angle=52,
                         image=flat_image)

    untilted_image = projective_transform_from_points(points_src=src,
                                                      points_dst=dst,
                                                      image=tilted_image,
                                                      inverse=False)


    # values that worked for rotated images and z-scan images
    low_pass = int(max(flat_image.data.shape) / 6)  # =128 worked for @768x512
    high_pass = int(max(flat_image.data.shape) / 64)  # =12 worked for @768x512
    sigma = 6
    print(f'lowpass = {low_pass}, highpass = {high_pass}, sigma = {sigma}')

    rect_mask = correlation.rectangular_mask(size=flat_image.shape, sigma=20)
    _, _, _ = correlation.filter_image_in_fourier_space(image=flat_image * rect_mask,
                                                                            low_pass=low_pass,
                                                                            high_pass=high_pass,
                                                                            sigma=sigma,
                                                                            plot=False,
                                                                            title= f"flat image \n"
                                                                                  f"lowpass {low_pass}, highpass {high_pass}, sigma {sigma}")
    _, _, _ = correlation.filter_image_in_fourier_space(image=untilted_image * rect_mask,
                                                                            low_pass=low_pass,
                                                                            high_pass=high_pass,
                                                                            sigma=sigma,
                                                                            plot=True,
                                                                            title=f"untilted image \n"
                                                                                  f"lowpass {low_pass}, highpass {high_pass}, sigma {sigma}")

    TYPES = ['crosscorrelation', 'phase', 'phase_subpixel']
    shift, untilted_image_aligned = test_alignment_from_crosscorrelation(flat_image, untilted_image,
                                                            type=TYPES[0],
                                                            filter='yes', low_pass=low_pass, high_pass=high_pass, sigma=sigma,
                                                            rect_mask=rect_mask,
                                                            plot=True,
                                                            title=f'cross-correlation:')


    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("define points for projective transform", fontsize=16)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(2, 2, 3, sharex=ax2, sharey=ax2)
    ax4 = plt.subplot(2, 2, 4, sharex=ax3, sharey=ax3)

    ax1.imshow(flat_image, cmap='gray')
    # [ax1.scatter(point[1], point[0], c='r') for point in src]
    ax1.set_axis_off()
    ax1.set_title('flat image')
    #
    ax2.imshow(tilted_image, cmap='gray')
    # [ax2.scatter(point[1], point[0], c='b') for point in dst]
    ax2.set_axis_off()
    ax2.set_title('tilted image')
    #
    ax3.imshow(untilted_image, cmap='gray')
    ax3.set_axis_off()
    ax3.set_title("untilted_image")
    #
    ax4.imshow(flat_image, cmap='Blues_r', alpha=1)
    ax4.imshow(untilted_image, cmap='Oranges_r', alpha=0.5)
    ax4.set_axis_off()
    ax4.set_title("overlayed")


#########################################################################################
#########################################################################################
#########################################################################################

if 0:
    image1 = correlation.load_image("01_tilted.tif")
    image2 = correlation.load_image("02_flat.tif")
    image3 = correlation.load_image("02_flat_shifted.tif")

    ref_image = image2
    image     = image1

    projective_transform_from_matrix(ref_image)


#########################################################################################
#########################################################################################
#########################################################################################

if 0:

    src = [[445.28349348, 713.83713241],
           [330.83521061, 713.83713241],
           [414.21895956, 860.98492467],
           [523.76231602, 775.96620025],
           [442.01354254, 975.43320754],
           [482.88792928, 844.63516997],
           [476.3480274, 1425.05146167],
           [293.23077481, 849.54009638]]

    dst = [[507.41256133, 727.96926583],
           [296.50072575, 723.06433942],
           [448.55344442, 878.38700903],
           [646.38547624, 788.4633582],
           [507.41256133, 984.66041455],
           [572.81158011, 857.13232792],
           [579.35148199, 1409.75403664],
           [229.4667315, 868.57715621]]

    flat_image = correlation.load_image('02_flat.tif')
    tilted_image = correlation.load_image("01_tilted.tif")
    untilted_image = projective_transform_from_points(points_src=src,
                                                      points_dst=dst,
                                                      image=tilted_image)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("untilting tilted image", fontsize=16)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(2, 2, 3, sharex=ax2, sharey=ax2)
    ax4 = plt.subplot(2, 2, 4, sharex=ax3, sharey=ax3)

    ax1.imshow(tilted_image, cmap='gray')
    [ax1.scatter(point[1],point[0]) for point in src]
    ax1.set_axis_off()
    ax1.set_title('tilted')

    ax2.imshow(flat_image, cmap='gray')
    [ax2.scatter(point[1],point[0]) for point in dst]
    ax2.set_axis_off()
    ax2.set_title('flat image')

    ax3.imshow(untilted_image, cmap='gray')
    ax3.set_axis_off()
    ax3.set_title("untilted_image")

    ax4.imshow(untilted_image, cmap='Blues_r', alpha=1)
    ax4.imshow(flat_image, cmap='Oranges_r', alpha=0.5)
    ax4.set_axis_off()
    ax4.set_title("overlayed")


#########################################################################################
#########################################################################################
#########################################################################################


