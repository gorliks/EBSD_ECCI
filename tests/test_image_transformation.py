import skimage.draw

import matplotlib.pyplot as plt

import os, glob
import numpy as np
import skimage.util
import skimage
import skimage.io
import skimage.transform
import skimage.color

from src import correlation


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
        fig = plt.figure(figsize=(16, 16))
        fig.suptitle(title + f"\n lowpass {low_pass}, highpass {high_pass}, sigma {sigma}"
                     f"\n {shift}", fontsize=16)
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
        ax3 = plt.subplot(2, 2, 3, sharex=ax2, sharey=ax2)
        ax4 = plt.subplot(2, 2, 4, sharex=ax3, sharey=ax3)


        ax1.imshow(ref_image, cmap='gray')
        ax1.set_axis_off()
        ax1.set_title('previous image ref')

        ax2.imshow(ref_image, cmap='Blues_r', alpha=1)
        ax2.imshow(offset_image, cmap='Oranges_r', alpha=0.5)
        ax2.set_axis_off()
        ax2.set_title('next image overlayed with previous')

        ax3.imshow(aligned, cmap='gray')
        ax3.set_axis_off()
        ax3.set_title('offset image after alignment')

        ax4.imshow(ref_image, cmap='Blues_r', alpha=1)
        ax4.imshow(aligned, cmap='Oranges_r', alpha=0.5)
        ax4.set_axis_off()
        ax4.set_title("aligned overlayed")

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
    tilted_image = correlation.load_image("../data/01_tilted.tif")
    flat_image = correlation.load_image("../data/02_flat.tif")
    image3 = correlation.load_image("../data/02_flat_shifted.tif")

    flat_image = correlation.load_image("../data/tilt00_000000_rot_tilt__220824.153205.tif")
    tilted_image = correlation.load_image("../data/tilt20_000440_rot_tilt__220824.155523.tif")

    DIR_rot_tilt = r'C:\Users\sergeyg\Github\DATA.Verios.01\stack_rot_tilt__220824.153205'
    DIR = DIR_rot_tilt
    files_names = sorted(glob.glob(os.path.join(DIR, '*.tif')))
    flat_image = correlation.load_image(files_names[0])
    tilted_image = correlation.load_image(files_names[440])

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
                         angle=20,
                         image=flat_image)

    untilted_image = projective_transform_from_points(points_src=src,
                                                      points_dst=dst,
                                                      image=tilted_image,
                                                      inverse=False)


    # values that worked for rotated images and z-scan images
    low_pass = int(max(flat_image.shape) / 6)  # =128 worked for @768x512
    high_pass = int(max(flat_image.shape) / 64)  # =12 worked for @768x512
    sigma = int(max(flat_image.shape) / 256)
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
                                                        plot=False,
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
    ax4.set_title("flat+untilted overlayed")


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


if 0:
    src = \
        [
            [335.35904836,  713.90903593],
            [ 447.66335657,  718.40179455],
            [ 429.67996485,  852.19822898],
            [ 457.73405594,  845.72420796],
            [ 521.03559481,  697.54106015],
            [ 527.50961583,  782.42266909],
            [ 508.80688843,  872.33962771],
            [ 638.06777134,  664.91359201],
            [ 709.95172888,  645.1088282 ],
            [ 643.9358495,   747.80019612],
            [ 714.3527875,   727.9954323 ],
            [ 528.77481549,  865.89526922],
            [ 650.53743744,  826.28574159],
            [ 720.22086567,  812.34905595],
            [ 569.11785288,  688.38590468],
            [ 574.25242128,  767.60495993],
            [ 539.04395228,  986.9243814 ],
            [ 579.38698967,  887.16705257],
            [ 650.53743744,  951.7159124 ],
            [ 722.42139498,  930.44412905],
            [ 260.0013192,   732.08244887],
            [ 380.61023968,  782.50093203],
            [ 309.43120464,  801.28428849],
            [ 384.56463051,  859.61155332],
            [ 317.33998631,  886.30369146],
            [ 245.17235356,  904.09845022],
            [ 442.29826574,  588.01146123],
            [ 434.78244823,  475.27419862],
            [ 425.11925429,  351.80005387],
            [ 352.10845566,  367.9053771 ],
            [ 364.99271424,  490.30583364],
            [ 368.21377889,  615.92735482],
            [ 497.05636472,  326.0315367 ],
            [ 509.9406233,   448.43199324],
            [ 515.30906438,  576.20089086],
            [ 467.85481565,  961.54009836],
            [ 473.25869199, 1085.82925397],
            [ 482.98566938, 1213.36073538],
            [ 392.20054702,  984.23637895],
            [ 399.76597389, 1108.52553456],
            [ 413.81605235, 1231.73391491],
            [ 319.78860419, 1005.85188428],
            [ 329.51558158, 1129.06026462],
            [ 256.02286348, 1031.79049067],
            [ 265.74984088, 1147.43344415],
            [ 318.63550879,  591.57848348],
            [ 288.23206032,  512.0617721 ],
            [ 280.0465165 ,  393.95606842],
            [ 418.03139802,  225.56773842],
            [ 371.25686191,  240.76946266]
    ]

    dst = \
        [
            [ 200.04546961,  678.77840711],
            [ 402.66444886,  716.5748545 ],
            [ 386.5272339 ,  860.01676524],
            [ 432.24934295,  851.05164582],
            [ 534.45170435,  697.74810372],
            [ 545.20984765,  778.4341785 ],
            [ 516.52146551,  872.56793243],
            [ 736.31739451,  660.34496188],
            [ 859.8659533 ,  637.12778305],
            [ 754.55946359,  741.6050878 ],
            [ 873.13291263,  721.7046488 ],
            [ 563.84692318,  861.0077218 ],
            [ 762.85131318,  825.35276859],
            [ 888.05824188,  805.45232959],
            [ 611.93965077,  674.44110618],
            [ 633.49845968,  772.28493126],
            [ 585.4057321 ,  992.01894521],
            [ 652.56971372,  889.20001038],
            [ 788.55604689,  951.38888225],
            [ 911.27542072,  929.83007334],
            [  67.58548763,  696.26871627],
            [ 287.37788447,  782.80115597],
            [ 168.82844208,  805.29959029],
            [ 309.01099439,  865.87229808],
            [ 188.73090321,  887.505408  ],
            [  62.39354125,  910.86916672],
            [ 384.56373423,  593.15895242],
            [ 365.99920433,  464.63528388],
            [ 346.00663367,  338.96769687],
            [ 220.33904665,  358.96026753],
            [ 246.04378036,  490.34001759],
            [ 267.46439178,  616.0076046 ],
            [ 465.96205764,  318.9751262 ],
            [ 488.81070982,  440.35859093],
            [ 511.659362  ,  571.738341  ],
            [ 455.14873197,  977.26108297],
            [ 476.48075526, 1092.45400872],
            [ 492.47977273, 1220.44614844],
            [ 331.42299691,  994.3267016 ],
            [ 349.55521671, 1122.31884132],
            [ 368.75403766, 1248.17777871],
            [ 204.49745836, 1011.39232023],
            [ 230.0958863 , 1139.38445995],
            [  80.7717233 , 1032.72434352],
            [ 101.03714542, 1162.84968556],
            [ 166.80233647,  543.31473107],
            [ 117.37369979,  509.91700358],
            [  97.3350633 ,  384.34154823],
            [ 323.10370111,  212.0092744 ],
            [ 237.60551875,  225.36836539]
        ]


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