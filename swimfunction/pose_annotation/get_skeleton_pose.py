''' Use skeletonization to get poses.
NOTE: this annotates in video coordinates, so if you
used CropTracker to get the video, coordinates will
be in fish frame, not lab frame.
'''
from collections import namedtuple
from swimfunction.global_config.config import config
from skimage.morphology import skeletonize
from scipy import ndimage, signal
from swimfunction.video_processing import extract_frames, image_processing
from zplib.curve import interpolate
import numpy

POINTS_PER_POSE = config.getint('POSES', 'points_per_pose')

ADJACENCY_KERN = numpy.asarray([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

AREA_KERN = numpy.ones(10*10).reshape((10, 10))

def get_fish_lims(mask):
    ''' Locate axis limits that include the fish in the frame.
    '''
    axis0, axis1 = numpy.where(mask)
    axis0lims = (int(max(axis0.min()-10, 0)), int(min(mask.shape[0], axis0.max()+10)))
    axis1lims = (int(max(axis1.min()-10, 0)), int(min(mask.shape[1], axis1.max()+10)))
    return axis0lims, axis1lims

def smooth_sides(mask):
    ''' Smooth the sides of the mask
    '''
    if mask is None or mask.max() == 0:
        return None
    axis0lims, axis1lims = get_fish_lims(mask)
    mini_mask = mask[axis0lims[0]:axis0lims[1], axis1lims[0]:axis1lims[1]]
    mini_mask = ndimage.binary_fill_holes(mini_mask).astype(numpy.uint8)
    # Smooth the mask's sides
    for _ in range(3):
        # Add pixels that have only one adjacent non-fish pixel
        mini_mask[numpy.where(signal.convolve2d(
            mini_mask,
            ADJACENCY_KERN,
            mode='same',
            boundary='wrap'
        ) >= 5)] = 1
    for _ in range(3):
        # Remove pixels that have fewer than two adjacent fish pixels
        mini_mask[numpy.where(signal.convolve2d(
            mini_mask,
            ADJACENCY_KERN,
            mode='same',
            boundary='wrap'
        ) == 1)] = 0
    mask[axis0lims[0]:axis0lims[1], axis1lims[0]:axis1lims[1]] = mini_mask
    return mask

def get_skeleton_img(frame):
    ''' Get skeletonized frame
    '''
    mask = image_processing.threshold(frame)
    mask = image_processing.isolate_object_in_mask(mask)
    mask = smooth_sides(mask)
    if mask is None or mask.max() == 0:
        return namedtuple('SkeletonImageReturnValue', ['skeleton_img', 'mask'])(None, None)
    # medial_axis() creates branches on the skeleton that obscure the centerline.
    # thin() looks good, but is much slower (45 fps).
    # Skeletonize is around 430 fps (almost 3000 fps if you first isolate the fish)
    # Zhang looks goofy at the head (curves to match some off-center part of the nose)
    # Lee looks better than Zhang.
    axis0lims, axis1lims = get_fish_lims(mask)
    skeleton_img = numpy.zeros_like(mask)
    skeleton_img[axis0lims[0]:axis0lims[1], axis1lims[0]:axis1lims[1]] = \
        skeletonize(mask[axis0lims[0]:axis0lims[1], axis1lims[0]:axis1lims[1]], method='lee')
    return namedtuple('SkeletonImageReturnValue', ['skeleton_img', 'mask'])(skeleton_img, mask)

def get_ordered_points(skeleton_image, mask):
    ''' Returns points in (y, x) order so that matplotlib plays nicely.
    '''
    if skeleton_image is None or mask is None:
        return None
    def get_relative_distances(p, arr):
        return numpy.sum(numpy.power(numpy.asarray(arr) - p, 2), axis=1)
    axis0lims, axis1lims = get_fish_lims(mask)
    # Shift points into mini coordinate space (speeds up)
    mini_img = skeleton_image[axis0lims[0]:axis0lims[1], axis1lims[0]:axis1lims[1]]
    mini_mask = mask[axis0lims[0]:axis0lims[1], axis1lims[0]:axis1lims[1]]
    adjacency = signal.convolve2d(mini_img, ADJACENCY_KERN, mode='same', boundary='wrap')
    mask_area_conv = signal.convolve2d(mini_mask, AREA_KERN, mode='same', boundary='wrap')
    xx_ends, yy_ends = numpy.where(numpy.logical_and(adjacency == 1, mini_img == 1))
    if len(xx_ends) < 2 or len(yy_ends) < 2:
        return None
    xx, yy = numpy.where(mini_img == 1)
    points = [[xx_ends[0], yy_ends[0]]]
    if mask_area_conv[xx_ends[0], yy_ends[0]] < mask_area_conv[xx_ends[1], yy_ends[1]]:
        points = [[xx_ends[1], yy_ends[1]]]
    remaining = [[x, y] for x, y in zip(xx, yy) if [x, y] != points[0]]
    while len(remaining) != 0:
        p = points[-1]
        next_i = get_relative_distances(p, remaining).argmin()
        points.append(remaining[next_i])
        remaining.pop(next_i)
    mini_ordered_points = numpy.array(points).take((1, 0), axis=1)
    # shifts points back into full coordinate space
    ordered_points = mini_ordered_points + [axis1lims[0], axis0lims[0]]
    return ordered_points

def frame_to_skeleton_points(frame):
    ''' Frame to coordinates of the skeleton keypoints.
    '''
    return get_ordered_points(*get_skeleton_img(frame))

def frame_to_smoothed_skeleton(frame):
    ''' Get smoothed skeleton from the frame
    '''
    pose = frame_to_skeleton_points(frame)
    if pose is not None and not numpy.any(numpy.isnan(pose)):
        pose = interpolate.spline_interpolate(interpolate.fit_spline(pose), POINTS_PER_POSE)
    else:
        pose = numpy.full((POINTS_PER_POSE, 2), numpy.nan)
    return pose

def get_skeleton_pose_with_frame(fish_name, assay_label, frame_num):
    ''' Get keypoint coordinates for
    the spline-smoothed skeleton
    calculated from the requested frame.
    '''
    frame = extract_frames.extract_frame(
        fish_name, assay_label, frame_num,
        frame_nums_are_full_video=True,
        transpose_to_match_convention=True)
    pose = frame_to_smoothed_skeleton(frame)
    return namedtuple('PoseAndFrame', ['pose', 'frame'])(pose, frame)
