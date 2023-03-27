''' Swim capcaity calculation helper functions

NOTE: flow comes from ymax, then pushes the fish toward y=0.
Moving in positive y direction means swimming against the flow.

    These calculations match as closely as possible to those described in
    Burris, B., Jensen, N., Mokalled, M. H.
    Assessment of Swim Endurance and Swim Behavior in Adult Zebrafish.
    J. Vis. Exp. (177), e63240, doi:10.3791/63240 (2021).
'''

import traceback
import threading
from collections import namedtuple
from scipy.ndimage import gaussian_filter
import numpy

from swimfunction.pose_processing import pose_conversion
from swimfunction.video_processing import CropTracker
from swimfunction.video_processing import extract_frames
from swimfunction.global_config.config import config
from swimfunction import loggers, FileLocations

REST_EPSILON = config.getfloat('BEHAVIOR_ANNOTATION', 'rest_epsilon')

def get_logger() -> loggers.logging.Logger:
    ''' Get the default metric calculation logger.
    '''
    return loggers.get_metric_calculation_logger(__name__)

CROP_HEIGHT = config.getint('VIDEO', 'crop_height')
SCALE_DIVISOR = config.getint('VIDEO', 'scale_divisor')

NO_FLOW = config.getint('FLOW', 'none')
SLOW_FLOW = config.getint('FLOW', 'slow')
FAST_FLOW = config.getint('FLOW', 'fast')

# This is our "not no flow" flag
HAS_FLOW = config.getint('FLOW', 'has_flow')

SLOW_START = config.getint('FLOW', 'slow_start')
FAST_START = config.getint('FLOW', 'fast_start')
EXPECTED_FINAL_FRAME = config.getint('FLOW', 'expected_final_frame')

ORIENTED_WITH = config.getint('RHEOTAXIS', 'with_flow')
ORIENTED_AGAINST = config.getint('RHEOTAXIS', 'against_flow')
NOT_QUITE_AGAINST = config.getint('RHEOTAXIS', 'not_quite_against_flow')
NOT_QUITE_WITH = config.getint('RHEOTAXIS', 'not_quite_with_flow')
NOT_ACTIVE = 111


FLOWS = (NO_FLOW, SLOW_FLOW, FAST_FLOW)
FLOW_TO_CMPS = {
    NO_FLOW: 0,
    SLOW_FLOW: 10,
    FAST_FLOW: 20
}

FPS = config.getint('VIDEO', 'fps')
TUNNEL_LENGTH = config.getint('VIDEO', 'tunnel_length_cm')

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')

RLE_TUPLE = namedtuple('RLE_TUPLE', ['lengths', 'positions', 'values'])

def rle(inarray):
    ''' https://stackoverflow.com/a/32681075
        run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)
    '''
    ia = numpy.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0:
        return RLE_TUPLE(None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = numpy.append(numpy.where(y), n - 1)   # must include last element posi
        z = numpy.diff(numpy.append(-1, i))       # run lengths
        p = numpy.cumsum(numpy.append(0, z))[:-1] # positions
        return RLE_TUPLE(z, p, ia[i])

class AssayProperties:
    ''' Holds important constants related to the assay,
    its size, and its flow. All units: distance is px, time is frames.
    '''
    DATA_ACCESS_LOCK = threading.Lock()
    CACHE_DICT = {}

    __slots__ = [
        'width',
        'height',
        'exhaustion_boundary',
        'centroid_activity_threshold',
        'max_expected_drift'
    ]
    def __init__(self, height, width, is_resized):
        ''' Please use the static method `get_assay_properties` to initialize.
        Sets up helpful constants given video dimensions.
        '''
        self.height = height
        self.width = width
        px_per_cm = self.height / TUNNEL_LENGTH
        self.exhaustion_boundary = 3 * px_per_cm
        # Centroid activity threshold is 5 px for videos at 30 fps.
        # So, we convert to the correct FPS below.
        self.centroid_activity_threshold = 5 * (30 / FPS)
        if is_resized:
            self.centroid_activity_threshold /= SCALE_DIVISOR
        self.max_expected_drift = {}
        for flow_key in FLOWS:
            cmps = FLOW_TO_CMPS[flow_key]
            self.max_expected_drift[flow_key] = cmps * px_per_cm / FPS

    @staticmethod
    def get_video_shape(fish_name, assay_label):
        ''' Get the (height, width) tuple of a video's shape.
        '''
        vfile = FileLocations.find_video(fish_name, assay_label)
        logfile = CropTracker.find_existing_logfile(vfile)
        height_width = (
            config.getint('VIDEO', 'default_video_height'),
            config.getint('VIDEO', 'default_video_width')
        )
        if logfile is None and vfile is not None and vfile.exists():
            frame = extract_frames.extract_frame(
                fish_name, assay_label, 0, frame_nums_are_full_video=False)
            height_width = frame.shape[:2]
        elif logfile is not None and logfile.exists():
            tracker = CropTracker.CropTracker.from_logfile(logfile)
            height_width = (tracker.full_height, tracker.full_width)
        return height_width

    @staticmethod
    def get_assay_properties(fish_name: str, assay_label: int):
        '''
        Parameters
        ----------
        fish_name : str
            One representative fish
        assay_label : int
            One assay label for that fish
        '''
        key = f'{config.experiment_name}_{fish_name}_{assay_label}'
        if key not in AssayProperties.CACHE_DICT:
            is_resized = False
            video_shape = AssayProperties.get_video_shape(fish_name, assay_label)
            is_resized = abs(video_shape[0] - CROP_HEIGHT) \
                > abs(video_shape[0] - (CROP_HEIGHT / SCALE_DIVISOR))
            with AssayProperties.DATA_ACCESS_LOCK:
                try:
                    AssayProperties.CACHE_DICT[key] = AssayProperties(*video_shape, is_resized)
                except Exception as e:
                    get_logger().error(key)
                    get_logger().error(e)
                    get_logger().error(traceback.format_exc())
        return AssayProperties.CACHE_DICT[key]

def buffer_numpys_to_full_length(arr: numpy.ndarray, buffer_val=numpy.nan):
    ''' Makes sure the array is exactly as long as EXPECTED_FINAL_FRAME.
    Buffers with nan.
    '''
    if arr.shape[1] >= EXPECTED_FINAL_FRAME:
        return arr[:, :EXPECTED_FINAL_FRAME, ...]
    rv = numpy.full(
        (arr.shape[0], EXPECTED_FINAL_FRAME, *arr.shape[2:]),
        buffer_val,
        dtype=arr.dtype)
    rv[:, :arr.shape[1], ...] = arr
    return rv

def buffer_lists_to_full_length(arrs: list, buffer_val=numpy.nan):
    ''' Makes sure the array is exactly as long as EXPECTED_FINAL_FRAME.
    Buffers with nan.
    '''
    final_arrs = [None for _ in arrs]
    for i, arr in enumerate(arrs):
        if arr.shape[0] >= EXPECTED_FINAL_FRAME:
            final_arrs[i] = arr[:EXPECTED_FINAL_FRAME, ...]
        else:
            tmp = numpy.full((EXPECTED_FINAL_FRAME, *arr.shape[1:]), buffer_val, dtype=arr.dtype)
            tmp[:arr.shape[0]] = arr
            final_arrs[i] = tmp
    return final_arrs

def to_rectangular_array(list_of_arrs: list) -> numpy.ndarray:
    ''' Clips right end of arrays to make all the same length.

    Returns
    -------
    rectangle : numpy.ndarray
    '''
    if len(list_of_arrs) == 1:
        return numpy.asarray(list_of_arrs)
    min_size = min([len(a) for a in list_of_arrs])
    return numpy.asarray([
        a[:min_size] for a in list_of_arrs
    ])

def impute_missing(arr: list, assay_properties: AssayProperties, inplace: bool = True) -> list:
    ''' Replaces nan centroids with reasonable estimates.
    Without flow, it's the most recent observation.
    With flow, x is the most recent observation,
    y is the most recent y minus expected drift (min set as 0).

    Parameters
    ----------
    arr : list
        List that may contain missing values
    inplace : bool
        Whether to modify the list itself.

    Returns
    -------
    list
        List with missing values imputed
    '''
    if not inplace:
        arr = arr.copy()

    recent_valid = None
    for i in range(SLOW_START):
        if i >= len(arr):
            break
        if numpy.any(numpy.isnan(arr[i])):
            arr[i] = recent_valid
        else:
            recent_valid = arr[i]
    for i in range(SLOW_START, FAST_START):
        if i >= len(arr):
            break
        if numpy.any(numpy.isnan(arr[i])):
            arr[i, 0] = recent_valid[0]
            arr[i, 1] = min(recent_valid[1] - assay_properties.max_expected_drift[SLOW_FLOW], 0)
        else:
            recent_valid = arr[i]
    for i in range(FAST_START, len(arr)):
        if len(arr) < FAST_START:
            break
        if numpy.any(numpy.isnan(arr[i])):
            arr[i, 0] = recent_valid[0]
            arr[i, 1] = min(recent_valid[1] - assay_properties.max_expected_drift[FAST_FLOW], 0)
        else:
            recent_valid = arr[i]
    return arr

def get_heatmap(centroids: numpy.ndarray, imshape: tuple, sigma: int = 60):
    ''' Creates a gaussian filtered image array where each original pixel
    magnitude is the number of pose points at that location over the whole video.

    Parameters
    ----------
    poses : numpy.ndarray
    imshape : tuple
        (height, width)
    '''
    img = numpy.zeros(imshape, dtype=float)
    # Make them usable as indices
    points =  numpy.clip(centroids.astype(int), [0, 0], [img.shape[1] - 1, img.shape[0] - 1])
    for x, y in points:
        img[y, x] += 1
    return gaussian_filter(img, sigma=sigma)

def _dx_dy(
        centroids: numpy.ndarray,
        flow_speed: int,
        assay_properties: AssayProperties,
        adjust_for_flow: bool,
        use_exhaustion_boundary: bool):
    ''' Frame to frame px displacement in x and y directions (centroids not smoothed)
    If the fish is exhausted, its y-position is very low. In that case,
    it may be resting its caudal fin against the back and its centroid could be greater than y=0.
    So, lest we assume that it's swimming well, we set an exhaustion boundary where
    we will treat motion without adjusting for flow.

    Returns
    -------
    dx, dy
        numpy.ndarray change in px for x and y directions, size=(centroids.shape[0] - 1)
    '''
    if len(centroids.shape) == 1:
        centroids = centroids[numpy.newaxis, :]
    dx = centroids[1:, 0] - centroids[:-1, 0]
    dy = centroids[1:, 1] - centroids[:-1, 1]
    if adjust_for_flow:
        if use_exhaustion_boundary:
            where_above_exhaustion_boundary = numpy.where(
                centroids[:-1, 1] > assay_properties.exhaustion_boundary)
            dy[where_above_exhaustion_boundary] = dy[where_above_exhaustion_boundary] \
                + assay_properties.max_expected_drift[flow_speed]
        else:
            dy += assay_properties.max_expected_drift[flow_speed]
    return dx, dy

def distance_swum(
        centroids: numpy.ndarray,
        flow_speed: int,
        assay_properties: AssayProperties,
        adjust_for_flow: bool,
        use_exhaustion_boundary: bool):
    ''' Frame to frame distance (px) traveled (centroids not smoothed)
    If the fish is exhausted, its y-position is very low. In that case,
    it may be resting its caudal fin against the back and its centroid could be greater than y=0.
    So, lest we assume that it's swimming well, we set an exhaustion boundary where
    we will treat motion without adjusting for flow.

    Returns
    -------
    distances: numpy.ndarray
        px, shape=(centroids.shape[0] - 1)
    '''
    dx, dy = _dx_dy(
        centroids,
        flow_speed,
        assay_properties,
        adjust_for_flow,
        use_exhaustion_boundary)
    return numpy.sqrt((dx**2) + (dy**2))

def swim_speeds(
        centroids: numpy.ndarray,
        flow_speed: int,
        assay_properties: AssayProperties,
        adjust_for_flow: bool,
        use_exhaustion_boundary: bool):
    ''' Frame to frame speeds (centroids not smoothed)

    Returns
    -------
    speeds: numpy.ndarray
        px per second, shape=(centroids.shape[0] - 1)
    '''
    return distance_swum(
        centroids,
        flow_speed,
        assay_properties,
        adjust_for_flow,
        use_exhaustion_boundary) * FPS

def get_full_assay_frame_to_frame_distance_swum(
        centroids: numpy.ndarray,
        adjust_for_flow: bool,
        assay_properties: AssayProperties,
        use_exhaustion_boundary: bool):
    ''' Get distances moved from one frame to the next,
    and adjust for flow if desired.

    Parameters
    ----------
    centroids: numpy.ndarray
        centroids as an array shape=(num_centroids, 2)
    adjust_for_flow: bool
        Whether to adjust for flow
    assay_properties: AssayProperties

    Returns
    -------
    distances: numpy.ndarray
        Pixels distance traveled from one frame to the next.
        If xs and ys have length n, then the return value has length n-1.
    '''
    distances = numpy.empty(centroids.shape[0])
    distances[:SLOW_START] = distance_swum(
        centroids[:SLOW_START+1],
        NO_FLOW, assay_properties,
        adjust_for_flow=adjust_for_flow,
        use_exhaustion_boundary=use_exhaustion_boundary)
    distances[SLOW_START:FAST_START] = distance_swum(
        centroids[SLOW_START:FAST_START+1],
        SLOW_FLOW, assay_properties,
        adjust_for_flow=adjust_for_flow,
        use_exhaustion_boundary=use_exhaustion_boundary)
    distances[FAST_START:-1] = distance_swum(
        centroids[FAST_START:],
        FAST_FLOW, assay_properties,
        adjust_for_flow=adjust_for_flow,
        use_exhaustion_boundary=use_exhaustion_boundary)
    distances[-1] = distances[-2] # Final frame, assume same as previous.
    return distances

def close_gaps(activity: numpy.ndarray):
    ''' Any single frames surrounded by two that are of the same type are relabeled as that type.
    xyx will be relabeled xxx.

    We do this first for low surrounded by high (True), then high surrounded by low (False).
    '''
    to_become_high = numpy.where(
        (activity[0:-2]) \
            & (activity[2:])
            & (~activity[1:-1]))[0] + 1
    activity[to_become_high] = True
    to_become_low = numpy.where(
        (~activity[0:-2]) \
            & (~activity[2:])
            & (activity[1:-1]))[0] + 1
    activity[to_become_low] = False
    return activity

def get_full_assay_frame_to_frame_activity(
        centroids: numpy.ndarray,
        adjust_for_flow: bool,
        assay_properties: AssayProperties,
        use_exhaustion_boundary: bool):
    ''' Get binary mask saying whether the distances moved from one frame to the next,
    adjusted for flow if desired, reached a threshold.

    Does not clean the activity at all.
    You should use _close_gaps after obtaining the activity mask.
        activity_mask = _close_gaps(activity_mask)

    Parameters
    ----------
    centroids: numpy.ndarray
        centroids as an array shape=(num_centroids, 2)
    adjust_for_flow: bool
        Whether to adjust for flow
    assay_properties: AssayProperties

    Returns
    -------
    activity_mask: numpy.ndarray
        True if active at that frame, false otherwise
    '''
    distances = get_full_assay_frame_to_frame_distance_swum(
        centroids,
        adjust_for_flow,
        assay_properties,
        use_exhaustion_boundary)
    return numpy.abs(distances) > assay_properties.centroid_activity_threshold

def get_full_assay_frame_to_frame_activity_pose_derived(coordinate_poses, predicted_behaviors):
    '''
    Does not clean the activity at all.
    You should use _close_gaps after obtaining the activity mask.
        activity_mask = _close_gaps(activity_mask)
    '''
    angle_poses = pose_conversion.complete_angles_poses_to_angles_pose(
        pose_conversion.points_to_complete_angles(coordinate_poses))
    # ## Using distance from mean pose
    # mean_pose = numpy.nanmean(angle_poses, axis=0)
    # distances = numpy.abs(numpy.nanmax(angle_poses - mean_pose, axis=1))
    # return distances > 9000 # need to figure out the limit
    ## Using change from one frame to next
    distances = numpy.zeros(angle_poses.shape[0])
    distances[1:] = numpy.max(numpy.abs(angle_poses[1:] - angle_poses[:-1]), axis=1)
    # If the maximum change in angle is less than rest threshold, it is not active.
    # (Some "unknown" behaviors are truly static.)
    activity = distances > REST_EPSILON
    # If the predicted behavior is rest, it is not active.
    activity[predicted_behaviors == BEHAVIORS.rest] = False
    return activity

def pose_orientations(pose_array: numpy.ndarray):
    ''' Positive is against flow, negative is with flow.
    Flow adjustment handles drift to predict static.
    Without flow adjustment, simply says which direction it's facing (in presence of flow).
    '''
    get_last = False
    if len(pose_array.shape) == 2:
        pose_array = pose_array.reshape(-1, *pose_array.shape)
        get_last = True
    head_vectors = pose_array[:, 0, :] - pose_array[:, 1, :]
    # Angles are relative to vector (1, 0)
    head_angles = numpy.arctan2(head_vectors[:, 1], head_vectors[:, 0])
    head_angles[~numpy.isnan(head_angles)] %= (2 * numpy.pi)
    range_of_error = numpy.deg2rad(45)
    perfectly_against = numpy.deg2rad(90)
    perfectly_with = numpy.deg2rad(270)
    directions = numpy.full(pose_array.shape[0], NOT_QUITE_WITH, dtype=int)
    directions[
        (head_angles > (perfectly_against - range_of_error)) \
        & (head_angles < (perfectly_against + range_of_error))] = ORIENTED_AGAINST
    directions[
        (head_angles > (perfectly_with - range_of_error)) \
         & (head_angles < (perfectly_with + range_of_error))] = ORIENTED_WITH
    directions[
        (head_angles < (perfectly_against - range_of_error)) \
         & (head_angles > 0)] = NOT_QUITE_AGAINST
    directions[
        (head_angles > (perfectly_against + range_of_error)) \
         & (head_angles < numpy.pi)] = NOT_QUITE_AGAINST
    if get_last:
        directions = directions[0]
    return directions

def get_burst_count(activity_mask, directions, flow_speed):
    ''' By creating a burst mask based on whether activity is in any allowable direction,
    we allow bursts to change direction. Previously, run length encoding was taken
    on the directions array, which meant that a burst turning from "against" to "not quite against"
    would have counted as two bursts. It would have overestimated the number of true bursts.
    '''
    directions[~activity_mask] = NOT_ACTIVE
    required_orientation = [ORIENTED_AGAINST, NOT_QUITE_AGAINST]
    if flow_speed == NO_FLOW:
        required_orientation = [ORIENTED_AGAINST, NOT_QUITE_AGAINST, NOT_QUITE_WITH, ORIENTED_WITH]
    burst_mask = numpy.in1d(directions, required_orientation)
    rle_tuple = rle(burst_mask)
    return (rle_tuple.values & (rle_tuple.lengths > 1)).sum()

MetricOptions = namedtuple(
    'MetricOptions',
    ['Centroid', 'PoseDerived', 'INVALID'])(0, 1, -1)

class FlowLimsCalculator:
    ''' Find the start and end of time
    according to flow speed.
    '''
    def __init__(self, ignore_time_without_flow: bool):
        '''
        Parameters
        ----------
        ignore_time_without_flow : bool
            Whether to return lims starting at SLOW_START
            if it asks for all time in the assay.
        '''
        self.ignore_time_without_flow = ignore_time_without_flow

    def get_start_end(self, flow_speed):
        ''' Find the start and end of time
        according to requested flow speed.
        If we should ignore time without flow,
        then only returns time in which water was flowing.
        '''
        start = SLOW_START if self.ignore_time_without_flow else 0
        end = None
        if flow_speed is NO_FLOW:
            end = SLOW_START
        if flow_speed is SLOW_FLOW:
            start, end = (SLOW_START, FAST_START)
        if flow_speed is FAST_FLOW:
            start = FAST_START
        if flow_speed == HAS_FLOW:
            start = SLOW_START
        return (start, end)
