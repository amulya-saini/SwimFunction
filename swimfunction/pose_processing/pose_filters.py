'''
Each filter returns the new_mask from the
parameter list as updated according to its filtering criteria.
Each filter function must have the following parameters:

Parameters
----------
swim_assay : SwimAssay
    swim object (only necessary for some functions)
'''

import functools
import numba
import numpy
from scipy import stats as spstats

from swimfunction.data_access.data_utils import AnnotationTypes
from swimfunction.global_config.config import config
from swimfunction.data_models.SwimAssay import SwimAssay

ANGLE_CHANGE_Z_LIM = config.getint('POSE FILTERING', 'angle_change_z_lim')
FRAME_SIDE_BUFFER_WIDTH = config.getint('POSE FILTERING', 'frame_side_buffer_width')
FRAME_REAR_BUFFER_WIDTH = config.getint('POSE FILTERING', 'frame_rear_buffer_width')
MIN_LIKELIHOOD_THRESHOLD = config.getfloat('POSE FILTERING', 'min_likelihood_threshold')

NO_FLOW = config.getint('FLOW', 'none')
SLOW_FLOW = config.getint('FLOW', 'slow')
FAST_FLOW = config.getint('FLOW', 'fast')

NO_FLOW_START = config.getint('FLOW', 'none_start')
SLOW_FLOW_START = config.getint('FLOW', 'slow_start')
FAST_FLOW_START = config.getint('FLOW', 'fast_start')

ORIENTED_WITH = config.getint('RHEOTAXIS', 'with_flow')
ORIENTED_AGAINST = config.getint('RHEOTAXIS', 'against_flow')
NOT_QUITE_AGAINST = config.getint('RHEOTAXIS', 'not_quite_against_flow')
NOT_QUITE_WITH = config.getint('RHEOTAXIS', 'not_quite_with_flow')

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')

ANGLE_LOWER_BOUNDS = config.getfloatlist('POSE FILTERING', 'angle_lower_bounds')
ANGLE_UPPER_BOUNDS = config.getfloatlist('POSE FILTERING', 'angle_upper_bounds')

def _ones_mask(swim_assay: SwimAssay):
    ''' Get a blank mask of "True" for the swim assay.
    '''
    return numpy.ones(swim_assay.raw_coordinates.shape[0], dtype=bool)

def _zeros_mask(swim_assay: SwimAssay):
    ''' Get a blank mask of "False" for the swim assay.
    '''
    return numpy.zeros(swim_assay.raw_coordinates.shape[0], dtype=bool)

@numba.njit(parallel=True, fastmath=True)
def _get_fish_lengths(poses):
    ''' For each pose: uses the Pythagorean Theorem
    to get segment lengths given by adjacent points,
    then sums the segment lengths.

    Parameters
    ----------
    poses: numpy.ndarray
        Array with shape=(num_poses, points_per_pose, 2)
    '''
    return numpy.sqrt(
        numpy.square(
            poses[:, :-1, 0] - poses[:, 1:, 0]
        ) + numpy.square(
            poses[:, :-1, 1] - poses[:, 1:, 1]
        )
    ).sum(axis=1)

def filter_by_cleaned(swim_assay: SwimAssay, *args, **kwargs):
    return numpy.logical_not(swim_assay.was_cleaned_mask.astype(bool))

def filter_by_likelihood(swim_assay: SwimAssay, *args, **kwargs):
    ''' Filters by DeepLabCut (DLC) likelihood score
    '''
    new_mask = _ones_mask(swim_assay)
    if swim_assay.likelihoods is None:
        return new_mask
    mins = swim_assay.likelihoods.min(axis=1)
    new_mask[numpy.where(mins < MIN_LIKELIHOOD_THRESHOLD)] = 0
    return new_mask

def filter_by_distance_from_neighbors(swim_assay: SwimAssay, *args, **kwargs):
    ''' Filters by comparing the distance between adjacent poses,
    and masking out any pose with distance z-score greater than ANGLE_CHANGE_Z_LIM.
    '''
    poses = swim_assay.smoothed_complete_angles
    # Ensure poses are between -pi and pi
    poses[numpy.logical_and(
        numpy.logical_not(numpy.isnan(poses)),
        poses > numpy.pi)] -= numpy.pi
    poses[numpy.logical_and(
        numpy.logical_not(numpy.isnan(poses)),
        poses < (-1 * numpy.pi))] += numpy.pi
    # The smallest difference between 2 angles
    # (https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles)
    differences = numpy.abs(numpy.arctan2(
        numpy.sin(numpy.abs(poses[:-1, :]) - numpy.abs(poses[1:, :])),
        numpy.cos(numpy.abs(poses[:-1, :]) - numpy.abs(poses[1:, :]))
    ))
    diff_mean = numpy.nanmean(differences)
    diff_std = numpy.nanstd(differences)
    max_differences = numpy.nanmax(differences, axis=1)
    # Absolute value zs
    # print(f'Differences min: {numpy.nanmin(differences)}')
    # print(f'Differences mean: {diff_mean}')
    # print(f'Differences max: {numpy.nanmax(differences)}')
    diff_zs = numpy.abs(max_differences - diff_mean) / diff_std
    max_zs = numpy.empty(poses.shape[0])
    max_zs[0] = diff_zs[0]
    max_zs[-1] = diff_zs[-1]
    max_zs[1:-1] = numpy.min(numpy.array((diff_zs[:-1], diff_zs[1:])), axis=0)
    new_mask = _ones_mask(swim_assay)
    new_mask[max_zs > ANGLE_CHANGE_Z_LIM] = 0
    new_mask[numpy.isnan(max_zs)] = 0
    return new_mask

def filter_by_distance_from_edges(swim_assay: SwimAssay, *args, **kwargs):
    ''' If the fish is too close to the back or left/right sides, mask that frame!
    '''
    from swimfunction.data_access.fish_manager import DataManager as FDM
    new_mask = _ones_mask(swim_assay)
    coordinate_poses = swim_assay.smoothed_coordinates
    # NOTE: maxy is not important. Flow comes from maxy, so if the fish swims really well,
    # then it may be close to maxy and that's ok.
    minx, miny, maxx, maxy = FDM.get_corners_of_swim_area(swim_assay.fish_name, swim_assay.assay_label)
    new_mask[numpy.any(coordinate_poses[..., 0] < (minx + FRAME_SIDE_BUFFER_WIDTH), axis=1)] = 0
    new_mask[numpy.any(coordinate_poses[..., 0] > (maxx - FRAME_SIDE_BUFFER_WIDTH), axis=1)] = 0
    # new_mask[numpy.any(coordinate_poses[..., 1] > (maxy - FRAME_REAR_BUFFER_WIDTH), axis=1)] = 0
    new_mask[numpy.any(coordinate_poses[..., 1] < (miny + FRAME_REAR_BUFFER_WIDTH), axis=1)] = 0
    return new_mask

def filter_away_inactive(swim_assay: SwimAssay, *args, **kwargs):
    return numpy.logical_not(TEMPLATE_filter_by_behavior(
        BEHAVIORS.rest,
        annotation_type=AnnotationTypes.predicted)(swim_assay))

def filter_raw_coordinates_nan(swim_assay: SwimAssay, *args, **kwargs):
    axis = tuple(range(1, len(swim_assay.raw_coordinates.shape)))
    return ~numpy.any(numpy.isnan(swim_assay.raw_coordinates), axis=axis)

def filter_caudal_fin_nan(swim_assay: SwimAssay, *args, **kwargs):
    axis = tuple(range(1, len(swim_assay.smoothed_coordinates_with_caudal.shape)))
    return ~numpy.any(numpy.isnan(swim_assay.smoothed_coordinates_with_caudal), axis=axis)

def filter_extreme_angles(swim_assay: SwimAssay, *args, **kwargs):
    from swimfunction.data_access.PoseAccess import get_feature_from_assay
    poses = get_feature_from_assay(swim_assay, 'smoothed_angles', [], keep_shape=True)
    mask = _ones_mask(swim_assay)
    mask[numpy.any(poses < ANGLE_LOWER_BOUNDS, axis=1) | numpy.any(poses > ANGLE_UPPER_BOUNDS, axis=1)] = 0
    return mask

def filter_by_fish_length(swim_assay: SwimAssay, *args, **kwargs):
    ''' Filters by comparing the fish's length to all other lengths in this swim,
    and masking out any pose with length outside the range (median ± 8 * MAD)
    '''
    # Calculate stats on a good subset
    check_mask = filter_raw_coordinates_nan(swim_assay) & filter_extreme_angles(swim_assay)
    coordinate_poses = swim_assay.smoothed_coordinates
    lengths = _get_fish_lengths(coordinate_poses)
    med = numpy.nanmedian(lengths[check_mask])
    mad = spstats.median_abs_deviation(lengths[check_mask], nan_policy='omit')
    # Apply to all lengths, even if they have extreme angles.
    new_mask = _ones_mask(swim_assay)
    upper_limit = med + (8 * mad)
    lower_limit = max((med - (8 * mad)), 0)
    new_mask[lengths >= upper_limit] = 0
    new_mask[lengths <= lower_limit] = 0
    new_mask[numpy.isnan(lengths)] = 0
    return new_mask

def TEMPLATE_filter_by_highest_recovery_score(min_recovery_score: int):
    ''' Checks get_highest_recovered_quality
    for the target recovery score. Masks out the entire SwimAssay if the target is not achieved.
    '''
    from swimfunction.data_access.fish_manager import DataManager as FDM
    def inner(swim_assay: SwimAssay, *args, **kwargs):
        new_mask = _ones_mask(swim_assay)
        quality = FDM.get_highest_recovered_quality(swim_assay.fish_name)
        if quality is None or quality < min_recovery_score:
            new_mask[:] = False
        return new_mask
    return inner

def TEMPLATE_filter_by_behavior(behavior: str, annotation_type: str = AnnotationTypes.all):
    if behavior not in BEHAVIORS:
        raise RuntimeError(f'Behavior must be in {list(BEHAVIORS.keys())}')
    from swimfunction.data_access.PoseAccess import get_behaviors
    def inner(swim_assay: SwimAssay, *args, **kwargs):
        new_mask = _zeros_mask(swim_assay)
        behaviors = get_behaviors(swim_assay, annotation_type)
        if behaviors is not None and len(behaviors) > 0:
            new_mask[numpy.where(numpy.asarray(behaviors) == behavior)] = 1
        return new_mask
    return inner

def TEMPLATE_filter_by_flow_rate(flow_speed):
    '''
    Parameters
    ----------
    flow_speed : int
        see config.ini for available values

    Raises
    ------
    ValueError
        If flow_speed is not recognized in config.ini
    '''
    def inner(swim_assay: SwimAssay, *args, **kwargs):
        mask = _zeros_mask(swim_assay)
        if flow_speed == FAST_FLOW:
            mask[FAST_FLOW_START:] = 1
        elif flow_speed == SLOW_FLOW:
            mask[SLOW_FLOW_START:FAST_FLOW_START] = 1
        elif flow_speed == NO_FLOW:
            mask[NO_FLOW_START:SLOW_FLOW_START] = 1
        else:
            raise ValueError(f'Flow speed ({flow_speed}) not recognized.')
        return mask
    return inner

def TEMPLATE_rheotaxis(allowed_orientations: list):
    ''' Keep in mind that rheotaxis only applies during flow.
    Do not use this filter for pre-flow (first five minutes of video) analysis.
    Parameters
    ----------
    orientation : int
        See config.ini
        Options are
            1 (with flow >= 45 degrees),
            -1 (against flow >= 45 degrees), or
            0 (not oriented to flow).
    '''
    allowed_orientations = set(allowed_orientations)
    presence_fn = numpy.vectorize(lambda x: x in allowed_orientations)
    def inner(swim_assay: SwimAssay, *args, **kwargs):
        from swimfunction.data_access.PoseAccess import get_feature_from_assay
        from swimfunction.recovery_metrics.metric_analyzers\
            .swim_capacity_calculations import pose_orientations
        coords = get_feature_from_assay(
            swim_assay,
            'smoothed_coordinates',
            filters=[],
            keep_shape=True)
        return presence_fn(pose_orientations(coords))
    return inner

def get_filter_mask(swim_assay: SwimAssay, filters=None):
    ''' Given a swim assay and filters, get a boolean mask for keep-able poses.
    '''
    if filters is None:
        filters = []
    masks = [f(swim_assay) for f in filters] + [_ones_mask(swim_assay)]
    return functools.reduce(lambda a, b: a & b, masks)

def apply_filter_mask(mask, poses, keep_shape=False, fill_value=numpy.nan):
    ''' Keeps poses where mask is 1.
    If keep_shape fills with fill_value where mask is 0,
    otherwise it removes that pose from the array.
    '''
    if not numpy.issubdtype(mask.dtype, bool):
        mask = mask.astype(bool)
    new_poses = poses.copy()
    if keep_shape:
        new_poses[numpy.logical_not(mask), ...] = fill_value
    else:
        new_poses = new_poses[mask, ...]
    return new_poses

# Basic filters are those that are required
# for reasonable posture estimation, regardless of behavior.
BASIC_FILTERS = [
    filter_raw_coordinates_nan,
    filter_by_likelihood,
    filter_by_fish_length,
    filter_by_cleaned,
    filter_extreme_angles
]
