import numpy
from zplib.curve import interpolate
from tqdm import tqdm

from swimfunction.pose_processing.pose_cleaners.AbstractPoseCleaner import AbstractPoseCleaner
from swimfunction.pose_annotation import get_skeleton_pose
from swimfunction.global_config.config import config

POINTS_PER_POSE = config.getint('POSES', 'points_per_pose')
MIN_LIKELIHOOD_THRESHOLD = config.getfloat('POSE FILTERING', 'min_likelihood_threshold')
MAX_EXPECTED_FISH_LENGTH = config.getfloat('POSES', 'max_expected_fish_length')

def is_extreme(coordinate_pose, mean_head_tail_dist):
    ''' A pose is extreme if the distance
    between adjacent points is greater
    than half the length of the fish.
    '''
    return numpy.any(
        numpy.linalg.norm(
            coordinate_pose[:-1, :] - coordinate_pose[1:, :], axis=1
        ) > mean_head_tail_dist / 2)

def resample_pose_smoother_keep_not_extreme(
        point_pose: numpy.ndarray,
        mean_head_tail_dist: float=MAX_EXPECTED_FISH_LENGTH):
    ''' Use spline smoothing.
    If the result is extreme, return the original.
    '''
    new_pose = interpolate.spline_interpolate(
        interpolate.fit_spline(point_pose),
        POINTS_PER_POSE)
    if not is_extreme(new_pose, mean_head_tail_dist):
        point_pose = new_pose
    return point_pose

def resample_pose_smoother_skeleton_fix(
        point_pose,
        fish_name,
        assay_label,
        frame_num,
        min_likelihood,
        mean_head_tail_dist=MAX_EXPECTED_FISH_LENGTH):
    ''' Resamples smoother. Uses skeleton if the pose is
    too extreme to begin with (fixes some DLC issues).
    '''
    if numpy.all(numpy.isnan(point_pose)):
        return point_pose
    if is_extreme(point_pose, mean_head_tail_dist) \
            and fish_name is not None \
            and assay_label is not None \
            and frame_num is not None \
            and min_likelihood >= MIN_LIKELIHOOD_THRESHOLD:
        new_pose = get_skeleton_pose.get_skeleton_pose_with_frame(fish_name, assay_label, frame_num).pose
        # Only reassign if no longer extreme.
        if new_pose is not None \
                and not numpy.any(numpy.isnan(new_pose)) \
                and not is_extreme(new_pose, mean_head_tail_dist):
            point_pose = new_pose
    else:
        point_pose = resample_pose_smoother_keep_not_extreme(
            point_pose, mean_head_tail_dist)
    return point_pose

class SplineSmootherNoVideoAvailable(AbstractPoseCleaner):
    ''' Smooths poses using splines.
    If still extreme after spline smoothing, keeps original pose.
    '''
    def clean(
            self,
            poses: numpy.ndarray,
            likelihoods: numpy.ndarray,
            *args,
            **kwargs):
        ''' Resample the pose from a spline. If the pose is extreme
        (according to mean_head_tail_dist which is calculated
        from high likelihood poses, if provided.)
        then the raw points are returned.

        Parameters
        ----------
        poses : numpy.ndarray or list
        likelihoods: list, default=None

        Returns
        -------
        numpy.ndarray
            shape (N, k, 2) where N is number of poses and k is number of points per pose
        '''
        rv = numpy.empty_like(poses)
        if len(poses.shape) == 3:
            likelihood_mins = numpy.ones(poses.shape[0])
            if likelihoods is not None:
                likelihood_mins = likelihoods.min(axis=1)
            # Each fish has its own mean head-tail distance for reference. Keeps things reasonable.
            where_best = numpy.where(likelihood_mins >= MIN_LIKELIHOOD_THRESHOLD)[0]
            # Still can't be above maximum, though.
            mean_head_tail_dist = min((
                MAX_EXPECTED_FISH_LENGTH,
                numpy.mean(
                    numpy.linalg.norm(
                        poses[where_best, 0, :] - poses[where_best, -1, :],
                        axis=1))))
            # Resample every point smoother based on this
            # fish's mean_head_tail_dist (which is at most MAX_EXPECTED_FISH_LENGTH)
            for i in tqdm(range(poses.shape[0])):
                rv[i, :, :] = resample_pose_smoother_keep_not_extreme(
                    poses[i, :, :], mean_head_tail_dist)
        else:
            rv[:, :] = resample_pose_smoother_keep_not_extreme(poses)
        return rv

class SplineSmootherVideoAvailable(AbstractPoseCleaner):
    ''' Smooths poses using splines.
    If still extreme after spline smoothing, tries skeletonization.
    If still extreme after skeletonization, keeps original pose.
    '''
    def clean(
            self,
            poses: numpy.ndarray,
            likelihoods: numpy.ndarray,
            fish_name: str,
            assay_label: int, *args, **kwargs):
        ''' Resample the pose from a spline. If the pose is extreme
        (according to mean_head_tail_dist
        which is calculated from high likelihood poses, if provided)
        then it tries to get a skeleton pose (this is the skeleton fix).
        If that pose is also extreme, then the raw points are returned.

        NOTE: this only works for single animal frames!

        Returns
        -------
        smoothed_point_poses : numpy.ndarray
            shape (N, k, 2) where N is number of poses and k is number of points per pose
        '''
        smoothed = numpy.empty_like(poses)
        likelihood_mins = numpy.ones(poses.shape[0])
        if likelihoods is not None:
            likelihood_mins = likelihoods.min(axis=1)
        # Each fish has its own mean head-tail distance for reference. Keeps things reasonable.
        where_best = numpy.where(likelihood_mins >= MIN_LIKELIHOOD_THRESHOLD)[0]
        # Still can't be above MAX_EXPECTED_FISH_LENGTH, though.
        mean_head_tail_dist = min((
            MAX_EXPECTED_FISH_LENGTH,
            numpy.mean(numpy.linalg.norm(
                poses[where_best, 0, :] - poses[where_best, -1, :], axis=1))))
        # Resample every point smoother based on this
        # fish's mean_head_tail_dist (which is at most MAX_EXPECTED_FISH_LENGTH)
        for i in tqdm(range(poses.shape[0])):
            smoothed[i, :, :] = resample_pose_smoother_skeleton_fix(
                poses[i, :, :],
                fish_name, assay_label, i, likelihood_mins[i], mean_head_tail_dist)
        return smoothed
