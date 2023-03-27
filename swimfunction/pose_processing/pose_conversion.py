''' Converts between poses of different types.
'''
from swimfunction.global_config.config import config
import numpy

POINTS_PER_POSE = config.getint('POSES', 'points_per_pose')
ANGLE_POSE_SIZE = POINTS_PER_POSE - 2

def _rotate(points: numpy.ndarray, angle: float) -> numpy.ndarray:
    ''' Rotates points in space.
    '''
    points = numpy.asarray(points)
    rotation_matrix = numpy.array([
        [numpy.cos(angle), -numpy.sin(angle)],
        [numpy.sin(angle), numpy.cos(angle)]])
    if len(points.shape) == 1:
        return rotation_matrix.dot(points)
    points = rotation_matrix.T.dot(points.T).T
    if points[-1][0] < 0:
        points[:, 0] = -1 * points[:, 0]
    return points

def angles_pose_to_points(
        angle_pose: numpy.ndarray,
        rotation_angle: float) -> numpy.ndarray:
    '''Converts angles to (x, y) coordinates.
    Uses scale of 1 unit per fish line (2 per angle),
    so shape is preserved but relative fish size is not.
    Starts at the origin (0, 0)

    Parameters
    ----------
    angle_pose : numpy.ndarray
        angle poses
    rotation_angle : number
        amount to rotate pose about the x-axis

    Returns
    -------
    points : numpy.ndarray
        shape = (num_points, 2)
    '''
    angle_pose = numpy.asarray(angle_pose)
    def _angle_from_x_axis(point):
        return numpy.arctan2(point[1], point[0])
    # We always start with a point at the origin and a point at (1, 0)
    points = list(_rotate([numpy.array([0, 0]), numpy.array([1, 0])], -rotation_angle))
    for angle_to_next in angle_pose:
        # Convert most recent two points to vector from origin:
        first_line = points[-1] - points[-2]
        # Get angle to the x-axis
        x_angle = _angle_from_x_axis(first_line)
        # Rotate unit vector to match the original point's rotation about the horizon
        v_next = _rotate([1, 0], angle_to_next + x_angle)
        # Add this vector onto the end of the line
        points.append(points[-1] + v_next)
    return numpy.asarray(points)

def complete_angles_poses_to_angles_pose(
        complete_poses: numpy.ndarray) -> numpy.ndarray:
    ''' Returns the portion of a complete pose that is an angle pose.

    Parameters
    ----------
    complete_poses : numpy.ndarray
        The last axis are complete poses. Can be one or many complete poses.
    '''
    complete_poses = numpy.asarray(complete_poses)
    return complete_poses[..., 3:]

def complete_angles_pose_to_points(
        complete_pose: numpy.ndarray) -> numpy.ndarray:
    '''
    Parameters
    ----------
    complete_pose : numpy.ndarray or list
        [headX, headY, headAngle, a1, a2, ... , an]
    '''
    complete_pose = numpy.asarray(complete_pose)
    return angles_pose_to_points(complete_pose[3:], complete_pose[2]) + complete_pose[:2]

def points_to_complete_angles(
        point_poses: numpy.ndarray) -> numpy.ndarray:
    ''' (parallelized, multiple poses supported)
    Represent a pose completely by its head coordinates,
    angle of the head, and angles between points.
    Converts pose to a list of all inner angles, defined by three points per angle.
    Define points[i-1] is a, points[i] is b, points[i+1] is c
    v1 = b-a
    v2 = c-b
    arccos( v1 dot v2 / (norm(v1) * norm(v2)) )
    the sign is determined by the cross product of the two vectors.

    NOTE: To protect arccos from adding nans into the mix
    (due to float point precision issues), I clip values between -1 and 1.

    Parameters
    ----------
    points : numpy.ndarray
        shape (num_poses, points_per_pose, 2)

    Returns
    -------
    complete_angles : numpy.ndarray
        [headX, headY, headAngle, a1, a2, ..., aN ]
        where a{n} are angles between point trios, from head to tail.
    '''
    point_poses = numpy.asarray(point_poses)
    num_poses, points_per_pose = point_poses.shape[:2]
    vectors_extended = numpy.empty_like(point_poses)
    vectors_extended[:, 1:, :] = point_poses[:, 1:, :] - point_poses[:, :-1, :]
    vectors_extended[:, 0, 0] = 1
    vectors_extended[:, 0, 1] = 0
    angles = numpy.arccos(numpy.clip(
        numpy.sum(vectors_extended[:, :-1, :] * vectors_extended[:, 1:, :], axis=2) /
        (numpy.linalg.norm(vectors_extended[:, :-1, :], axis=2) *
         numpy.linalg.norm(vectors_extended[:, 1:, :], axis=2)),
        -1, 1)) * numpy.sign(numpy.cross(vectors_extended[:, :-1, :], vectors_extended[:, 1:, :]))
    complete_angles = numpy.empty((num_poses, points_per_pose+1))
    complete_angles[:, 0] = point_poses[:, 0, 0]
    complete_angles[:, 1] = point_poses[:, 0, 1]
    complete_angles[:, 2:] = angles
    return complete_angles
