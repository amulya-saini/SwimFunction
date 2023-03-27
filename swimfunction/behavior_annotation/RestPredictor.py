''' Predict rests using a pose change threshold.
'''

import functools
import numpy
from swimfunction import progress
from scipy import optimize

from swimfunction.data_access import BehaviorAnnotationDataManager as behavior_data_manager
from swimfunction.data_models import WindowsDataSet
from swimfunction.behavior_annotation.AbstractBehaviorPredictor import AbstractBehaviorPredictor, BEHAVIORS
from swimfunction.pose_processing.pose_conversion import ANGLE_POSE_SIZE

from swimfunction.global_config.config import config

# This value was the output of the train_models function, stored for prediction.
REST_EPSILON = config.getfloat('BEHAVIOR_ANNOTATION', 'rest_epsilon')

class RestPredictor(AbstractBehaviorPredictor):

    def __init__(self, feature: str, rest_epsilon: float = REST_EPSILON):
        '''
        Parameters
        ----------
        feature: str
            Type of poses that will be input
            (e.g., smoothed_angles, smoothed_complete_angles)
        rest_epsilon: float
            If all angles in the pose changed by less than this
            compared to the previous or next pose, it is a "rest" pose.
        '''
        if feature not in ['smoothed_angles', 'smoothed_complete_angles']:
            raise RuntimeError('Feature must be "smoothed_angles" or "smoothed_complete_angles"')
        if feature == 'smoothed_complete_angles':
            raise NotImplementedError()
        self.rest_epsilon = rest_epsilon
        self.feature = feature
        self.numeric_rest = self.labels_to_numeric([BEHAVIORS.rest])[0]

    def no_change(self, a: numpy.ndarray, b: numpy.ndarray):
        return numpy.max(numpy.abs(a - b)) < self.rest_epsilon

    def maximum_distance_to_an_adjacent_pose_pw(
            self, pose_windows: numpy.ndarray, direction: int) -> numpy.ndarray:
        ps = ANGLE_POSE_SIZE # Dimension of one pose
        ppw = WindowsDataSet.PPW # Poses per window
        middle_pose = (ppw // 2) * ps
        pose_windows = numpy.asarray(pose_windows)
        dists = None
        if direction == 1:
            # max distance between pose and next pose
            dists = numpy.abs(
                pose_windows[:, middle_pose:middle_pose+ps] - pose_windows[:, middle_pose+ps:middle_pose+(ps*2)]
            ).max(axis=1)
        elif direction == -1:
            # max distance between pose and previous pose
            dists = numpy.abs(
                pose_windows[:, middle_pose:middle_pose+ps] - pose_windows[:, middle_pose-ps:middle_pose]
            ).max(axis=1)
        else:
            raise ValueError('direction must be -1 or 1 (adjacent).')
        return dists

    def maximum_distance_to_an_adjacent_pose(
            self, sequential_poses: numpy.ndarray, direction: int) -> numpy.ndarray:
        sequential_poses = numpy.asarray(sequential_poses)
        APS = ANGLE_POSE_SIZE
        if sequential_poses.shape[-1] != APS:
            raise RuntimeError('Can only find distances between angle poses!!!!')
        if len(sequential_poses.shape) == 2:
            sequential_poses = sequential_poses.reshape((1, *sequential_poses.shape))
        if not len(sequential_poses.shape) == 3:
            raise RuntimeError('Must give a list of sequential poses or multiple lists of sequential poses.')
        dists = numpy.zeros(sequential_poses.shape[:-1])
        if direction == 1:
            # max distance between pose and next pose
            dists[:, :-1] = numpy.abs(
                sequential_poses[:, :-1] - sequential_poses[:, 1:]
            ).max(axis=2)
        elif direction == -1:
            # max distance between pose and previous pose
            dists[:, 1:] = numpy.abs(
                sequential_poses[:, 1:] - sequential_poses[:, :-1]
            ).max(axis=2)
        else:
            raise ValueError('direction must be -1 or 1 (adjacent).')
        return dists

    # This function was originally used to set the rest change threshold (rest epsilon)
    # and its results were used in the Functional Trajectories paper.
    # I would rather use train_models_smarter(...) in the future.
    def train_models(self,
        feature: str='smoothed_complete_angles',
        labels_to_predict: list=None,
        percent_test: int=25,
        enforce_equal_representations: bool=True,
        use_cached_training_data=True,
        save: bool=True,
        do_plot: bool=True):
        ''' Trains and cross-validates

        Parameters
        ----------
        feature : str, default='smoothed_complete_angles'
            feature to use as data
        labels_to_predict : list, default=None
            Pretty much ignore this. RestPredictor only really works for rests.
        percent_test : int, default=25
            integer percent of dataset to reserve for testing.
        enforce_equal_representations : bool, default=True
            for compatibility only, ignored.
        use_cached_training_data : bool, default=True
            whether to keep existing models and train/test datasets
        save : bool, default=True
            Ignored
        do_plot : bool, default=True
            whether to show confusion matrix plot
        '''
        if labels_to_predict is None:
            labels_to_predict = list(BEHAVIORS)
        train, test = self.get_raw_training_data(percent_test, use_cached_training_data, feature, save)
        behavior_data_manager.report_train_test_summary(train, test)
        rest_label = self.numeric_rest \
                if numpy.issubdtype(train.labels.dtype, numpy.number) \
                else BEHAVIORS.rest
        def calculate_sum_error(min_angle_change):
            self.rest_epsilon = min_angle_change
            rest_mask = self.find_rests_windows(train.data)
            truth_mask = (train.labels == rest_label)
            return (rest_mask != truth_mask).sum()
        low = 0
        high = numpy.pi
        lowerbound_err = calculate_sum_error(low)
        upperbound_err = calculate_sum_error(high)
        i = 0
        max_iters = 1000
        # When the difference between high and low is less than this amount, break.
        target_precision_of_estimate = 0.00001
        progress.init(max_iters)
        while i < max_iters:
            progress.progress(i, f'Search diameter: {high - low}')
            mid = (high + low) / 2
            mid_err = calculate_sum_error(mid)
            if mid_err <= upperbound_err:
                upperbound_err = mid_err
                high = mid
            else:
                low_score = mid_err
                low = mid
            if (high - low) < target_precision_of_estimate:
                break
            i += 1
        progress.finish()
        self.rest_epsilon = high
        print(f'Best rest epsilon: {self.rest_epsilon}')
        self.evaluate(test, do_plot)
        return True

    # This function was created after preparing the Functional Trajectories manuscript,
    # but it is faster than the original "train_models(...)" and it uses a library optimizer.
    def train_models_smarter(self,
        feature: str='smoothed_complete_angles',
        labels_to_predict: list=None,
        percent_test: int=25,
        use_cached_training_data=True,
        save: bool=True,
        do_plot: bool=True,
        **kwargs):
        ''' Trains and cross-validates

        Parameters
        ----------
        feature : str, default='smoothed_complete_angles'
            feature to use as data
        labels_to_predict : list, default=None
            Pretty much ignore this. RestPredictor only really works for rests.
        percent_test : int, default=25
            integer percent of dataset to reserve for testing.
        use_cached_training_data : bool, default=True
            whether to keep existing models and train/test datasets
        save : bool, default=True
            Ignored
        do_plot : bool, default=True
            whether to show confusion matrix plot
        **kwargs : ignored
        '''
        if labels_to_predict is None:
            labels_to_predict = list(BEHAVIORS)
        train, test = self.get_raw_training_data(percent_test, use_cached_training_data, feature, save)

        behavior_data_manager.report_train_test_summary(train, test)

        rest_label = self.numeric_rest \
                if numpy.issubdtype(train.labels.dtype, numpy.number) \
                else BEHAVIORS.rest
        train_rest_mask = (train.labels == rest_label)
        train_deltas = numpy.min(
            (
                self.maximum_distance_to_an_adjacent_pose_pw(train.data, -1),
                self.maximum_distance_to_an_adjacent_pose_pw(train.data, 1)
            ),
            axis=0
        )

        def calculate_sum_error(min_angle_change, deltas, truth_mask):
            return ((deltas < min_angle_change) != truth_mask).sum()

        to_minimize = functools.partial(calculate_sum_error, deltas=train_deltas, truth_mask=train_rest_mask)
        res = optimize.minimize_scalar(to_minimize, bounds=(0, numpy.pi), method='bounded')
        print(f'Newly optimized rest epsilon: {res.x}')
        print(f'Stored rest epsilon: {self.rest_epsilon}')
        print('Stored epsilon results below:')
        self.evaluate(test, do_plot)
        print('Newly optimized epsilon results below:')
        self.rest_epsilon = res.x
        self.evaluate(test, do_plot)
        return True

    def classify_windows(self, pose_windows: numpy.ndarray):
        self.confirm_is_pose_window_list(pose_windows)
        behaviors = numpy.full(pose_windows.shape[0], BEHAVIORS.unknown)
        rest_mask = self.find_rests_windows(pose_windows)
        behaviors[numpy.where(rest_mask)] = BEHAVIORS.rest
        return behaviors

    def predict_behaviors(self, sequential_poses: numpy.ndarray):
        self.confirm_is_angle_pose_list(sequential_poses)
        behaviors = numpy.full(sequential_poses.shape[0], BEHAVIORS.unknown)
        rest_mask = self.find_rests_angles(sequential_poses)
        behaviors[numpy.where(rest_mask)] = BEHAVIORS.rest
        return behaviors

    def _sap(self, pose):
        ''' sap means Simple Angle Pose
        Meant to be a super short function name intentionally.
        Sorry the name isn't super descriptive.
        We just want to throw away the first bits of a "complete angle pose"
        and keep the angle pose itself.
        '''
        if pose.shape[0] > ANGLE_POSE_SIZE:
            return pose[pose.shape[0] - ANGLE_POSE_SIZE:]
        return pose

    def find_rests_angles(self, sequential_poses: numpy.ndarray):
        '''Predicts rests from sequential angle poses.
        Rest is when all angle changes between adjacent poses is less than epsilon.
        Rest is only classified for defined (non-NaN) distances.

        Parameters
        ----------
        sequential_poses : numpy.ndarray
            angle poses

        Returns
        -------
        numpy.ndarray
            Boolean mask of whether pose is a rest (True)
            or anything else (False).
            Shape = sequential_poses.shape[0]
        '''
        delta_prev = self.maximum_distance_to_an_adjacent_pose(sequential_poses, direction=-1)
        delta_next = self.maximum_distance_to_an_adjacent_pose(sequential_poses, direction=1)
        mask = (delta_prev < self.rest_epsilon) \
            & (delta_next < self.rest_epsilon) \
            & (~numpy.isnan(delta_prev)) \
            & (~numpy.isnan(delta_next))
        return mask.reshape(sequential_poses.shape[:-1])

    def find_rests_windows(self, PW: numpy.ndarray, verbose=False):
        '''
        Parameters
        ----------
        PW : numpy.ndarray
            pose windows, must be angle poses not "complete" angle poses
        '''
        if PW.shape[1] != (WindowsDataSet.PPW * ANGLE_POSE_SIZE):
            raise ValueError('You must pass simple angle poses only, not complete angle poses.')
        delta_prev = self.maximum_distance_to_an_adjacent_pose_pw(PW, -1)
        delta_next = self.maximum_distance_to_an_adjacent_pose_pw(PW, 1)
        mask = ((delta_prev < self.rest_epsilon) & (~numpy.isnan(delta_prev)) \
            | (delta_next < self.rest_epsilon) & (~numpy.isnan(delta_next)))
        if verbose:
            print(f'Found {mask.sum()} rests')
        return mask

if __name__ == '__main__':
    PERCENT_TEST = 25
    PREDICT_FEATURE = config.get('MACHINE_LEARNING', 'predict_feature')
    pred = RestPredictor(PREDICT_FEATURE)
    from swimfunction.data_access import PoseAccess
    from swimfunction.pose_processing.pose_filters import BASIC_FILTERS
    # pred.find_rests_angles(PoseAccess.get_feature(['M16'], [-1], 'smoothed_angles', BASIC_FILTERS, keep_shape=True).features[0])
    pred.train_models_smarter(
        PREDICT_FEATURE,
        labels_to_predict=[BEHAVIORS.rest],
        percent_test=PERCENT_TEST,
        use_cached_training_data=True,
        save=False,
        do_plot=True)
