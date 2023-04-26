from swimfunction.global_config.config import config
from swimfunction.data_models import DataSet
from swimfunction.pose_processing.pose_conversion import ANGLE_POSE_SIZE
import numpy

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
PPW = config.getint('BEHAVIOR_ANNOTATION', 'poses_per_window')

class WindowsDataSet(DataSet.DataSet):
    ''' Transforms a series of 1d poses into temporal windows of poses.
    WindowsDataSet only works with 1d (angle or complete-angle) poses,
    not with 2d (coordinate) poses.
    '''
    __slots__ = ['margin']
    WINDOW_SIZE = int(PPW * ANGLE_POSE_SIZE)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = PPW // 2

    ''' Concatenates poses into windows of length POSES_PER_WINDOW
    Example if POSES_PER_WINDOW is 3:
        Input
            poses = [p1, p2, ... , pN]
            behaviors = [b1, b2, ... , bN]
            meta_labels = [m1, m2, ... , mN]
        Becomes
            windows = [p1p2p3, p2p3p4, ... , pN-2pN-1pN]
            labels = [b2, b3, ... , bN-1]
            meta_labels = [m2, m3, ... , mN-1]
    '''
    def from_poses(self, poses, behaviors=None, meta_labels=None):
        if poses is None:
            return self
        n_windows = poses.shape[0] - PPW + 1
        windows = []
        for i in range(n_windows):
            window = poses[i:i+PPW].reshape(PPW * poses.shape[1])
            windows.append(window)
        self.data = numpy.asarray(windows)
        if behaviors is None:
            self.labels = numpy.full(len(self.data), BEHAVIORS.unknown)
        else:
            self.labels = behaviors[self.margin:self.margin+len(self.data)]
        if meta_labels is not None:
            self.meta_labels = self.meta_labels[self.margin:self.margin+len(self.data)]
        return self

    def from_concatenated_poses(self, concatenated_poses, labels=None, meta_labels=None):
        self.data = numpy.asarray(concatenated_poses)
        if labels is None:
            self.labels = numpy.full(len(self.data), BEHAVIORS.unknown)
        else:
            self.labels = labels
        self.meta_labels = meta_labels
        return self

    def get_middle_poses(self):
        ''' Returns the middle pose of the window.
        '''
        pose_length = self.data.shape[-1] // PPW
        middle_poses = self.data[
            :, # all windows
            pose_length*((PPW + 1)//2):pose_length*((PPW + 1)//2)+pose_length
        ]
        return middle_poses

    def to_pose_dataset(self):
        d = self.as_dict()
        d['data'] = self.get_middle_poses()
        return DataSet.DataSet.from_dict(d)

    def reshape_to_WNA(self):
        ''' Reshapes data from (W, N*A) to (W, N, A)
        Where W is number of windows, N is number of poses in a window, A is number of angles in a pose.

        Returns
        -------
        self
            For convenience.
        '''
        if len(self.data.shape) == 3:
            return self
        self.data = self.data.reshape((self.data.shape[0], PPW, self.data.shape[1] // PPW))
        return self

    @staticmethod
    def from_dict(ds_dict):
        ''' Creates a WindowsDataSet from a dict.
        '''
        return WindowsDataSet(
            data=ds_dict['data'] if 'data' in ds_dict else None,
            labels=ds_dict['labels'] if 'labels' in ds_dict else None,
            meta_labels=ds_dict['meta_labels'] if 'meta_labels' in ds_dict else None
        )

    @staticmethod
    def cast(ds):
        return WindowsDataSet(ds.data, ds.labels, ds.meta_labels)

    @staticmethod
    def to_poses(window):
        return window.reshape((PPW, -1))
