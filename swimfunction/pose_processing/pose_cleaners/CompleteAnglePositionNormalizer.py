import numpy

from swimfunction.pose_processing.pose_cleaners.AbstractPoseCleaner import AbstractPoseCleaner
from swimfunction.global_config.config import config

MAX_X = config.getint('VIDEO', 'crop_width')
MAX_Y = config.getint('VIDEO', 'crop_height')

class CompleteAnglePositionNormalizer(AbstractPoseCleaner):
    ''' Normalizes position elements in complete angle poses to between 0 and 1.
    Assumes array has complete angle poses of form [head_x, head_y, head_angle, a1, a2, ..., aN]
    Returns array of form [head_x / MAX_X, head_y / MAX_Y, head_angle, a1, a2, ..., aN]
    '''
    def clean(self, arr: numpy.ndarray):
        arr[:, 0] = arr[:, 0] / MAX_X
        arr[:, 1] = arr[:, 1] / MAX_Y
        assert arr[:, :2].max() <= 1.0001
        return arr
