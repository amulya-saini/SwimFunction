from swimfunction.data_access import PoseAccess
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.Fish import Fish
from swimfunction.pytest_utils import assert_equal_lists, set_test_cache_access
from swimfunction.global_config.config import config

import numpy
import pytest

def __setup():
    ''' Allows fixture to be called directly if absolutely necessary.
    '''
    set_test_cache_access()

@pytest.fixture(autouse=True)
def SETUP_FILES():
    __setup()

def test_split_into_consecutive_indices():
    ''' Checks whether a series of indices
    can be split into their consecutive blocks
    successfully.
    '''
    pairs = [
        (
            [0, 1, 2, 3, 4, 5, 6],
            [[0, 1, 2, 3, 4, 5, 6]]
        ),
        (
            [0, 1, 2, 4, 5, 6],
            [[0, 1, 2], [4, 5, 6]]
        ),
        (
            [0, 1, 4, 6, 7],
            [[0, 1], [4], [6, 7]]
        ),
        (
            [3, 4, 11, 20, 21],
            [[3, 4], [11], [20, 21]]
        ),
        (
            [145, 743, 2622, 64543],
            [[145], [743], [2622], [64543]]
        ),
        (
            [0, 2, 3, 4, 6],
            [[0], [2, 3, 4], [6]]
        ),
        (
            [0, 1, 3, 4, 6, 7],
            [[0, 1], [3, 4], [6, 7]]
        ),
        (
            [34, 35, 36, 123, 124, 125, 126, 127, 345, 657, 678, 679, 876, 899],
            [[34, 35, 36], [123, 124, 125, 126, 127], [345], [657], [678, 679], [876], [899]]
        )
    ]
    for arg, expected in pairs:
        for val, ex in zip(list(PoseAccess.split_into_consecutive_indices(arg)), expected):
            assert_equal_lists(val, ex)

def test_episode_indices_to_feature():
    feature='smoothed_angles'
    swim = Fish('F1').load()[1]
    poses = PoseAccess.get_feature_from_assay(swim, feature, filters=[], keep_shape=True).tolist()
    episode_indices = [[6,7,8], [34,35,36,37,38], [1,2,3], [56,57,58,59]]
    expected = [[poses[i] for i in l] for l in episode_indices]
    result = PoseAccess.episode_indices_to_feature(swim, episode_indices, feature)
    assert_equal_lists(expected, result)

if __name__ == '__main__':
    test_episode_indices_to_feature()
