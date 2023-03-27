from swimfunction.pose_processing.pose_cleaners.TeleportationCleaner import TeleportationCleaner
from swimfunction.pytest_utils import assert_equal_lists, set_test_cache_access
import numpy
import pytest


def __setup():
    ''' Allows fixture to be called directly if absolutely necessary.
    '''
    set_test_cache_access()

@pytest.fixture(autouse=True)
def SETUP_FILES():
    __setup()

def test_pose_cleaning_simple():
    cleaner = TeleportationCleaner(5)
    xx_pairs = [
        ([1, 2, 3, 4, 5, 54, 56, 57, 58, 59],
        [1, 2, 3, 4, 5, 54, 56, 57, 58, 59]),
        ([1, 2, 3, 4, 5, 54, 56, 57, 90],
        [1, 2, 3, 4, 5, 5, 5, 5, 5]),
        ([1, 2, 3, 4, 15, 6, 7, 8, 9],
        [1, 2, 3, 4, 4, 6, 7, 8, 9])
    ]
    for x, expected_x in xx_pairs:
        x = numpy.asarray(x)
        expected_x = numpy.asarray(expected_x)
        cleaned_x, _was_cleaned_mask = cleaner.clean(x)
        assert_equal_lists(expected_x, cleaned_x, f'{x} != {expected_x}')

if __name__ == '__main__':
    __setup()
    test_pose_cleaning_simple()
