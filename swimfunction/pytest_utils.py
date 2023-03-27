from swimfunction.global_config.config import config, CacheAccessParams

from swimfunction import FileLocations
import numpy

TEST_FILES_ROOT = FileLocations.get_test_root()
CACHE_DUMP = TEST_FILES_ROOT / 'tmp'
CACHE_DUMP.mkdir(exist_ok=True)
TMP_FILE = CACHE_DUMP / 'tmp.pickle'
TMP_NUMPY_FILE = CACHE_DUMP / 'tmp.npy'
DOES_NOT_EXIST = CACHE_DUMP / 'DOES_NOT_EXIST.pickle'

def assert_equal_lists(a1, a2, msg=''):
    for x1, x2 in zip(a1, a2):
        if isinstance(x1, (list, numpy.ndarray)):
            assert_equal_lists(x1, x2)
        else:
            assert x1 == x2, msg

def set_test_cache_access():
    config.set_access_params(CacheAccessParams.get_test_access())

