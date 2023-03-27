from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import data_utils, PoseAccess
from swimfunction.data_models import Fish
from swimfunction.context_managers import CacheContext
from swimfunction.pose_processing import pose_filters
from swimfunction.global_config.config import config
from swimfunction.pytest_utils import assert_equal_lists, set_test_cache_access, TMP_FILE, TMP_NUMPY_FILE, DOES_NOT_EXIST
from swimfunction import FileLocations

import numpy
import pytest
import traceback

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')
POINTS_PER_POSE = config.getint('POSES', 'points_per_pose')

@pytest.fixture(autouse=True)
def SETUP_FILES():
    set_test_cache_access()
    if TMP_FILE.exists():
        TMP_FILE.unlink()
    if TMP_NUMPY_FILE.exists():
        TMP_NUMPY_FILE.unlink()

def get_valid_assays(fish_name):
    return Fish.Fish(name=fish_name).load().swim_keys()

def test_does_not_exist():
    try:
        if DOES_NOT_EXIST.exists():
            DOES_NOT_EXIST.unlink()
        CacheContext.read_cache(DOES_NOT_EXIST)
    except Exception as e:
        print(traceback.format_exc())
        raise AssertionError(f'Must not throw error when reading non-existant cache: {e}')

def test_save_load():
    stuff = { 'this': 42 }
    CacheContext.dump_cache(stuff, TMP_FILE)
    result = CacheContext.read_cache(TMP_FILE)
    assert stuff['this'] == result['this']

def test_save_load_context():
    stuff = { 'this': 42 }
    with CacheContext.CacheContext(TMP_FILE) as cache:
        cache.saveContents(stuff)
    with CacheContext.CacheContext(TMP_FILE) as cache:
        assert stuff['this'] == cache.getContents()['this']

def test_save_load_numpy():
    stuff = numpy.arange(12)
    CacheContext.dump_numpy_cache(stuff, TMP_NUMPY_FILE)
    result = CacheContext.read_numpy_cache(TMP_NUMPY_FILE)
    assert_equal_lists(stuff, result)

def test_save_load_numpy_movements():
    stuff = [BEHAVIORS.burst]
    CacheContext.dump_numpy_cache(stuff, TMP_NUMPY_FILE)
    result = CacheContext.read_numpy_cache(TMP_NUMPY_FILE)
    assert_equal_lists(stuff, result)

def test_find_videos():
    set_test_cache_access()
    should_contain = set(
        (fish, assay_label)
            for fish in FDM.get_available_fish_names()
                for assay_label in FDM.get_available_assay_labels(fish)
    )
    for name, assay_label in should_contain:
        assert FileLocations.find_video(name, assay_label) is not None, \
            f'{name} assay {assay_label} does not exist. \
                Video must exist for every living timepoint.'

def test_get_all_poses():
    expected_count = sum([len(get_valid_assays(name)) for name in FDM.get_available_fish_names()])
    pose_lists = PoseAccess.get_feature(feature='smoothed_complete_angles').features
    assert len(pose_lists) == expected_count
    for p in pose_lists:
        assert len(p.shape) == 2
        # Complete angle poses have POINTS_PER_POSE + 1 elements
        elems = POINTS_PER_POSE + 1
        assert p.shape[0] > 0 and p.shape[1] == elems

def test_get_all_poses_filtered():
    expected_count = sum([len(get_valid_assays(name)) for name in FDM.get_available_fish_names()])
    pose_lists = PoseAccess.get_feature(
        feature='smoothed_complete_angles',
        filters=pose_filters.BASIC_FILTERS).features
    assert len(pose_lists) == expected_count
    for p in pose_lists:
        assert len(p.shape) == 2
        # Complete angle poses have POINTS_PER_POSE + 1 elements
        elems = POINTS_PER_POSE + 1
        assert p.shape[0] > 0 and p.shape[1] == elems

def test_simple_filename_parsing():
    names = ['M1', 'F14', 'M785', 'F0']
    assays = [-1, 3, 5, 6, 23, 64326]
    for name in names:
        for assay in assays:
            fnames = [
                f'{assay}wpi_{name}.avi',
                f'{assay}wpi_{name}L.avi',
                f'{assay}wpi_{name}R.avi',
                f'/PATH/TO/VIDEOS/HERE/1-23-2020_17491_EKAB_{assay}wpi_{name}RDLC_resnet50_tracked_fishDec16shuffle1_180000.csv',
                f'/PATH/TO/VIDEOS/HERE/1-23-2020_17491_EKAB_{assay}wpi_{name}LDLC_resnet50_tracked_fishDec16shuffle1_180000.csv'
            ]
            for f in fnames:
                details = data_utils.parse_details_from_filename(f)
                assert len(details) == 1
                assert details[0].name == name
                assert details[0].assay_label == assay
    data_utils.parse_details_from_filename('-1wpi_M1R.avi')[0].assay_label == -1
    data_utils.parse_details_from_filename('Pre-inj_M1R.avi')[0].assay_label == -1
    data_utils.parse_details_from_filename('Preinj_M1R.avi')[0].assay_label == -1
    data_utils.parse_details_from_filename('pre-inj_M1R.avi')[0].assay_label == -1
    data_utils.parse_details_from_filename('preinj_M1R.avi')[0].assay_label == -1

def test_complex_filename_parsing():
    def check(assay_label, n1, side1, n2, side2):
        details = data_utils.parse_details_from_filename(
            f'/PATH/TO/VIDEOS/HERE/1-7-2020_17491_EKAB_{assay_label}wpi_{n1}{side1}_{n2}{side2}.avi')
        assert len(details) == 2
        found1 = False
        found2 = False
        for d in details:
            assert d.assay_label == assay_label
            if d.name == n1:
                assert d.side == side1
                found1 = True
            if d.name == n2:
                assert d.side == side2
                found2 = True
        assert found1 and found2
    check(n1 = 'M15',
        side1 = 'R',
        n2 = 'F13',
        side2 = 'L',
        assay_label = 2)
    check(n1 = 'M15',
        side1 = 'L',
        n2 = 'F13',
        side2 = 'R',
        assay_label = 1)

if __name__ == '__main__':
    test_complex_filename_parsing()
