import numpy
import pytest
from matplotlib import pyplot as plt

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import BehaviorAnnotationDataManager as behavior_data_manager
from swimfunction.global_config.config import config
from swimfunction.behavior_annotation.UmapClassifier import UmapClassifier
from swimfunction.data_models import Fish
from swimfunction.pytest_utils import assert_equal_lists, set_test_cache_access, TMP_FILE
from swimfunction import FileLocations

plt.switch_backend('agg') # No gui will be created.

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')

TMP_FISH = 'B52'
TMP_WPI = 20

def __setup():
    ''' Allows fixture to be called directly if absolutely necessary.
    '''
    set_test_cache_access()
    tmp_annotation_path = FileLocations.get_behavior_annotations_path(TMP_FISH, TMP_WPI)
    if tmp_annotation_path.exists():
        tmp_annotation_path.unlink()
    if TMP_FILE.exists():
        TMP_FILE.unlink()

@pytest.fixture(autouse=True)
def SETUP_FILES():
    __setup()

def test_load_empty_labels():
    nframes = 10
    result, _annotator_names = behavior_data_manager.load_behaviors('AHGSDH', 23, nframes=nframes)
    result = numpy.asarray(result)
    assert len(result) == nframes
    for x in result:
        assert x == BEHAVIORS.unknown

def test_load_empty_movements():
    nframes = 10
    result, _annotator_names = behavior_data_manager.load_behaviors('GOIEWG', 58, nframes=nframes)
    result = numpy.asarray(result)
    assert len(result) == nframes
    for x in result:
        assert x == BEHAVIORS.unknown

def test_save_load_simple():
    data = [BEHAVIORS.burst]
    ann_arr = ['TEST' for _ in data]
    behavior_data_manager.save_behaviors(TMP_FISH, TMP_WPI, data, ann_arr)
    result, annotator_names = behavior_data_manager.load_behaviors(TMP_FISH, TMP_WPI, nframes=len(data))
    assert_equal_lists(result, data, f'Movements are incorrect: {result} != {data}')
    assert_equal_lists(ann_arr, annotator_names, f'Annotator names were not saved correctly.')

def test_save_load():
    data = [
        BEHAVIORS.burst,
        BEHAVIORS.burst,
        BEHAVIORS.burst,
        BEHAVIORS.turn_ccw,
        BEHAVIORS.turn_ccw,
        BEHAVIORS.turn_cw,
        BEHAVIORS.turn_cw,
        BEHAVIORS.rest,
        BEHAVIORS.rest ]
    ann_arr = ['TEST' for _ in data]
    behavior_data_manager.save_behaviors(TMP_FISH, TMP_WPI, data, ann_arr)
    result, annotator_names = behavior_data_manager.load_behaviors(TMP_FISH, TMP_WPI, nframes=len(data))
    assert_equal_lists(result, data)
    assert_equal_lists(ann_arr, annotator_names, f'Annotator names were not saved correctly.')

def test_save_load_with_unknown():
    data = [
        BEHAVIORS.burst,
        BEHAVIORS.burst,
        BEHAVIORS.burst,
        BEHAVIORS.unknown,
        BEHAVIORS.turn_ccw,
        BEHAVIORS.unknown,
        BEHAVIORS.turn_cw,
        BEHAVIORS.rest,
        BEHAVIORS.rest ]
    ann_arr = [
        'TEST',
        'TEST',
        'TEST',
        None,
        'TEST',
        None,
        'TEST',
        'TEST',
        'TEST'
    ]
    behavior_data_manager.save_behaviors(TMP_FISH, TMP_WPI, data, ann_arr)
    result, annotator_names = behavior_data_manager.load_behaviors(TMP_FISH, TMP_WPI, nframes=len(data))
    assert_equal_lists(result, data)
    assert_equal_lists(ann_arr, annotator_names, f'Annotator names were not saved correctly.')

def test_behavior_training():
    predictor = UmapClassifier()
    predictor.train_models(
        percent_test=25,
        enforce_equal_representations=False,
        use_cached_training_data=False,
        save=True,
        do_plot=(__name__=='__main__'))

def test_behavior_prediction():
    poses_to_predict = 30
    predictor = UmapClassifier()
    predictor.train_models(
        percent_test=25,
        enforce_equal_representations=False,
        use_cached_training_data=True,
        save=False,
        do_plot=(__name__=='__main__'))
    for fish_name in FDM.get_available_fish_names():
        print(f'Loading fish {fish_name}')
        fish = Fish.Fish(name=fish_name).load()
        for wpi in fish.swim_keys():
            sequential_poses = fish[wpi].smoothed_angles
            print(f'Predicting behaviors from first and last {poses_to_predict} poses')
            print(predictor.predict_behaviors(sequential_poses[:poses_to_predict]))
            print(predictor.predict_behaviors(sequential_poses[-poses_to_predict:]))

if __name__=='__main__':
    __setup()
    # test_save_load_with_unknown()
    test_behavior_training()
    # test_behavior_prediction()
