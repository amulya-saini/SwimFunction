''' Test basic inputs,
make sure it can calculate metrics without crashing.
'''
import time
import pytest
from matplotlib import pyplot as plt

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import BehaviorAnnotationDataManager as BDM
from swimfunction.data_models.Fish import Fish
from swimfunction.pytest_utils import set_test_cache_access
from swimfunction.main_scripts import metrics, plotting, qc
from swimfunction.recovery_metrics.metric_analyzers\
    .PostureNoveltyAnalyzer import PostureNoveltyAnalyzer
from swimfunction.global_config.config import config, CacheAccessParams
from swimfunction import FileLocations

plt.switch_backend('agg') # No gui will be created.

TEST_NULL_ASSAY = 1

def __setup():
    ''' Allows fixture to be called directly if absolutely necessary.
    '''
    set_test_cache_access()
    _set_predicted_behaviors_as_annotated()
    # Necessary because
    PostureNoveltyAnalyzer(null_assay=TEST_NULL_ASSAY)\
        .load_model_and_counts().save_model_and_counts()

@pytest.fixture(autouse=True)
def SETUP_FILES():
    ''' Set test cache params
    '''
    __setup()

def _set_predicted_behaviors_as_annotated():
    ''' Forces behaviors to be as annotated.
    '''
    for name in FDM.get_available_fish_names():
        fish = Fish(name).load()
        for assay in fish.swim_keys():
            fish[assay].predicted_behaviors = BDM.load_behaviors(name, assay).behaviors
        fish.save()

def test_metrics_from_annotations_only():
    ''' Test get metrics from simple inputs: annotations only.
    Simple run-and-don't-crash test
    '''
    config.set('TEST', 'test', 'false') # Allow folder creation for this function
    config.set_access_params(CacheAccessParams.get_test_access())
    # Copy annotation files into temporary folder
    annotation_fpaths = dlc_outputs_dir = FileLocations.get_dlc_outputs_dir().glob('*')
    experiment_name = f'from_annotations_only_{int(time.time())}'
    config.set_access_params(CacheAccessParams('/tmp', experiment_name))
    dlc_outputs_dir = FileLocations.get_dlc_outputs_dir()
    for fp in annotation_fpaths:
        (dlc_outputs_dir / fp.name).symlink_to(fp)
    qc.qc_main()
    metrics.metrics_main(control_assay_for_rostral_compensation=TEST_NULL_ASSAY)
    plotting.plotting_main()
    config.set('TEST', 'test', 'true')

def test_qc_and_metrics():
    ''' Test get metrics from simple inputs: videos and annotations.
    Simple run-and-don't-crash test
    '''
    config.set('TEST', 'test', 'false') # Allow folder creation for this function
    qc.qc_main()
    metrics.metrics_main(control_assay_for_rostral_compensation=TEST_NULL_ASSAY)
    plotting.plotting_main()
    config.set('TEST', 'test', 'true')

def test_rostral_compensation_only():
    config.set('TEST', 'test', 'false') # Allow folder creation for this function
    qc.qc_main()
    metrics.rostral_compensation_only(
        control_assay_for_rostral_compensation=TEST_NULL_ASSAY)
    config.set('TEST', 'test', 'true')
