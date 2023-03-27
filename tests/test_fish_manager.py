from swimfunction.data_access import PoseAccess
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models import Fish
from swimfunction.global_config.config import config
from swimfunction.pytest_utils import set_test_cache_access
import numpy
from swimfunction import progress
import pytest
import warnings

from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
WorkerSwarm.num_allowed_workers = 3

MIN_LIKELIHOOD_THRESHOLD = config.getfloat('POSE FILTERING', 'min_likelihood_threshold')

DEBUG = print

def __setup():
    ''' Allows fixture to be called directly if absolutely necessary.
    '''
    set_test_cache_access()

@pytest.fixture(autouse=True)
def SETUP_FILES():
    __setup()

def test_load_dlc_output():
    # Confirm no errors reading pose output files
    FDM.cache_all_annotation_files(force_reload=True)
    # Check all collected data
    DEBUG('Verifying loaded successfully...')
    fish_names = FDM.get_available_fish_names()
    progress.init(len(fish_names))
    for i, name in enumerate(fish_names):
        progress.progress(i, name)
        fish = Fish.Fish(name=name).load()
        for assay_label in fish.swim_keys():
            sa = fish[assay_label]
            likelihood_mins = sa.likelihoods.min(axis=1)
            where_best = numpy.where(likelihood_mins >= MIN_LIKELIHOOD_THRESHOLD)[0]
            mean_head_tail_dist = min((100, numpy.mean(numpy.linalg.norm(sa.raw_coordinates[where_best, 0, :] - sa.raw_coordinates[where_best, -1, :], axis=1))))
            is_extreme = lambda p: numpy.any(numpy.linalg.norm(p[:-1, :] - p[1:, :], axis=1) > mean_head_tail_dist / 2)
            num_ext_raw = sum([is_extreme(sa.raw_coordinates[i, :, :]) for i in range(sa.raw_coordinates.shape[0])])
            num_ext_smoothed = sum([is_extreme(sa.smoothed_coordinates[i, :, :]) for i in range(sa.smoothed_coordinates.shape[0])])
            # Confirm that smoothing never increased number of extreme poses
            if num_ext_raw < num_ext_smoothed:
                warnings.warn(f'Should have fewer extreme poses after smoothing! {num_ext_raw} to {num_ext_smoothed}')
            # Confirm basic pose access works correctly.
            assert numpy.all(numpy.isclose(fish[assay_label].likelihoods, numpy.concatenate(PoseAccess.get_feature([name], [assay_label], feature='likelihoods').features)))
            assert numpy.all(numpy.isclose(fish[assay_label].raw_coordinates, numpy.concatenate(PoseAccess.get_feature([name], [assay_label], feature='raw_coordinates').features)))
            assert numpy.all(numpy.isclose(fish[assay_label].smoothed_coordinates, numpy.concatenate(PoseAccess.get_feature([name], [assay_label], feature='smoothed_coordinates').features)))
    progress.finish()

def test_setup_cache():
    FDM.create_cache(True)

if __name__ == '__main__':
    __setup()
    test_setup_cache()

