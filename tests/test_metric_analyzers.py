import pytest
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import BehaviorAnnotationDataManager as BDM
from swimfunction.pytest_utils import set_test_cache_access
from swimfunction.recovery_metrics.metric_analyzers\
    .PostureNoveltyAnalyzer import PostureNoveltyAnalyzer
from swimfunction.recovery_metrics.metric_analyzers\
    .CruiseWaveformAnalyzer import CruiseWaveformAnalyzer
from swimfunction.recovery_metrics.metric_analyzers\
    .ScoliosisAnalyzer import ScoliosisAnalyzer
from swimfunction.recovery_metrics.metric_analyzers\
    .StrouhalAnalyzer import StrouhalAnalyzer
from swimfunction.recovery_metrics.metric_analyzers\
    .SwimCapacityAnalyzer import SwimCapacityAnalyzer
from swimfunction.recovery_metrics.metric_analyzers\
    .PrecalculatedMetricsAnalyzer import PrecalculatedMetricsAnalyzer
from swimfunction.data_models.Fish import Fish


def __setup():
    ''' Allows fixture to be called directly if absolutely necessary.
    '''
    set_test_cache_access()
    _set_predicted_behaviors_as_annotated()
    PostureNoveltyAnalyzer(null_assay=1).load_model_and_counts().save_model_and_counts()

def _set_predicted_behaviors_as_annotated():
    for name in FDM.get_available_fish_names():
        fish = Fish(name).load()
        for assay in fish.swim_keys():
            fish[assay].predicted_behaviors = BDM.load_behaviors(name, assay).behaviors
        fish.save()

@pytest.fixture(autouse=True)
def SETUP_FILES():
    __setup()

def test_can_analyze_without_crashing():
    assay = Fish('F1').load()[1]
    print(PostureNoveltyAnalyzer().analyze_assay(assay))
    print(CruiseWaveformAnalyzer().set_control(assay=1).analyze_assay(assay))
    print(ScoliosisAnalyzer().analyze_assay(assay))
    print(StrouhalAnalyzer().analyze_assay(assay))
    print(SwimCapacityAnalyzer().analyze_assay(assay))
    print(PrecalculatedMetricsAnalyzer().analyze_assay(assay))

if __name__ == '__main__':
    __setup()
    test_can_analyze_without_crashing()
