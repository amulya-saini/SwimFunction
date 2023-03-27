'''
After you have processed the videos and used DeepLabCut to get posture annotations,
this script will walk through the basic post-processing and analysis steps for your experiment.
'''
import functools
import traceback
from collections import namedtuple
from tqdm import tqdm
import numpy

from swimfunction.behavior_annotation.UmapClassifier import UmapClassifier
from swimfunction.behavior_annotation.RestPredictor import RestPredictor
from swimfunction.data_access import PoseAccess
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.Fish import Fish

from swimfunction.data_access.data_utils import AnnotationTypes
from swimfunction.pose_processing import pose_filters
from swimfunction.data_access.MetricLogger import MetricLogger
from swimfunction.recovery_metrics.metric_analyzers.CruiseWaveformCalculator \
    import CruiseWaveformCalculator
from swimfunction.recovery_metrics.metric_analyzers.MetricCalculator import MetricCalculator
from swimfunction.global_config.config import config
from swimfunction.recovery_metrics.metric_analyzers.ScoliosisAnalyzer import ScoliosisAnalyzer
from swimfunction.recovery_metrics.metric_analyzers.PostureNoveltyAnalyzer \
    import PostureNoveltyAnalyzer
from swimfunction.recovery_metrics.metric_analyzers.CruiseWaveformAnalyzer \
    import CruiseWaveformAnalyzer
from swimfunction.recovery_metrics.metric_analyzers.PrecalculatedMetricsAnalyzer \
    import PrecalculatedMetricsAnalyzer
from swimfunction.recovery_metrics.metric_analyzers.SwimCapacityAnalyzer import SwimCapacityAnalyzer
from swimfunction.recovery_metrics.metric_analyzers.StrouhalAnalyzer import StrouhalAnalyzer
from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
from swimfunction import FileLocations
from swimfunction import loggers
from swimfunction import progress
from swimfunction.trajectories.RecoveryPredictor import RecoveryPredictor
from swimfunction.main_scripts.optional_training_models.train_predictors\
    import train_umap_classifier_predictor

# Replace existing training data and models
USE_CACHED_TRAINING_DATA = True
SAVE_MODELS = True

ENFORCE_EQUAL_LABEL_REPRESENTATION = True
BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
PREDICT_FEATURE = config.get('MACHINE_LEARNING', 'predict_feature')

EXPERIMENT_NAME_TO_WAS_TRACKED = config.getbooldict(
    'EXPERIMENT DETAILS', 'names',
    'EXPERIMENT DETAILS', 'individuals_were_tracked')

PREDICTORS = namedtuple('predictors', ['rest_predictor', 'cruise_predictor'])

PREINJURY_ASSAY = config.getint('EXPERIMENT DETAILS', 'uninjured_assay_label')

def get_logger() -> loggers.logging.Logger:
    ''' Get the default metric calculation logger.
    '''
    return loggers.get_metric_calculation_logger(__name__)

def header(msg):
    ''' Log the message loud and clear.
    '''
    get_logger().info('\n------- %s -------\n', msg)

def fish_identities_were_tracked_in_the_experiment() -> bool:
    ''' Whether the fish identities carry over from assay to assay.
    For example, did you house the fish separately and keep track
    of which is which for the entire expriment? If so, put it in the config.ini file.
    '''
    return config.experiment_name in EXPERIMENT_NAME_TO_WAS_TRACKED \
        and EXPERIMENT_NAME_TO_WAS_TRACKED[config.experiment_name]

def analyzer_to_metrics(analyzer):
    ''' Get all expected metrics from the analyzer
    '''
    return list(analyzer.keys_to_printable.keys())

def get_metrics_list(opts, list_metrics_fn: analyzer_to_metrics) -> list:
    ''' Operates a given function (which must return a list)
    upon all relevant analyzers and returns the concatenated result.
    '''
    to_reduce = [
        list_metrics_fn(AnalyzerClass())
        for attr, AnalyzerClass in ANALYZER_CLASSES.items()
        if getattr(opts, attr)
    ]
    if not to_reduce:
        return []
    return functools.reduce(lambda a, b: a + b, to_reduce)

ANALYZER_CLASSES = {
        'capacity_metrics': SwimCapacityAnalyzer(),
        'scoliosis': ScoliosisAnalyzer(),
        'posture_novelty': PostureNoveltyAnalyzer(),
        'waveforms': CruiseWaveformAnalyzer(),
        'strouhal': StrouhalAnalyzer(),
        'precalculated': PrecalculatedMetricsAnalyzer()
}

SWARMABLE_CLASSES = {
        SwimCapacityAnalyzer,
        CruiseWaveformAnalyzer,
        StrouhalAnalyzer,
        PrecalculatedMetricsAnalyzer
}

def get_behavior_predictors() -> PREDICTORS:
    ''' Get behavior predictors,
    and if the cruise predictor has not yet been trained, train it.
    '''
    cruise_predictor = UmapClassifier()
    rest_predictor = RestPredictor(PREDICT_FEATURE)
    cruise_predictor.load_models()
    if cruise_predictor.umap_mapper is None:
        get_logger().info(' '.join([
            'Cruise predictor could not load.',
            'Will attempt to train a new model instead.'
        ]))
        train_umap_classifier_predictor()
        cruise_predictor = UmapClassifier()
        cruise_predictor.load_models()
    return PREDICTORS(
        rest_predictor,
        cruise_predictor
    )

def populate_data_storage_structures():
    '''
        Loads dlc centerline points into Fish objects
        with all other pose calculations (angle poses, etc.)
    '''
    header('Creating cache structure from pose annotation files')
    FDM.create_cache()
    FDM.create_available_assays_csv(force_recalculate=True)
    header('Finished basic cache structure.')

def predict_behaviors_as_required(force_recalculate: bool=False):
    ''' Only predicts behaviors if swim_assay.predicted_behaviors has non-unknown items.

    NOTE: this cannot be in a WorkerSwarm because the internal functions are incompatible.
    '''
    predictors = None
    header('Predicting behaviors if not yet predicted')
    for fish_name in tqdm(FDM.get_available_fish_names()):
        fish = Fish(name=fish_name).load()
        for assay in tqdm(fish.swim_keys(), leave=False):
            existing_behaviors = fish[assay].predicted_behaviors
            if not force_recalculate \
                    and isinstance(existing_behaviors, (list, numpy.ndarray)) \
                    and len(existing_behaviors) \
                    and not numpy.all(numpy.asarray(existing_behaviors) == BEHAVIORS.unknown):
                continue
            sequential_poses = PoseAccess.get_feature_from_assay(
                fish[assay], PREDICT_FEATURE, pose_filters.BASIC_FILTERS, True)
            if predictors is None:
                predictors = get_behavior_predictors()
            rest_mask = predictors.rest_predictor.find_rests_angles(sequential_poses)
            behaviors = predictors.cruise_predictor.predict_behaviors(sequential_poses)
            behaviors[rest_mask] = BEHAVIORS.rest
            fish[assay].predicted_behaviors = behaviors
            fish.save()

def calculate_waveform_stats_as_required():
    '''
    Precalculate waveform stats for all assays.
    '''
    header('Calculating waveform stats as required')
    waveform_calculator = CruiseWaveformCalculator(
        config.experiment_name,
        AnnotationTypes.predicted
    )
    for assay in FDM.get_available_assay_labels():
        waveform_calculator.get_waveform_stats(None, assay, None, None)

def calculate_metrics_as_required(control_assay_for_rostral_compensation):
    ''' Calculate all metrics that have not yet been calculated.
    '''
    header('Calculating metrics as required')

    df = MetricLogger.metric_dataframe

    def task(fish_name, analyzer_list):
        try:
            MetricCalculator.get_metrics_for_analyzers(
                fish_name,
                analyzer_list,
                recalc_mask=numpy.full(len(analyzer_list), False, dtype=bool),
                df=df,
                progress_fn=progress.increment)
            MetricLogger.save_metric_dataframe()
        except Exception as e:
            get_logger().error(fish_name)
            get_logger().error(e)
            get_logger().error(traceback.format_exc())

    ANALYZER_CLASSES['waveforms'].set_control(assay=control_assay_for_rostral_compensation)

    swarmable_analyzers_dict = {
        a: v for a, v in ANALYZER_CLASSES.items()\
            if v.__class__ in SWARMABLE_CLASSES}
    nonswarmable_a_d = {
        a: v for a, v in ANALYZER_CLASSES.items()\
            if v.__class__ not in SWARMABLE_CLASSES}

    total = sum([
        len(ANALYZER_CLASSES) * len(FDM.get_available_assay_labels(n)) \
        for n in FDM.get_available_fish_names()])

    with progress.Progress(total):
        with WorkerSwarm(get_logger()) as swarm:
            analyzers = [
                swarmable_analyzers_dict[k]
                for k in sorted(swarmable_analyzers_dict.keys())]
            for fish_name in FDM.get_available_fish_names():
                swarm.add_task(lambda n=fish_name, a=analyzers: task(n, a))

    analyzers = [nonswarmable_a_d[k] for k in sorted(nonswarmable_a_d.keys())]
    for fish_name in FDM.get_available_fish_names():
        task(fish_name, analyzers)

def predict_outcomes():
    ''' Predict outcomes and save in the csv output folder.
    '''
    res_df = RecoveryPredictor.predict_as_published().sort_index()
    res_df.to_csv(FileLocations.get_outcome_prediction_csv(), mode='w')
    with open(FileLocations.get_outcome_prediction_final_name_key_csv(), 'wt') as fh:
        fh.write('fish,predicted_well\n'
            'example_fish_name_at_final_assay,True'
            'YOU MUST MAKE THIS FILE YOURSELF ACCORDING TO INSTRUCTIONS IN THE README FILE\n')

def metrics_main(control_assay_for_rostral_compensation=PREINJURY_ASSAY):
    ''' Calculate metrics
    '''
    get_logger().info(
        'Find csv outputs in %s',
        FileLocations.get_csv_output_dir().as_posix())
    # Import all data into a useable structure.
    populate_data_storage_structures()
    predict_behaviors_as_required()
    calculate_waveform_stats_as_required()
    # Calculate all metrics
    calculate_metrics_as_required(control_assay_for_rostral_compensation)
    if fish_identities_were_tracked_in_the_experiment():
        predict_outcomes()

if __name__ == '__main__':
    ARGS = FileLocations.parse_default_args()
    try:
        metrics_main()
    except Exception as _E:
        get_logger().error('Exception occurred in %s', __name__)
        get_logger().error(_E)
        get_logger().error(traceback.format_exc())
