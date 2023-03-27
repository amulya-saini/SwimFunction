''' Logs calculated metric values for all fish in the experiment.

All analyzers must extend AbstractMetricAnalyzer
which have a function called `.analyze_assay(SwimAssay)`
which return dicts {metric_name: value}
because some analyzers test multiple metrics.
'''

import threading
import traceback
from typing import List
import pandas

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access.MetricLogger import MetricLogger
from swimfunction.data_models.Fish import Fish
from swimfunction.context_managers.AccessContext import AccessContext
from swimfunction.recovery_metrics.metric_analyzers.AbstractMetricAnalyzer import AbstractMetricAnalyzer
from swimfunction.global_config.config import config
from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
from swimfunction.recovery_metrics.metric_analyzers.CruiseWaveformAnalyzer import CruiseWaveformAnalyzer
from swimfunction.recovery_metrics.metric_analyzers.StrouhalAnalyzer import StrouhalAnalyzer

from swimfunction import loggers
from swimfunction import progress

def get_logger() -> loggers.logging.Logger:
    ''' Get the default metric calculation logger.
    '''
    return loggers.get_metric_calculation_logger(__name__)

class MetricCalculator:

    CAPACITY_METRICS_TO_ANALYZER = {}
    POSE_METRICS_TO_ANALYZER = {}
    GAIT_METRICS_TO_ANALYZER = {}
    STRUCTURAL_METRICS_TO_ANALYZER = {}
    METRICS_TO_ANALYZER = {}
    METRICS_TO_PRINTABLE = {}
    ACCESS_LOCK = threading.Lock()

    @staticmethod
    def _late_assignment():
        ''' It takes a long time to load some of these,
        so we're only loading them if we absolutely must.
        '''
        if MetricCalculator.METRICS_TO_PRINTABLE:
            return
        from swimfunction.recovery_metrics.metric_analyzers.ScoliosisAnalyzer import ScoliosisAnalyzer
        from swimfunction.recovery_metrics.metric_analyzers.PostureNoveltyAnalyzer import PostureNoveltyAnalyzer
        from swimfunction.recovery_metrics.metric_analyzers.SwimCapacityAnalyzer import SwimCapacityAnalyzer
        MetricCalculator.CAPACITY_METRICS_TO_ANALYZER = {
            **SwimCapacityAnalyzer().keys_to_self()
        }

        MetricCalculator.POSE_METRICS_TO_ANALYZER = {
            **ScoliosisAnalyzer().keys_to_self(),
            **PostureNoveltyAnalyzer().keys_to_self()
        }

        MetricCalculator.GAIT_METRICS_TO_ANALYZER = {
            **CruiseWaveformAnalyzer().keys_to_self(),
            **StrouhalAnalyzer().keys_to_self()
        }

        MetricCalculator.METRICS_TO_ANALYZER = {
            **MetricCalculator.STRUCTURAL_METRICS_TO_ANALYZER,
            **MetricCalculator.CAPACITY_METRICS_TO_ANALYZER,
            **MetricCalculator.POSE_METRICS_TO_ANALYZER,
            **MetricCalculator.GAIT_METRICS_TO_ANALYZER
        }

        MetricCalculator.METRICS_TO_PRINTABLE = {
            metric: analyzer.keys_to_printable[metric]
            for metric, analyzer in MetricCalculator.METRICS_TO_ANALYZER.items()
        }

    @staticmethod
    def get_metrics_to_analyzer_dict() -> dict:
        MetricCalculator._late_assignment()
        return MetricCalculator.METRICS_TO_ANALYZER

    @staticmethod
    def get_metrics_to_printable_dict() -> dict:
        MetricCalculator._late_assignment()
        return MetricCalculator.METRICS_TO_PRINTABLE

    @staticmethod
    def assay_has_been_analyzed(df, fish_name: str, assay: int, analyzer: AbstractMetricAnalyzer):
        ''' If the fish is in df and the fish has any measurment from this analyzer,
        it is assumed to already have been analyzed.
        '''
        was_analyzed = False
        with MetricCalculator.ACCESS_LOCK:
            try:
                if (fish_name, assay) in df.index:
                    fields = [f for f in analyzer.keys_to_printable.keys() if f in df.columns]
                    was_analyzed = not pandas.isna(df.loc[(fish_name, assay), fields]).all()
            except Exception as e:
                get_logger().error('%s %d, %s', fish_name, assay, analyzer.__class__)
                get_logger().error(e)
                get_logger().error(traceback.format_exc())
        return was_analyzed

    @staticmethod
    def get_metrics_for_analyzers(
            fish_name: str,
            analyzers: List[AbstractMetricAnalyzer],
            recalc_mask: List[bool],
            df: pandas.DataFrame,
            progress_fn):
        fish = Fish(fish_name).load()
        for assay in fish.swim_keys():
            for analyzer, force_recalculate in zip(analyzers, recalc_mask):
                progress_fn(f'{fish_name} {assay} {str(analyzer.__class__)}')
                if not force_recalculate and MetricCalculator.assay_has_been_analyzed(
                    df, fish_name, assay, analyzer):
                    continue
                for field, value in analyzer.analyze_assay(fish[assay]).items():
                    with MetricCalculator.ACCESS_LOCK:
                        try:
                            df.loc[(fish_name, assay), field] = value
                        except Exception as e:
                            get_logger().error(
                                'Crashed on fish %s assay %d, analyzer: %s',
                                fish_name, assay, analyzer.__class__)
                            get_logger().error(e)
                            get_logger().error(traceback.format_exc())
        return df

    @staticmethod
    def _calculate_and_log_analyzer_metrics(
            analyzer: AbstractMetricAnalyzer,
            swarm_process_fish: bool=False):
        '''
        Parameters
        ----------
        analyzer: AbstractMetricAnalyzer
        swarm_process_fish: bool
            Whether to allow a worker swarm to perform inner tasks
            Only use True if you are in the main thread of the program.
            Note:
        '''
        if isinstance(analyzer, (CruiseWaveformAnalyzer, StrouhalAnalyzer)):
            swarm_process_fish = False
            get_logger().warning(' '.join([
                'You cannot swarm process cruise waveform analyzers.',
                'This avoids a buggy process.']))
        all_names = FDM.get_available_fish_names()
        progress.init(total=len(all_names))
        progress.progress(0, 'Starting...', len(all_names))
        def task(name):
            fish = Fish(name).load()
            for assay in fish.swim_keys():
                for field, value in analyzer.analyze_assay(fish[assay]).items():
                    try:
                        MetricLogger.store_metric_value(name, assay, field, value, save_to_file=False)
                    except Exception as e:
                        print(e)
                        print(traceback.format_exc())
            progress.increment(name, total=len(all_names))
        if swarm_process_fish:
            with WorkerSwarm(get_logger()) as swarm:
                for name in all_names:
                    swarm.add_task(lambda n=name: task(n))
        else:
            for name in all_names:
                task(name)
        progress.finish()
        MetricLogger.save_metric_dataframe()

    @staticmethod
    def calculate_and_log_analyzer_metrics(
            analyzer: AbstractMetricAnalyzer,
            experiment_name=None,
            swarm_process_fish: bool=False):
        '''
        Parameters
        ----------
        metric: str
        experiment_name: str, None
        swarm_process_fish: bool
            Whether to allow a worker swarm to perform inner tasks
            Only use True if you are in the main thread of the program.
        '''
        if experiment_name is None:
            experiment_name = config.experiment_name
        with AccessContext(experiment_name=experiment_name):
            MetricCalculator._calculate_and_log_analyzer_metrics(
                analyzer, swarm_process_fish=swarm_process_fish)

    @staticmethod
    def calculate_and_log_metric(metric, experiment_name=None):
        '''
        Parameters
        ----------
        metric: str
        experiment_name: str, None
        '''
        if experiment_name is None:
            experiment_name = config.experiment_name
        with AccessContext(experiment_name=experiment_name):
            MetricCalculator._calculate_and_log_analyzer_metrics(
                analyzer=MetricCalculator.get_metrics_to_analyzer_dict()[metric])
