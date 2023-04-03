import functools
import threading
import traceback
import numpy

from swimfunction.data_access.data_utils import AnnotationTypes
from swimfunction.recovery_metrics.metric_analyzers\
    .AbstractMetricAnalyzer import AbstractMetricAnalyzer
from swimfunction.recovery_metrics.metric_analyzers\
    .CruiseWaveformCalculator import CruiseWaveformCalculator
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.global_config.config import config
from swimfunction.data_access import data_utils

from swimfunction import FileLocations
from swimfunction import loggers

NO_FLOW = config.getint('FLOW', 'none')
SLOW_FLOW = config.getint('FLOW', 'slow')
FAST_FLOW = config.getint('FLOW', 'fast')

def get_logger() -> loggers.logging.Logger:
    ''' Get the default metric calculation logger.
    '''
    return loggers.get_metric_calculation_logger(__name__)

class CruiseWaveformAnalyzer(AbstractMetricAnalyzer):
    ''' Get cruise waveform stats: rostral compensation score and body frequency.

    Regarding frequencies: we want the mean frequency because frequency
    is pretty much uniform along the dorsal centerline.

    Regarding amplitudes: we want the the sum of each position's mean observation
        because we care about total deviation along the centerline.

        Because we have per-dimensions means, we can take the sum of the series.
        This is because mean(a + b + c) == mean(a) + mean(b) + mean(c).
        In English, if there are three positions on the fish, a, b, and c,
        then the average sum for all observations of these positions
        is equal to the sum of each position's mean observation.

        If we took series.mean() instead, then it would be the mean of all observations
        of a, b, and c concatenated. mean([mean(a), mean(b), mean(c)]) == mean(concat(a, b, c))

        These properties are true because a, b, and c are all observed the same number of times
        (same denominator for the means).
    '''
    PRECALCULATION_LOCK = threading.Lock()
    __slots__ = ['control_series']
    def __init__(self, rostral_compensation_only=True):
        super().__init__()
        self.control_series = None
        self.keys_to_printable = {
            'rostral_compensation': 'Rostral Compensation'
        }
        if not rostral_compensation_only:
            self.keys_to_printable['body_frequency'] = 'Body Cruise Frequency'

    def set_control(
            self,
            groups=None,
            assay=-1,
            flows=None,
            names_to_omit=None):
        ''' Takes the mean waveform stat at the fish (biological replicate) level.
        Note: the fish stats are the mean of all their cruises (technical replicates).
        '''
        if groups is None:
            groups = FDM.get_groups()
        print('Setting predicted cruises as the control, if available.')
        wf_calc = CruiseWaveformCalculator(config.experiment_name, AnnotationTypes.predicted)
        self.control_series = {
            group: wf_calc.get_waveform_stats(
                    group=group,
                    assay=assay,
                    flows=flows,
                    names_to_omit=names_to_omit).mean(axis=0)
            for group in groups
        }
        if numpy.all(numpy.logical_not([v.size for v in self.control_series.values()])) \
                and not list(FileLocations.get_behaviors_model_dir().glob('*.gz')):
            print('Predicted cruises are unavailable. Using human annotations if possible.')
            wf_calc = CruiseWaveformCalculator(config.experiment_name, AnnotationTypes.human)
            self.control_series = {
                group: wf_calc.get_waveform_stats(
                        group=group,
                        assay=assay,
                        flows=flows,
                        names_to_omit=names_to_omit).mean(axis=0)
                for group in groups
            }
        return self

    def _select_from_series(self, series, feature: str, dimensions: list=None):
        condition = (series.index.get_level_values('feature') == feature) \
            & (series.index.get_level_values('stat') == 'mean')
        if dimensions:
            dimension_conditions = [
                (series.index.get_level_values('dim') == dim)
                for dim in dimensions]
            condition = condition & functools.reduce(lambda a, b: a | b, dimension_conditions)
        return series.loc[condition]

    def _get_deviation_from_control(self, series, stat_fn):
        if not series.size:
            return numpy.nan
        group = data_utils.fish_name_to_group(series.name)
        if len(self.control_series) == 1:
            group = list(self.control_series.keys())[0]
        vals = (self.control_series[group][series.index] - series).abs().values
        if not vals.size:
            return numpy.nan
        return stat_fn(vals)

    def analyze_assay(self, swim_assay) -> dict:
        rv = {key: None for key in self.keys_to_printable}
        if not swim_assay.predicted_behaviors.size:
            return rv
        logger = get_logger()
        with CruiseWaveformAnalyzer.PRECALCULATION_LOCK:
            try:
                if self.control_series is None:
                    logger.info('Must precalculate control waveform stats...')
                    self.set_control()
                    logger.info('Waveform control precalculation is complete.')
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(e)
        df = CruiseWaveformCalculator(
                    config.experiment_name,
                    AnnotationTypes.predicted).get_waveform_stats(
                group=None,
                assay=swim_assay.assay_label,
                flows=None,
                names_to_omit=None)
        if swim_assay.fish_name not in df.index.values:
            return rv
        series = df.loc[swim_assay.fish_name]
        rv['rostral_compensation'] = self._get_deviation_from_control(
            self._select_from_series(series, 'amplitudes', list(range(5))),
            numpy.nanmax)
        if 'body_frequency' in rv:
            rv['body_frequency'] = self._select_from_series(
                series,
                'frequencies',
                list(range(2, 9))).mean(skipna=True)
        return rv
