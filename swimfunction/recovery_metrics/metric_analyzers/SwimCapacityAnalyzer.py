''' Gets swim capacity metrics from a SwimAssay.
'''
from typing import List
from swimfunction.data_access import PoseAccess
from swimfunction.pose_processing import pose_filters
from swimfunction.recovery_metrics.metric_analyzers.AbstractMetricAnalyzer import AbstractMetricAnalyzer
from swimfunction.recovery_metrics.metric_analyzers.SwimCapacityProfiler \
    import SwimCapacityProfiler
from swimfunction.recovery_metrics.metric_analyzers.swim_capacity_calculations \
    import AssayProperties, MetricOptions
from swimfunction.global_config.config import config

NO_FLOW = config.getint('FLOW', 'none')
SLOW_FLOW = config.getint('FLOW', 'slow')
FAST_FLOW = config.getint('FLOW', 'fast')
HAS_FLOW = config.getint('FLOW', 'has_flow')
FULL_ASSAY = None
FLOW_SPEEDS = (None, NO_FLOW, SLOW_FLOW, FAST_FLOW, HAS_FLOW)

def join_em(*args):
    ''' Return any number of strings, joined together with a separater.
    '''
    return '\n'.join(filter(lambda x: x, args))

def get_flow_str(flows: List[int], join_with: str='_', to_postpend: str='cms'):
    ''' Join flow values into a string with 'cms' on the end.
    '''
    def flow_to_str(f):
        return join_with.join([
                str(SLOW_FLOW), str(FAST_FLOW)
            ])if f == HAS_FLOW else str(f)
    if isinstance(flows, int):
        return f'{flow_to_str(flows)}{to_postpend}'
    if flows is None or not len(flows):
        return ''
    return f'{join_with.join(map(flow_to_str, sorted(flows)))}{to_postpend}'

class SwimCapacityAnalyzer(AbstractMetricAnalyzer):
    ''' Calculates swim capacity metrics for a swim assay.
    These calculations match as closely as possible to those described in
    Burris, B., Jensen, N., Mokalled, M. H.
    Assessment of Swim Endurance and Swim Behavior in Adult Zebrafish.
    J. Vis. Exp. (177), e63240, doi:10.3791/63240 (2021).
    '''
    def __init__(self):
        super().__init__()
        no_flow_str_printable = get_flow_str(NO_FLOW, join_with=',', to_postpend="cm/s ")
        flow_str_printable = get_flow_str(HAS_FLOW, join_with=',', to_postpend="cm/s ")
        self.keys_to_printable = {
            'swim_distance': 'Swim Distance (px)',
            'activity': 'Activity (%)',
            'centroid_burst_freq_0cms': join_em(
                'Centroid Burst Frequency', no_flow_str_printable),
            'pose_burst_freq_0cms': join_em(
                'Pose Burst Frequency', no_flow_str_printable),
            'mean_y_10_20cms': join_em(
                'Mean Y (px)', flow_str_printable),
            'time_against_flow_10_20cms': join_em(
                'Time Against Flow', flow_str_printable),
        }

    def _profiler(self, swim_assay):
        ''' Get a profiler for the assay. Default flow speed is None (entire assay).
        '''
        return SwimCapacityProfiler(
            [PoseAccess.get_feature_from_assay(
                swim_assay,
                'smoothed_coordinates',
                pose_filters.BASIC_FILTERS,
                keep_shape=True)],
            enforce_same_length=False,
            fish_names=[swim_assay.fish_name],
            labels=[swim_assay.assay_label],
            assay_properties=AssayProperties.get_assay_properties(
                swim_assay.fish_name, swim_assay.assay_label)
        )

    def analyze_assay(self, swim_assay) -> dict:
        rv = {}
        profiler = self._profiler(swim_assay)
        profiler.set_flow_speed_requirement(FULL_ASSAY)
        rv['swim_distance'] = profiler.distance_swum(MetricOptions.Centroid)[0]
        rv['activity'] = profiler.percent_active(MetricOptions.Centroid)[0]
        profiler.set_flow_speed_requirement(NO_FLOW)
        rv['centroid_burst_freq_0cms'] = profiler.burst_frequency(MetricOptions.Centroid)[0]
        rv['pose_burst_freq_0cms'] = profiler.burst_frequency(MetricOptions.PoseDerived)[0]
        profiler.set_flow_speed_requirement(HAS_FLOW)
        rv['mean_y_10_20cms'] = profiler.y_mean_std()[:, 0][0]
        rv['time_against_flow_10_20cms'] = profiler.percent_against_flow(MetricOptions.Centroid)[0]
        return rv
