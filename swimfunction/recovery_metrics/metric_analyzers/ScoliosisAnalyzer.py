from typing import List
import numpy

from swimfunction.data_access import PoseAccess
from swimfunction.data_access.data_utils import AnnotationTypes
from swimfunction.pose_processing import pose_filters
from swimfunction.recovery_metrics.metric_analyzers\
    .AbstractMetricAnalyzer import AbstractMetricAnalyzer
from swimfunction.global_config.config import config

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
SLOW_FLOW_START = config.getint('FLOW', 'slow_start')

METRIC_COLUMN_NAME = 'scoliosis'

class ScoliosisAnalyzer(AbstractMetricAnalyzer):
    ''' Calculate lateral scoliosis
    from episodes of rest behavior.
    '''

    def __init__(self):
        super().__init__()
        self.keys_to_printable = {'scoliosis': 'Lateral Scoliosis'}

    def analyze_assay(self, swim_assay) -> dict:
        ''' We take the weighted average of scoliosis scores for all rest episodes.
        Why use a weighted average?
            Longer rest episodes are likely more "restful".
            The beginning or end of a rest are more likely to include borderline active behavior.
        '''
        episode_idxs = PoseAccess.get_behavior_episodes_from_assay(
            swim_assay,
            BEHAVIORS.rest,
            AnnotationTypes.predicted,
            filters=pose_filters.BASIC_FILTERS + [pose_filters.filter_by_distance_from_edges])
        scoliosis = self.get_averaged_scoliosis(swim_assay, episode_idxs)
        if numpy.isnan(scoliosis):
            self.logger.warning(
                'Fish %s may not exist at %d',
                swim_assay.fish_name,
                swim_assay.assay_label)
        return {
            'scoliosis': scoliosis
        }

    def get_averaged_scoliosis(self, swim_assay, rest_episode_idxs: List[numpy.ndarray]):
        ''' Get average scoliosis for all rest episodes
        weighed by the length of the episodes.
        '''
        episode_lengths = list(map(len, rest_episode_idxs))
        if not sum(episode_lengths):
            return numpy.nan
        return numpy.average([
            self.get_lateral_scoliosis_from_angle_poses(
                PoseAccess.episode_indices_to_feature(
                    swim_assay, e, 'smoothed_angles'))
            for e in rest_episode_idxs], weights=episode_lengths)

    @staticmethod
    def get_lateral_scoliosis_from_angle_poses(angle_poses):
        ''' Absolute sum of the average angle pose.
        Note: do not absolute value before taking the mean! Signed oscillatory signal averages out to zero, which is good.
        '''
        return numpy.abs(angle_poses.mean(axis=0)).sum()
