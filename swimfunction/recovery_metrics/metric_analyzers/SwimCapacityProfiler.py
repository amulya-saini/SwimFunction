''' Swim Capacity Metrics

NOTE: flow comes from ymax, then pushes the fish toward y=0.
Moving in positive y direction means swimming against the flow.

    These calculations match as closely as possible to those described in
    Burris, B., Jensen, N., Mokalled, M. H.
    Assessment of Swim Endurance and Swim Behavior in Adult Zebrafish.
    J. Vis. Exp. (177), e63240, doi:10.3791/63240 (2021).
'''

from typing import List
import numpy

from swimfunction.data_access import PoseAccess, data_utils
from swimfunction.data_models.Fish import Fish
from swimfunction.global_config.config import config
from swimfunction.recovery_metrics.metric_analyzers \
    import swim_capacity_calculations as SCC

NO_FLOW = config.getint('FLOW', 'none')
SLOW_FLOW = config.getint('FLOW', 'slow')
FAST_FLOW = config.getint('FLOW', 'fast')

# This is our "not no flow" flag
HAS_FLOW = config.getint('FLOW', 'has_flow')

ORIENTED_AGAINST = config.getint('RHEOTAXIS', 'against_flow')
NOT_QUITE_AGAINST = config.getint('RHEOTAXIS', 'not_quite_against_flow')
NOT_QUITE_WITH = config.getint('RHEOTAXIS', 'not_quite_with_flow')

FPS = config.getint('VIDEO', 'fps')

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')

FLC = SCC.FlowLimsCalculator(False)
FLC_REQUIRES_FLOW = SCC.FlowLimsCalculator(True)

class SwimCapacityProfiler:
    ''' Gets values for traditional metrics
    of swim performance.
    Calculating centroids and assay properties can take a bit longer that expected
    if you want to do all metrics for all fish.
    So, if used properly, this class calculates those attributes one time only.

    You can change the flow speed later, if desired, or set to None for full-assay.

    Makes sure all list lengths are at least as long as the expected final frame,
    buffering with NaN or other value as appropriate.
    '''
    x_dim = 0
    y_dim = 1

    __slots__ = [
        'pose_arrays',
        'centroids',
        'imputed_centroids',
        'predicted_behaviors',
        'orientations',
        'labels',
        'fish_names',
        'flow_speed',
        'use_exhaustion_boundary',
        '_default_assay_properties'
    ]
    def __init__(
            self, coordinate_pose_arrays: list,
            enforce_same_length: bool,
            fish_names: list,
            labels: list,
            assay_properties: SCC.AssayProperties = None,
            flow_speed=None,
            use_exhaustion_boundary: bool=True):
        '''
        Parameters
        ----------
        coordinate_pose_arrays : list
            list of arrays of coordinate poses.
        enforce_same_length : bool
            whether to clip all assays to the same length
        fish_names : list
            names of fish (same length as pose_arrays)
        labels : list
            assay labels (same length as pose_arrays)
        assay_properties : AssayProperties, None
            Provide only if you want the profiler to only work for one assay.
        flow_speed: bool, None
            Flow speed to select for metrics. Affects denominator in % activity, for example.
            If None, gives the full assay stats.
        '''
        loaded_fish = {f: Fish(f).load() for f in set(fish_names)}
        self._default_assay_properties = assay_properties
        self.pose_arrays = SCC.buffer_lists_to_full_length(coordinate_pose_arrays)
        if enforce_same_length:
            self.pose_arrays = SCC.to_rectangular_array(coordinate_pose_arrays)
            self.centroids = self.pose_arrays.mean(axis=-2)
        else:
            self.centroids = [pa.mean(axis=-2) for pa in self.pose_arrays]
        self.imputed_centroids = [
            SCC.impute_missing(a, self.get_assay_properties(name, label), inplace=False)
            for a, name, label in zip(self.centroids, fish_names, labels)]
        if enforce_same_length:
            self.imputed_centroids = numpy.asarray(self.imputed_centroids)
        self.predicted_behaviors = SCC.buffer_lists_to_full_length([
            PoseAccess.get_behaviors(loaded_fish[name][assay], data_utils.AnnotationTypes.predicted)
            for (name, assay) in zip(fish_names, labels)
        ], BEHAVIORS.unknown)
        self.orientations = list(map(SCC.pose_orientations, self.pose_arrays))
        self.labels = labels
        self.fish_names = fish_names
        self.flow_speed = flow_speed
        self.use_exhaustion_boundary = use_exhaustion_boundary

    def set_flow_speed_requirement(self, flow_speed):
        ''' Will only calculate metrics
        under this flow speed condition.
        '''
        self.flow_speed = flow_speed
        return self

    def get_assay_properties(self, fish_name, assay) -> SCC.AssayProperties:
        ''' Get assay properties for the fish and assay.
        '''
        return SCC.AssayProperties.get_assay_properties(fish_name, assay) \
            if self._default_assay_properties is None \
            else self._default_assay_properties

    def _get_appropriate_activity_metric(self, idx: int, option: int):
        ''' Gets the activity metric that matches the chosen option (see MetricOptions for choices)
        and fills gaps in the activity mask.
        '''
        assay_properties = self.get_assay_properties(self.fish_names[idx], self.labels[idx])
        activity_mask = None
        if option == SCC.MetricOptions.Centroid:
            activity_mask = SCC.get_full_assay_frame_to_frame_activity(
                self.centroids[idx],
                False,
                assay_properties,
                self.use_exhaustion_boundary)
        elif option == SCC.MetricOptions.PoseDerived:
            activity_mask = SCC.get_full_assay_frame_to_frame_activity_pose_derived(
                self.pose_arrays[idx], self.predicted_behaviors[idx])
        else:
            raise RuntimeError(f'Unrecognized metric option: {option}')
        return SCC.close_gaps(activity_mask)

    def _get_appropriate_frame_distances(self, idx: int, option: int):
        ''' Gets full assay frame to frame distances
        '''
        assay_properties = self.get_assay_properties(self.fish_names[idx], self.labels[idx])
        distance = None
        if option == SCC.MetricOptions.Centroid:
            distance = SCC.get_full_assay_frame_to_frame_distance_swum(
                self.centroids[idx],
                False,
                assay_properties,
                self.use_exhaustion_boundary)
        elif option == SCC.MetricOptions.FlowAdjustedCentroid:
            distance = SCC.get_full_assay_frame_to_frame_distance_swum(
                self.centroids[idx],
                True,
                assay_properties,
                self.use_exhaustion_boundary)
        return distance

    ##### Locations Derived

    def distance_swum(self, metric_option: int) -> list:
        ''' Returns the distance the centroid moves in the assay (swim distance)

        Parameters
        ----------
        adjust_for_flow : bool
            whether flow speed should be subtracted from distance

        Returns
        -------
        list
            numbers (distances)
        '''
        start, end = FLC.get_start_end(self.flow_speed)
        distances = []
        for i in range(len(self.fish_names)):
            distances.append(
                numpy.nansum(
                    self._get_appropriate_frame_distances(i, metric_option)[start:end]))
        return distances

    def percent_active(self, metric_option: int) -> list:
        ''' Get percent active frames using swim distance calculation.
        Activity means the centroid moved more than ACTIVITY_THRESHOLD pixels from the last frame.

        Parameters
        ----------
        adjust_for_flow : bool
            whether flow speed should be subtracted from distance
        pose_derived : bool
            whether activity should be defined as "behavior that is not rest and not unknown"
        combine_centroid_and_pose : bool
            whether to combine raw centroids with pose derived activity (logical or)

        Returns
        -------
        list
            numbers (percent active)
        '''
        start, end = FLC.get_start_end(self.flow_speed)
        percents = []
        for i in range(len(self.fish_names)):
            activity_mask = self._get_appropriate_activity_metric(i, metric_option)
            percents.append(activity_mask[start:end].sum() / activity_mask[start:end].size)
        return percents

    def burst_frequency(self, metric_option: int) -> list:
        ''' Get burst frequency (bursts / min) against flow

        "against" flow is loose here, allowing full 180 degree arc
        in the direction of flow (ORIENTED_AGAINST and NOT_QUITE_AGAINST)

        Parameters
        ----------
        adjust_for_flow : bool
            whether flow speed should be subtracted from distance
        pose_derived : bool
            whether activity should be defined as "behavior that is not rest and not unknown"
        combine_centroid_and_pose : bool
            whether to combine raw centroids with pose derived activity (logical or)

        Returns
        -------
        list
            numbers (bursts / min)
        '''
        start, end = FLC.get_start_end(self.flow_speed)
        burst_freqs = []
        for i in range(len(self.fish_names)):
            activity_mask = self._get_appropriate_activity_metric(i, metric_option)
            minutes = activity_mask[start:end].size / FPS / 60
            nbursts = 0
            if self.flow_speed in [NO_FLOW, SLOW_FLOW, FAST_FLOW]:
                nbursts += SCC.get_burst_count(activity_mask[start:end], self.orientations[i][start:end].copy(), self.flow_speed)
            elif self.flow_speed is None:
                s, e = FLC.get_start_end(NO_FLOW)
                nbursts += SCC.get_burst_count(activity_mask[s:e], self.orientations[i][s:e].copy(), NO_FLOW)
                s, e = FLC.get_start_end(SLOW_FLOW)
                nbursts += SCC.get_burst_count(activity_mask[s:e], self.orientations[i][s:e].copy(), SLOW_FLOW)
                s, e = FLC.get_start_end(FAST_FLOW)
                nbursts += SCC.get_burst_count(activity_mask[s:e], self.orientations[i][s:e].copy(), FAST_FLOW)
            elif self.flow_speed == HAS_FLOW:
                s, e = FLC.get_start_end(SLOW_FLOW)
                nbursts += SCC.get_burst_count(activity_mask[s:e], self.orientations[i][s:e].copy(), SLOW_FLOW)
                s, e = FLC.get_start_end(FAST_FLOW)
                nbursts += SCC.get_burst_count(activity_mask[s:e], self.orientations[i][s:e].copy(), FAST_FLOW)
            else:
                raise RuntimeError('Invalid flow speed.')
            burst_freqs.append(nbursts / minutes)
        return burst_freqs

    def percent_against_flow(self, metric_option: int) -> list:
        ''' Get percent time spent swimming against the flow.
        Numerator and denominator only include the frames that have flow, which is 2/3 of the video maximum.
        Against the flow means at least ACTIVITY_THRESHOLD pixels forward.
        If adjusted for flow, even drifting backward slightly
        may be "active", depending on the drift speed.

        "against" flow is loose here, allowing full 180 degree arc
        in the direction of flow (ORIENTED_AGAINST and NOT_QUITE_AGAINST)

        Parameters
        ----------
        adjust_for_flow : bool
            whether flow speed should be subtracted from distance
        pose_derived : bool
            whether activity should be defined as "behavior that is not rest and not unknown"
        combine_centroid_and_pose : bool
            whether to combine raw centroids with pose derived activity (logical or)

        Returns
        -------
        list
            numbers (percent total time swimming against flow)
        '''
        start, end = FLC_REQUIRES_FLOW.get_start_end(self.flow_speed)
        percents = []
        for i in range(len(self.fish_names)):
            if self.flow_speed == NO_FLOW:
                percents.append(numpy.nan)
                continue
            activity_mask = self._get_appropriate_activity_metric(i, metric_option)
            activity_mask = activity_mask[start:end]

            directions = self.orientations[i][start:end]

            percents.append(activity_mask[numpy.in1d(directions, [ORIENTED_AGAINST, NOT_QUITE_AGAINST])].sum() / activity_mask.size)
        return percents

    ##### Locations

    def location_heatmap(self) -> List[numpy.ndarray]:
        ''' Get heatmap of location in the tunnel.
        Each point along the centerline in each frame contributes a 2D gaussian blob.
        All blobs together over the whole video are aggregated to show where
        the fish liked to hang out.

        Uses imputed centroids.

        Returns
        -------
        heatmaps : numpy.ndarray
            location heatmaps
        '''
        start, end = FLC.get_start_end(self.flow_speed)
        def get_imshape(fish_name, assay):
            assay_properties = self.get_assay_properties(fish_name, assay)
            return (assay_properties.height, assay_properties.width)
        heatmaps = [
            SCC.get_heatmap(
                i_centroids[start:end, ...],
                get_imshape(fish_name, assay))
            for fish_name, assay, i_centroids in zip(self.fish_names, self.labels, self.imputed_centroids)
        ]
        return heatmaps

    def _position(self, dim) -> list:
        ''' Calculates (imputed) centroid y- or x-positions for each fish.
        It will be a numpy.ndarray if enforce_same_length,
        otherwise each row is the length of the assay thus
        returning a list of numpy.ndarray instead.

        Returns
        -------
        centroid_dimension_positions : list, numpy.ndarray
            list of x or y position time series
            shape=(num_fish, num_frames)
        '''
        start, end = FLC.get_start_end(self.flow_speed)
        if isinstance(self.imputed_centroids, numpy.ndarray):
            return self.imputed_centroids[..., start:end, dim]
        return [c[start:end, dim] for c in self.imputed_centroids]

    def y_position(self) -> list:
        ''' Calculates (imputed) centroid y-positions for each fish.
        It will be a numpy.ndarray if enforce_same_length,
        otherwise each row is the length of the assay thus
        returning a list of numpy.ndarray instead.

        Returns
        -------
        centroid_y_positions : list
            list of y position time series
            shape=(num_fish, num_frames)
        '''
        return self._position(SwimCapacityProfiler.y_dim)

    def y_mean_std(self) -> numpy.ndarray:
        ''' Get mean and std for y positions.
        '''
        return numpy.asarray([
            [numpy.nanmean(l), numpy.nanstd(l)]
            for l in self.y_position()])

    def x_position(self) -> list:
        ''' Calculates (imputed) centroid x-positions for each fish.
        It will be a numpy.ndarray if enforce_same_length,
        otherwise each row is the length of the assay
        thus returning a list of numpy.ndarray instead.

        Returns
        -------
        centroid_x_positions : list
            list of x position time series
            shape=(num_fish, num_frames)
        '''
        return self._position(SwimCapacityProfiler.x_dim)

    def x_mean_std(self) -> numpy.ndarray:
        ''' Get mean and std for x positions.
        '''
        return numpy.asarray([
            [numpy.nanmean(l), numpy.nanstd(l)]
            for l in self.x_position()
        ])
