''' All calculations derived from the caudal tail tip:
tail beat frequency and Strouhal.

Strouhal is only reported for 0 cm/s
Tail beat frequency is reported for 0 cm/s and full assay.
'''

from collections import namedtuple
import numpy
from tqdm import tqdm
from zplib.scalar_stats import moving_mean_std

from swimfunction.data_access.data_utils import AnnotationTypes
from swimfunction.data_access import PoseAccess
from swimfunction.recovery_metrics.metric_analyzers.AbstractMetricAnalyzer \
    import AbstractMetricAnalyzer
from swimfunction.recovery_metrics.metric_analyzers.CruiseWaveformCalculator \
    import CruiseWaveformCalculator
from swimfunction.recovery_metrics.metric_analyzers.series_to_waveform_stats \
    import get_alternating_argextrema, argpeaks_to_periods, peaks_to_amplitudes2d
from swimfunction.pose_processing import pose_filters
from swimfunction.recovery_metrics.metric_analyzers.swim_capacity_calculations \
    import distance_swum, AssayProperties
from swimfunction.global_config.config import config

FPS = config.getint('VIDEO', 'fps')
BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
ANY_FLOW = (
    config.getint('FLOW', 'none'),
    config.getint('FLOW', 'slow'),
    config.getint('FLOW', 'fast')
)
NO_FLOW = (config.getint('FLOW', 'none'),)

"""
MOVING MEAN PARAMETERS
"""
SMOOTH = 0.5
ITERS = 3

AmpFreq = namedtuple('AmpFreq', ['amplitude', 'frequency', 'speed', 'npeaks'])

AvgStdSem = namedtuple('MeanStdSem', ['avg', 'std', 'sem'])

def ampfreq_to_strouhal(ampfreq: AmpFreq):
    ''' Calculate Strouhal number for the given amplitude, frequency, speed.
    '''
    return ampfreq.frequency * ampfreq.amplitude / ampfreq.speed

class StrouhalSwimAssayHandler:
    ''' Calculates tail beat frequencies, amplitudes, and strouhal numbers.
    Only uses cruises against the flow if water is flowing, or any cruise if no flow.

    Requires 5 extrema per cruise.
    Requires the cruise to be against flow.
    '''

    min_extrema_per_cruise = 5 # 2.5 waveforms

    __slots__ = [
        'logger',
        'strouhal_numbers', 'frequencies', 'amplitudes',
        'speeds', 'peak_counts', 'fish_name', 'assay_label']
    def __init__(self, logger):
        self.logger = logger
        self.strouhal_numbers = None
        self.amplitudes = None
        self.frequencies = None
        self.speeds = None
        self.peak_counts = None
        self.fish_name = None
        self.assay_label = None

    def weighted_nan_avg_std_sem(self, data, weights) -> AvgStdSem:
        """
        Return the nan-ignored weighted average,
        weighted standard deviation,
        and weighted standard error of the mean.
        """
        if not data.size:
            return AvgStdSem(numpy.nan, numpy.nan, numpy.nan)
        masked_data = numpy.ma.masked_array(data, numpy.isnan(data))
        average = numpy.ma.average(
            masked_data,
            weights=weights)
        variance = numpy.ma.average(
            (masked_data-average)**2,
            weights=weights)
        # Weighted standard errors, see link below:
        #  https://stats.stackexchange.com/a/548429
        sem = numpy.sqrt(variance * ((weights / weights.sum())**2).sum())
        return AvgStdSem(average, numpy.sqrt(variance), sem)

    def strouhal_stats(self, flows) -> AvgStdSem:
        ''' Strouhal stats for the given flows
        '''
        st_data = numpy.concatenate([self.strouhal_numbers[f] for f in flows]).astype(float)
        weights = numpy.concatenate([self.peak_counts[f] for f in flows]).astype(float)
        return self.weighted_nan_avg_std_sem(st_data, weights)

    def tailbeat_stats(self, flows) -> AvgStdSem:
        ''' Tailbeat stats for the given flows
        '''
        tb_data = numpy.concatenate([self.frequencies[f] for f in flows]).astype(float)
        weights = numpy.concatenate([self.peak_counts[f] for f in flows]).astype(float)
        return self.weighted_nan_avg_std_sem(tb_data, weights)

    def handle_assay(self, swim_assay, base_filters):
        ''' Calculate stats for all flow speeds.
        '''
        self.fish_name = swim_assay.fish_name
        self.assay_label = swim_assay.assay_label
        self.strouhal_numbers = {
            f: [] for f in ANY_FLOW
        }
        self.amplitudes = {
            f: [] for f in ANY_FLOW
        }
        self.frequencies = {
            f: [] for f in ANY_FLOW
        }
        self.speeds = {
            f: [] for f in ANY_FLOW
        }
        self.peak_counts = {
            f: [] for f in ANY_FLOW
        }
        can_use_caudal_tip = swim_assay.caudal_fin_coordinates is not None \
            and len(swim_assay.caudal_fin_coordinates) > 0
        if not can_use_caudal_tip:
            self.logger.warning(''.join([f'Caudal fin tip not available for {self.fish_name} {self.assay_label}wpi, ',
                'so Strouhal analysis must use base of the caudal fin instead.']))
        for flow in tqdm(ANY_FLOW, leave=False):
            filters = base_filters + [
                pose_filters.TEMPLATE_filter_by_flow_rate(flow)]
            if flow != ANY_FLOW[0]:
                filters.append(pose_filters.TEMPLATE_rheotaxis([pose_filters.ORIENTED_AGAINST, pose_filters.NOT_QUITE_AGAINST]))
            if can_use_caudal_tip:
                filters.append(pose_filters.filter_caudal_fin_nan)
            else:
                filters.append(pose_filters.filter_raw_coordinates_nan)
            episodes_idx = sorted(
                PoseAccess.get_episodes_of_unmasked_data(
                    swim_assay,
                    filters,
                    min_length=CruiseWaveformCalculator.min_cruise_length),
                key=len, reverse=True)
            for episode_idx in tqdm(episodes_idx, leave=False):
                episode = PoseAccess.episode_indices_to_feature(swim_assay, episode_idx, 'smoothed_coordinates')
                if can_use_caudal_tip:
                    episode = PoseAccess.episode_indices_to_feature(swim_assay, episode_idx, 'smoothed_coordinates_with_caudal')

                ampfreq = self.estimate_mean_amplitude_frequency_speed(episode[:, -1, :], flow)
                if ampfreq.npeaks < self.min_extrema_per_cruise:
                    continue

                self.strouhal_numbers[flow].append(ampfreq_to_strouhal(ampfreq))
                self.frequencies[flow].append(ampfreq.frequency)
                self.amplitudes[flow].append(ampfreq.amplitude)
                self.speeds[flow].append(ampfreq.speed)
                self.peak_counts[flow].append(ampfreq.npeaks)

    def caudal_points_to_argpeaks(
            self,
            caudal_tip_coords: numpy.ndarray,
            smooth: float,
            iters: int):
        ''' Gets indices for peaks and troughs
        (alternating peak-trough-peak-trough-...)
        Before getting peaks, it removes a major smoothed trend.
        '''
        x_trend = moving_mean_std.moving_mean(
            numpy.arange(caudal_tip_coords.shape[0]),
            caudal_tip_coords[:, 0],
            caudal_tip_coords.shape[0],
            smooth=smooth,
            iters=iters)[1]
        return get_alternating_argextrema(caudal_tip_coords[:, 0] - x_trend)

    def estimate_mean_amplitude_frequency_speed(
            self,
            caudal_tip_coords: numpy.ndarray,
            flow_speed: int,
            smooth: float=SMOOTH,
            iters: int=ITERS):
        ''' Get mean amplitude, frequency, speed, and total number of peaks.
        '''
        argpeaks = self.caudal_points_to_argpeaks(caudal_tip_coords, smooth, iters)
        if len(argpeaks) < 3:
            # Must have at least three extrema for any waveform calculations.
            return AmpFreq(numpy.nan, numpy.nan, numpy.nan, 0)
        return AmpFreq(
            amplitude=numpy.nanmean(peaks_to_amplitudes2d(caudal_tip_coords[argpeaks])),
            frequency=numpy.nanmean(self.argpeaks_to_frequencies(argpeaks)),
            speed=numpy.nanmean(self.peaks_to_speeds(argpeaks, caudal_tip_coords, flow_speed)),
            npeaks=argpeaks.shape[0]
        )

    def peaks_to_speeds(self, argpeaks, tails, flow_speed) -> numpy.ndarray:
        ''' Estimates extreme-to-extreme speeds (px / second) for moving window of three consecutive extremes
        (peak-trough-peak; trough-peak-trough; peak-trough-peak; ...)
        '''
        pxs = numpy.asarray([
            distance_swum(
                tails[(argpeaks[i], argpeaks[i+2]), :],
                flow_speed,
                AssayProperties.get_assay_properties(self.fish_name, self.assay_label),
                adjust_for_flow=True,
                use_exhaustion_boundary=False)
            for i in range(argpeaks.shape[0]-2)
        ])
        frames = argpeaks[2:] - argpeaks[:-2]
        return FPS * pxs / frames

    def argpeaks_to_frequencies(self, argpeaks) -> numpy.ndarray:
        ''' Estimates extreme-to-extreme frequency (per second [Hz]) for end-to-end windows of three extrema
        (peak-trough-peak; peak-trough-peak; ...)
        '''
        return numpy.power(argpeaks_to_periods(argpeaks), -1, dtype=float) * FPS


class StrouhalAnalyzer(AbstractMetricAnalyzer):
    ''' Gets Strouhal number for cruises at 0 cm/s flow.
    Gets tail beat frequencies across the entire assay.
    Only uses "against the flow" cruises, or any cruise if no flow.
    '''
    def __init__(self, annotation_type=AnnotationTypes.predicted):
        super().__init__()
        self.keys_to_printable = {
            'strouhal_0cms': 'St (0 cm/s)',
            'tail_beat_freq_0cms': 'Tail Beat Frequency (0 cm/s)',
            'tail_beat_freq': 'Tail Beat Frequency'
        }
        # Get cruises away from the edges of the video.
        self.base_filters = pose_filters.BASIC_FILTERS + [
                pose_filters.filter_by_distance_from_edges,
                pose_filters.TEMPLATE_filter_by_behavior(
                    BEHAVIORS.cruise,
                    annotation_type)]

    def analyze_assay(self, swim_assay) -> dict:
        assay_handler = StrouhalSwimAssayHandler(self.logger)
        assay_handler.handle_assay(swim_assay, self.base_filters)
        tb_res_full_assay = assay_handler.tailbeat_stats(ANY_FLOW)
        tb_res_no_flow = assay_handler.tailbeat_stats(NO_FLOW)
        st_res_no_flow = assay_handler.strouhal_stats(NO_FLOW)
        rv = {
            'strouhal_0cms': st_res_no_flow.avg,
            'tail_beat_freq_0cms': tb_res_no_flow.avg,
            'tail_beat_freq': tb_res_full_assay.avg
        }
        return rv
