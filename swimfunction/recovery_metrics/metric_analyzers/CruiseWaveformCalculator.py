''' Calculate waveform features (amplitudes, frequencies)
at all keypoint positions along the dorsal centerline
for episodes of cruise behavior.
'''
import threading
from typing import List
from collections import namedtuple
from scipy.signal import argrelextrema
from tqdm import tqdm
import numpy
import pandas

from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import PoseAccess, data_utils
from swimfunction.data_models.Fish import Fish
from swimfunction.pose_processing import pose_filters
from swimfunction.context_managers.AccessContext import AccessContext
from swimfunction.global_config.config import config
from swimfunction import loggers
from swimfunction import FileLocations

def get_logger() -> loggers.logging.Logger:
    ''' Get the default metric calculation logger.
    '''
    return loggers.get_metric_calculation_logger(__name__)

TESTING = config.getboolean('TEST', 'test')

FPS = config.getint('VIDEO', 'fps')
BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
FLOW_SPEEDS = (
    config.getint('FLOW', 'none'),
    config.getint('FLOW', 'slow'),
    config.getint('FLOW', 'fast')
)

WaveformStats = namedtuple(
    'WaveformStats',
    ['periods', 'frequencies', 'amplitudes', 'start_end_frames']
)

def get_alternating_argextrema(one_series: numpy.ndarray) -> numpy.ndarray:
    ''' Locates extrema in series. Ensures alternates crest-trough-crest-trough-...

    Parameters
    ----------
    one_series : numpy.ndarray
        1D array

    Returns
    -------
    peaks : numpy.ndarray
    '''
    argpeaks = argrelextrema(one_series, numpy.greater)[0]
    argtroughs = argrelextrema(one_series, numpy.less)[0]
    argextrema = []
    # Using len because it may be list or numpy.ndarray
    # pylint: disable-next=use-implicit-booleaness-not-len
    if not len(argpeaks) or not len(argtroughs):
        return numpy.asarray(argextrema, dtype=int)
    ti = 0
    pi = 0
    collecting_peak = argpeaks[0] < argtroughs[0]
    def collect_extrema(argvals, i):
        while argextrema and i < len(argvals) and argvals[i] < argextrema[-1]:
            # ignore any peaks that got skipped (if two peaks in a row, for example)
            i += 1
        return i
    while pi < len(argpeaks) or ti < len(argtroughs):
        if collecting_peak:
            pi = collect_extrema(argpeaks, pi)
            if pi < len(argpeaks):
                argextrema.append(argpeaks[pi])
            pi += 1
        else:
            ti = collect_extrema(argtroughs, ti)
            if ti < len(argtroughs):
                argextrema.append(argtroughs[ti])
            ti += 1
        collecting_peak = not collecting_peak
    return numpy.asarray(argextrema, dtype=int)

def remove_false_waves(
        argpeaks: numpy.ndarray, one_series: numpy.ndarray) -> numpy.ndarray:
    ''' If the distance between two extrema is less than a threshold,
    then it skips the next two extrema
    (to make sure the peak-trough-peak-trough-... pattern is kept)
    '''
    if argpeaks.size < 3:
        return argpeaks
    approx_amplitudes = numpy.abs(one_series[argpeaks[:-1]] - one_series[argpeaks[1:]])
    threshold = numpy.median(approx_amplitudes) * 0.1
    if numpy.all(approx_amplitudes > threshold):
        return argpeaks
    new_peaks = []
    i = 0
    while i < argpeaks.size:
        if i < approx_amplitudes.size and approx_amplitudes[i] < threshold:
            i += 2
        else:
            new_peaks.append(argpeaks[i])
            i += 1
    return numpy.asarray(new_peaks, dtype=int)

def peaks_to_amplitudes2d(peaks2d) -> numpy.ndarray:
    ''' Estimates amplitudes for end-to-end windows of three extrema
    (peak-trough-peak; peak-trough-peak; ...)
    Note: amplitude is half the peak-to-peak distance, or the distance
    from a peak to the midline.

    Note on Implementation
    ----------------------
    A wave with the following peak-trough pattern
        p-t-p-t-p-t-p
    has three measurable periods (peak to peak)
    and three measurable amplitudes (peak to trough to peak).

    Midline is estimated by the line through
    midpoint(peak1->peak2) and midpoint(peak2->peak3),
    not assumed to be at y=0

    A measureable amplitude is the distance between three successive extrema divided by 4.
    Basically it is the average of peak to trough divided by 2 and trough to next peak divided by 2.
    The value is calculated from the original series, but the limits
    (beginning and end of waveform) are calculated from the series minus the trend.

    https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
    '''
    peak1 = peaks2d[:-2:2]
    peak2 = peaks2d[1:-1:2]
    peak3 = peaks2d[2::2]
    midp1 = (peak2 + peak1) / 2
    midp2 = (peak3 + peak2) / 2
    def dist_to_midline(point):
        return numpy.abs(
            numpy.cross(midp2 - midp1, midp1 - point, axis=1)
            / numpy.linalg.norm(midp2 - midp1, axis=1)
        )
    amps = (
        dist_to_midline(peak1) + (2 * dist_to_midline(peak2)) + dist_to_midline(peak3)
    ) / 4
    return amps

def argpeaks_to_periods(argpeaks: numpy.ndarray) -> numpy.ndarray:
    ''' Estimates extreme-to-extreme periods (unit: frames) for end-to-end windows of three extrema
        (peak-trough-peak; peak-trough-peak; ...)

    Note on Implementation
    ----------------------
    A wave with the following peak-trough pattern
        p-t-p-t-p-t-p
    has three measurable periods (peak to peak)
    and three measurable amplitudes (peak to trough to peak).

    A measurable period is the time it takes for the time series to
    go from one extrema to the next extrema of the same type (peak to peak or trough to trough).
    The extrema type is the first extrema observed, peak or trough.
    Shouldn't matter which one is chosen.
    This is calculated from the series minus the trend.
    '''
    return argpeaks[2::2] - argpeaks[:-2:2]

def periods_to_frequencies(periods: numpy.ndarray) -> numpy.ndarray:
    ''' Converts periods to frequencies
    '''
    # Using len because it may be list or numpy.ndarray
    # pylint: disable-next=use-implicit-booleaness-not-len
    if periods is None or not len(periods):
        return []
    return FPS / periods

def get_waveform_stats_from_series(
        one_series: numpy.ndarray,
        series_idx: list,
        min_waveforms_per_cruise: int) -> WaveformStats:
    ''' Gets all measurable periods and amplitudes for each dimension in series.

    Parameters
    ----------
    one_series : numpy.ndarray
        NxD array where N is number of observations and D is number of measured dimensions
    min_waveforms_per_cruise : int
        If there is no dimension that has at least this number of waveforms, None is returned.

    Returns
    -------
    WaveformStats : namedtuple, None
        (periods, frequencies, amplitudes) or None if does not have min_waveforms_per_cruise
    '''
    series = numpy.asarray(one_series)
    if len(series.shape) == 1:
        series = series[:, numpy.newaxis]
    periods = [[] for _ in range(series.shape[1])]
    amplitudes = [[] for _ in range(series.shape[1])]
    for d in range(series.shape[1]):
        argpeaks = remove_false_waves(get_alternating_argextrema(
            series[:, d]).astype(int), series[:, d])
        if len(argpeaks) < 3:
            continue
        periods[d] = argpeaks_to_periods(argpeaks)
        peaks2d = numpy.asarray([argpeaks, numpy.take(series[:, d], argpeaks)]).T
        amplitudes[d] = peaks_to_amplitudes2d(peaks2d)
    if max([len(x) for x in periods]) < min_waveforms_per_cruise:
        return None
    frequencies = list(map(periods_to_frequencies, periods))
    return WaveformStats(periods, frequencies, amplitudes, [series_idx[0], series_idx[-1]])

def get_waveform_stats_from_many_series(
        episodes, episodes_idx, min_waveforms_per_cruise) -> List[WaveformStats]:
    ''' Get waveform features from a list of episodes
    '''
    return list(filter(
        lambda x: x is not None,
        [
            get_waveform_stats_from_series(e, e_idx, min_waveforms_per_cruise)
            for e, e_idx in tqdm(
                zip(episodes, episodes_idx),
                total=len(episodes),
                leave=False)
        ]))

class SummaryDFHolder:
    ''' Holds DataFrames with summary statistics
    '''
    def __init__(self):
        self.summary_dfs = {}

    def contains(self, annotation_type, assay_key, h5_flows_key):
        return (annotation_type in self.summary_dfs \
                and assay_key in self.summary_dfs[annotation_type] \
                and h5_flows_key in self.summary_dfs[annotation_type][assay_key] \
                and self.summary_dfs[annotation_type][assay_key][h5_flows_key] is not None)

    def ensure_exists(self, annotation_type, assay_key, h5_flows_key):
        if annotation_type not in self.summary_dfs:
            self.summary_dfs[annotation_type] = {}
        if assay_key not in self.summary_dfs[annotation_type]:
            self.summary_dfs[annotation_type][assay_key] = {}
        if h5_flows_key not in self.summary_dfs[annotation_type][assay_key]:
            self.summary_dfs[annotation_type][assay_key][h5_flows_key] = None

    def set_summary_df(self, df: pandas.DataFrame, annotation_type, assay_key, h5_flows_key):
        self.ensure_exists(annotation_type, assay_key, h5_flows_key)
        self.summary_dfs[annotation_type][assay_key][h5_flows_key] = df

    def get_summary_df(self, annotation_type, assay_key, h5_flows_key):
        self.ensure_exists(annotation_type, assay_key, h5_flows_key)
        return self.summary_dfs[annotation_type][assay_key][h5_flows_key]

class DimensionHolder:
    ''' Stores information for a specific keypoint (dimension)
    '''
    __slots__ = ['fish_name', 'flow_speed', 'dim', 'periods', 'frequencies', 'amplitudes']
    def __init__(self, fish_name, flow_speed, dim):
        self.fish_name = fish_name
        self.flow_speed = flow_speed
        self.dim = dim
        self.periods = []
        self.frequencies = []
        self.amplitudes = []

class PandaMaker:
    ''' Stores content that can become a pandas DataFrame
    '''
    FEATURES_COLUMNS = ['name', 'flow', 'feature', 'cruise', 'dimension']
    STATS_COLUMNS = ['name', 'flow', 'feature', 'stat', 'dimension']

    __slots__ = ['data', 'indices', 'content_lock']
    def __init__(self):
        self.content_lock = threading.Lock()
        self.data = []
        self.indices = []

    def add_content(self, panda_maker):
        with self.content_lock:
            self.data = self.data + panda_maker.data
            self.indices = self.indices + panda_maker.indices

    def to_pandas(self, is_summary=False):
        if is_summary:
            df = pandas.DataFrame(self.data).transpose()
            df.columns = self.indices
            df.columns = pandas.MultiIndex.from_tuples(df.columns, names=self.STATS_COLUMNS)
        else:
            df = pandas.DataFrame.from_dict(
                dict(zip(self.indices, self.data)),
                orient='index'
            ).transpose()
            df.columns = self.indices
            df.columns = pandas.MultiIndex.from_tuples(df.columns, names=self.FEATURES_COLUMNS)
        return df

def _update_data_and_indices(
        dim_holder: DimensionHolder,
        pm_data: PandaMaker,
        wstats: WaveformStats, cruise):
    dim = dim_holder.dim
    fish_name = dim_holder.fish_name
    flow_speed = dim_holder.flow_speed
    if dim == 0:
        pm_data.data.append(wstats.start_end_frames)
        pm_data.indices.append((fish_name, flow_speed, 'start_end_frames', cruise, dim))
    dim_holder.periods.append(wstats.periods[dim])
    dim_holder.frequencies.append(wstats.frequencies[dim])
    dim_holder.amplitudes.append(wstats.amplitudes[dim])
    pm_data.data.append(wstats.periods[dim])
    pm_data.data.append(wstats.frequencies[dim])
    pm_data.data.append(wstats.amplitudes[dim])
    pm_data.indices.append((fish_name, flow_speed, 'periods', cruise, dim))
    pm_data.indices.append((fish_name, flow_speed, 'frequencies', cruise, dim))
    pm_data.indices.append((fish_name, flow_speed, 'amplitudes', cruise, dim))
    return dim_holder, pm_data

def store_waveform_stats_for_assay(
        waveform_calculator,
        name,
        flow,
        episodes,
        episodes_idx,
        panda_maker: PandaMaker):
    pm_data2 = waveform_calculator.parse_new_data_and_indices(
        name,
        flow,
        get_waveform_stats_from_many_series(
            episodes,
            episodes_idx,
            waveform_calculator.min_waveforms_per_cruise))
    panda_maker.add_content(pm_data2)

class CruiseWaveformCalculator:
    ''' Calculates cruise waveform statistics
    like the amplitude and frequency of
    undulatory movement.
    '''
    df_summaries = SummaryDFHolder()

    __slots__ = [
        'experiment',
        'annotation_type',
        'base_filters']

    PRECALCULATION_LOCK = threading.Lock()
    min_waveforms_per_cruise = 3
    min_cruise_length = 10 # poses
    h5featureskey = 'wavestats'

    def __init__(self, experiment_name: str, annotation_type: str):
        self.experiment = experiment_name
        self.annotation_type = annotation_type
        with AccessContext(experiment_name=self.experiment):
            self.base_filters = pose_filters.BASIC_FILTERS + [
                pose_filters.filter_by_distance_from_edges,
                pose_filters.TEMPLATE_filter_by_behavior(
                    BEHAVIORS.cruise,
                    annotation_type)]

    def get_raw_waveform_stats(
            self,
            group: str,
            assay: int,
            flows: list,
            names_to_omit=None) -> pandas.DataFrame:
        ''' Gets waveform stats per fish
        (each dimension's mean of all cruise means in the chosen assay.)
        Rows are fish, columns are multi-index [dim, feature, stat]

        Not necessarily threadsafe!

        Parameters
        ----------
        group: str, None
            If None, selects all groups.
        assay: int
            Must not be None.
        flows: list, None
            Which flow speeds you want to include. If None, selects all flow speeds.
        names_to_omit: (optional) list, None
            List of names that should not be included in the calculations.
        '''
        with AccessContext(experiment_name=self.experiment):
            df = self._read_cached(assay)
            if not df.size:
                get_logger().info(
                    'Precalculating all waveform stats for group %s assay %d flows %s',
                    group, assay, ','.join([str(fl) for fl in flows]))
                df = self._load_and_store_assay_data(assay)
            if not df.size:
                return None
            if flows is None:
                flows = FLOW_SPEEDS
            names = set(self._choose_names(df.columns.get_level_values(0), group, names_to_omit))
            flows = set(flows)
            if not names or not flows:
                get_logger().critical(
                    'group:%s assay:%d flows:%s There are no waveform stats for these.',
                    group, assay, ','.join([str(fl) for fl in flows]))
            in_names = numpy.vectorize(lambda x: x in names)
            in_flows = numpy.vectorize(lambda x: x in flows)
            df = df.loc[:,
                in_names(df.columns.get_level_values(0)) \
                & in_flows(df.columns.get_level_values(1))]
        return df

    def get_waveform_stats(
            self,
            group: str,
            assay: int,
            flows: list,
            names_to_omit=None) -> pandas.DataFrame:
        ''' Gets waveform stats per fish
        (each dimension's mean of all cruise means in the chosen assay.)
        Rows are fish, columns are multi-index [dim, feature, stat]

        Parameters
        ----------
        group: str, None
            If None, selects all groups.
        assay: int
            Must not be None.
        flows: list, None
            Which flow speeds you want to include. If None, selects all flow speeds.
        names_to_omit: (optional) list, None
            List of names that should not be included in the calculations.
        '''
        with AccessContext(experiment_name=self.experiment):
            if flows is None:
                flows = FLOW_SPEEDS
            with CruiseWaveformCalculator.PRECALCULATION_LOCK:
                summarydf = self._read_summary_cached(assay, flows)
                if summarydf is None:
                    get_logger().info('Precalculating waveform stats...')
                    waveform_df = self.get_raw_waveform_stats(None, assay, flows)
                    if waveform_df is None:
                        return pandas.DataFrame()
                    summarydf = self._summarize_fish(waveform_df)
                    self._write_summary_to_cache(summarydf, flows, assay)
            # Choose groups and fish
            names = self._choose_names(summarydf.index.values, group, names_to_omit)
        return summarydf.loc[names]

    # NOTE if you want to test medians, update this to include median and MAD.
    @staticmethod
    def _summarize_fish(df):
        get_logger().info('Summarizing fish...')
        stats_df = df.T.sort_index()
        names = stats_df.index.get_level_values(0).unique()
        ndims = stats_df.index.get_level_values(4).unique().size
        summary_df = pandas.DataFrame(
            [],
            index=names,
            columns=pandas.MultiIndex.from_product(
                [
                    range(ndims),
                    ['amplitudes', 'frequencies', 'periods'],
                    ['mean', 'std']
                ],
                names=['dim', 'feature', 'stat']))
        for feature, stat in summary_df.columns.droplevel(0).unique().values:
            # Summarize each cruise separately (per fish).
            grouped = stats_df.loc[stats_df.index.get_level_values(2) == feature] \
                .mean(axis=1, skipna=True) \
                .groupby(by=['name', 'dimension'])
            # Summarize all cruises (per fish).
            if stat == 'mean':
                grouped = grouped.mean()
            else:
                grouped = grouped.std()
            vals = grouped.values.reshape((-1, ndims))
            col_tuples = [(d, feature, stat) for d in range(ndims)]
            summary_df[col_tuples] = vals
        return summary_df

    def _choose_names(self, all_names, group, names_to_omit):
        if names_to_omit is None:
            names_to_omit = []
        names = all_names
        if group is not None:
            names = list(filter(lambda n: data_utils.fish_name_to_group(n) == group, names))
        names = list(filter(lambda x: x not in names_to_omit, names))
        return names

    def _load_and_store_assay_data(self, assay, store_cache: bool=True):
        pm_data = PandaMaker()
        with WorkerSwarm(loggers.get_metric_calculation_logger(__name__)) as swarm:
            with AccessContext(experiment_name=self.experiment):
                for name in tqdm(FDM.get_available_fish_names()):
                    if assay not in FDM.get_available_assay_labels(name):
                        continue
                    swim = Fish(name=name).load()[assay]
                    for flow in tqdm(FLOW_SPEEDS, leave=False):
                        episodes_idx = PoseAccess.get_episodes_of_unmasked_data(
                            swim,
                            self._get_filters(flow),
                            min_length=self.min_cruise_length)
                        if not episodes_idx:
                            continue
                        episodes_idx = sorted(episodes_idx, key=len, reverse=True)
                        episodes = PoseAccess.episode_indices_to_feature(
                            swim, episodes_idx, 'smoothed_angles')
                        swarm.add_task(
                            lambda s=self,
                                n=name,
                                f=flow,
                                e=episodes,
                                e_idx=episodes_idx,
                                p=pm_data: store_waveform_stats_for_assay(s, n, f, e, e_idx, p))
        featuresdf = pm_data.to_pandas(is_summary=False)
        if store_cache:
            self._write_to_cache(featuresdf, assay)
        return featuresdf

    def _get_filters(self, flow_speed):
        return self.base_filters + [pose_filters.TEMPLATE_filter_by_flow_rate(flow_speed)]

    def parse_new_data_and_indices(self, fish_name, flow_speed, stats_arr) -> PandaMaker:
        ''' Compiles waveform properties into a PandaMaker
        '''
        with AccessContext(experiment_name=self.experiment):
            pm_data = PandaMaker()
            if len(stats_arr) == 0:
                return pm_data
            for dim in range(len(stats_arr[0][0])):
                dim_holder = DimensionHolder(fish_name, flow_speed, dim)
                for cruise, wstats in enumerate(stats_arr):
                    dim_holder, pm_data = _update_data_and_indices(
                        dim_holder, pm_data, wstats, cruise)
        return pm_data

    def _choose_cache_file(self, assay):
        with AccessContext(experiment_name=self.experiment):
            fpath = FileLocations.get_waveform_file(assay, self.annotation_type)
        return fpath

    def _read_cached(self, assay) -> pandas.DataFrame:
        with AccessContext(experiment_name=self.experiment):
            key = self.h5featureskey
            prior_data = pandas.DataFrame()
            outpath = self._choose_cache_file(assay)
            if outpath.exists() and not TESTING:
                try:
                    prior_data = pandas.read_hdf(outpath, key=key)
                except KeyError:
                    get_logger().warning('Cache is corrupt. Will calculate a fresh copy.')
        return prior_data

    def _read_summary_cached(self, assay, summary_flows):
        with AccessContext(experiment_name=self.experiment):
            assay_key = self.summary_df_assay_key(assay)
            h5_flows_key = self.h5summarykey(summary_flows)
            outpath = self._choose_cache_file(assay)
            if CruiseWaveformCalculator.df_summaries.contains(
                    self.annotation_type, assay_key, h5_flows_key):
                return CruiseWaveformCalculator.df_summaries.get_summary_df(
                    self.annotation_type, assay_key, h5_flows_key)
            result_df = None
            if outpath.exists() and not TESTING:
                try:
                    result_df = pandas.read_hdf(outpath, key=h5_flows_key)
                except KeyError:
                    get_logger().info(
                        'Cache does not contain summary for these flows. Will calculate now.')
            CruiseWaveformCalculator.df_summaries.set_summary_df(
                result_df, self.annotation_type, assay_key, h5_flows_key)
        return CruiseWaveformCalculator.df_summaries.get_summary_df(
            self.annotation_type, assay_key, h5_flows_key)

    def _write_to_cache(self, featuresdf: pandas.DataFrame, assay):
        ''' Save the big dataframe to a file
        '''
        with AccessContext(experiment_name=self.experiment):
            outpath = self._choose_cache_file(assay)
            if outpath is not None and not TESTING:
                featuresdf.to_hdf(outpath, key=self.h5featureskey, mode='w')

    def _write_summary_to_cache(self, summarydf: pandas.DataFrame, summary_flows, assay):
        ''' Save the summary data to a file.
        '''
        with AccessContext(experiment_name=self.experiment):
            get_logger().info('Writing summary to cache...')
            outpath = self._choose_cache_file(assay)
            if outpath is not None and not TESTING:
                assay_key = self.summary_df_assay_key(assay)
                h5_flows_key = self.h5summarykey(summary_flows)
                summarydf.to_hdf(outpath, key=h5_flows_key, mode='a')
                flowstr = h5_flows_key.replace('wavestats_', '')
                csv_outpath = FileLocations.mkdir_and_return(
                        FileLocations.get_csv_output_dir() / 'cruise_waveform'
                    ) / f'{outpath.with_suffix("").name}_at_{flowstr}cms.csv'
                summarydf.to_csv(csv_outpath, mode='w')
                CruiseWaveformCalculator.df_summaries.set_summary_df(
                    summarydf, self.annotation_type, assay_key, h5_flows_key)

    @staticmethod
    def h5summarykey(flows):
        return f'wavestats_{"_".join(map(str, sorted(flows)))}'

    @staticmethod
    def summary_df_assay_key(assay):
        return f'{config.experiment_name}_{assay}'
