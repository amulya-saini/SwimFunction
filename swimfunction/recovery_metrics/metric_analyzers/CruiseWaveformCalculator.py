''' Calculate waveform features (amplitudes, frequencies)
at all keypoint positions along the dorsal centerline
for episodes of cruise behavior.
'''
import threading
from tqdm import tqdm
import numpy
import pandas

from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import PoseAccess, data_utils
from swimfunction.data_models.Fish import Fish
from swimfunction.pose_processing import pose_filters
from swimfunction.context_managers.AccessContext import AccessContext
from swimfunction.recovery_metrics.metric_analyzers import series_to_waveform_stats
from swimfunction.global_config.config import config
from swimfunction import loggers
from swimfunction import FileLocations

def get_logger() -> loggers.logging.Logger:
    ''' Get the default metric calculation logger.
    '''
    return loggers.get_metric_calculation_logger(__name__)

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
FLOW_SPEEDS = (
    config.getint('FLOW', 'none'),
    config.getint('FLOW', 'slow'),
    config.getint('FLOW', 'fast')
)


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
        wstats: series_to_waveform_stats.WaveformStats, cruise):
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
        series_to_waveform_stats.get_waveform_stats_from_many_series(
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
            if outpath.exists():
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
            if outpath.exists():
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
            if outpath is not None:
                featuresdf.to_hdf(outpath, key=self.h5featureskey, mode='w')

    def _write_summary_to_cache(self, summarydf: pandas.DataFrame, summary_flows, assay):
        ''' Save the summary data to a file.
        '''
        with AccessContext(experiment_name=self.experiment):
            get_logger().info('Writing summary to cache...')
            outpath = self._choose_cache_file(assay)
            if outpath is not None:
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
