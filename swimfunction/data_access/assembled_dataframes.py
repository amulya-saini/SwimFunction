''' Get assembled data, metrics and cruise waveforms according to criteria.
'''

from collections import namedtuple
import numpy
import pandas

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access.MetricLogger import MetricLogger
from swimfunction.data_access import data_utils
from swimfunction.recovery_metrics.metric_analyzers\
    .CruiseWaveformCalculator import CruiseWaveformCalculator
from swimfunction.global_config.config import config
from swimfunction import loggers

LOGGER = loggers.get_console_logger(__name__)

BRIDGING_SELECTION = namedtuple(
    'BRIDGING_SELECTION',
    ['any', 'high', 'low'])(0, 1, -1)
LOW_BRIDGING_THRESHOLD = 0.2 # 20%

def _select_bridging(df: pandas.DataFrame, bridging_selection):
    ''' Choose only fish in the df that match the selected bridging (high, low, all)
    '''
    if df is None or not df.size or bridging_selection == BRIDGING_SELECTION.any:
        return df
    full_metric_df = MetricLogger.metric_dataframe
    if 'glial_briding' not in full_metric_df.columns \
            or not full_metric_df['glial_bridging'].dropna().size:
        LOGGER.critical('Glial bridging is not available for these fish.')
    all_fish = set(FDM.get_available_fish_names())
    low_bridging = []
    high_bridging = []
    new_index_keys = []
    if 'fish' in df.columns:
        new_index_keys.append('fish')
        if 'assay' in df.columns:
            new_index_keys.append('assay')
    if new_index_keys:
        df = df.set_index(new_index_keys)
    for n in all_fish.intersection(df.index.get_level_values('fish').unique()):
        b = FDM.get_final_percent_briding(n, missing_val=None)
        if b is None:
            continue
        elif b <= LOW_BRIDGING_THRESHOLD:
            low_bridging.append(n)
        elif b > LOW_BRIDGING_THRESHOLD:
            high_bridging.append(n)
    if bridging_selection == BRIDGING_SELECTION.low:
        df = df.loc[numpy.in1d(df.index.get_level_values('fish'), low_bridging)]
    elif bridging_selection == BRIDGING_SELECTION.high:
        df = df.loc[numpy.in1d(df.index.get_level_values('fish'), high_bridging)]
    if new_index_keys:
        df = df.reset_index()
    return df

def get_metric_dataframe(
        group: str=None,
        assays: list=None,
        names_to_ignore: list=None,
        bridging_selection: int=BRIDGING_SELECTION.any) -> pandas.DataFrame:
    ''' Get the metric dataframe with certain restrictions.
    By default, unrestricted.

    Parameters
    ----------
    group: str, None
    assays: list, None
    names_to_ignore: list, None
    bridging_selection: int
    '''
    df = MetricLogger.metric_dataframe.copy().reset_index()
    if names_to_ignore:
        df = df[~numpy.in1d(df['fish'], names_to_ignore)]
    if assays is not None:
        df = df[numpy.in1d(df['assay'], assays)]
    if group is not None:
        df = df[df['group'] == group]
    df = _select_bridging(df, bridging_selection)
    return df

def _get_simple_waveform_dataframe(
        feature: str, estimator: str,
        experiment_name: str, groups: list,
        assays: list, names_to_ignore: list,
        is_control: bool) -> pandas.DataFrame:
    wf_calc = CruiseWaveformCalculator(
        experiment_name, data_utils.AnnotationTypes.predicted)
    flows = None
    dfs = []
    for g in groups:
        for a in assays:
            df = wf_calc.get_waveform_stats(g, a, flows, names_to_omit=names_to_ignore)
            if not df.size:
                continue
            df = df.loc[:, (
                df.columns.get_level_values('feature') == feature) \
                & (df.columns.get_level_values('stat') == estimator)]
            df.columns = df.columns.get_level_values('dim').values
            df.index = df.index.rename('fish')
            df = df.reset_index()
            df['group'] = g
            df['assay'] = a
            df['is_control'] = is_control
            dfs.append(df.set_index(['fish', 'assay', 'is_control', 'group']))
    if not dfs:
        return pandas.DataFrame()
    return pandas.concat(dfs)

def get_waveform_dataframe(
        feature: str,
        estimator: str='mean',
        group: str=None,
        assays: list=None,
        names_to_ignore: list=None,
        bridging_selection: int=BRIDGING_SELECTION.any):
    '''
    assays [-1, 1, 8] is generally preferred for plotting

    Parameters
    ----------
    feature: str
    estimator: str
        'mean' or 'std. 'mean' by default
    group: str, None
        All groups by default
    assays: list, None
        All assays by default
    names_to_ignore: int, None
    bridging_selection: int
        BRIDGING_SELECTION.any by default
    '''
    if assays is None:
        assays = FDM.get_available_assay_labels()
    if names_to_ignore is None:
        names_to_ignore = []
    experiment = config.experiment_name
    groups = [group]
    if group is None:
        groups = FDM.get_groups()
    return _select_bridging(
        _get_simple_waveform_dataframe(
            feature,
            estimator,
            experiment,
            groups,
            assays,
            names_to_ignore,
            is_control=False),
        bridging_selection)
