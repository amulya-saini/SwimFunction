''' Log metrics.
Access logged metrics as a full dataframe.
'''
import threading
import warnings
from typing import List
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
import pandas
import numpy

from swimfunction.data_access import data_utils
from swimfunction.recovery_metrics.metric_analyzers\
    .AbstractMetricAnalyzer import AbstractMetricAnalyzer
from swimfunction.global_config.config import config
from swimfunction import FileLocations

def _get_column_value_from_dataframe(df, fish_name, assay_label, column_name):
    val = None
    if column_name in df.columns and (fish_name, assay_label) in df.index:
        val = df.loc[(fish_name, assay_label), column_name]
    if val is not None and numpy.isnan(val):
        val = None
    return val

class _MetricLogger:
    __slots__ = []
    _dataframes = {}
    group_column = 'group'
    DATAFRAME_SETTER_LOCK = threading.Lock()

    @property
    def metric_dataframe(self) -> pandas.DataFrame:
        ''' Get the metric dataframe for the active experiment.
        DOES NOT RETURN A COPY
        '''
        key = config.experiment_name
        if key not in _MetricLogger._dataframes:
            _MetricLogger._dataframes[key] = self._read_metric_dataframe()
        return _MetricLogger._add_group_column(_MetricLogger._dataframes[key])

    @metric_dataframe.setter
    def metric_dataframe(self, df: pandas.DataFrame):
        ''' Sets the metric dataframe
        '''
        key = config.experiment_name
        _MetricLogger._dataframes[key] = df

    @property
    def normalized_metric_dataframe(self) -> pandas.DataFrame:
        ''' Get a COPY of the metric dataframe which is normalized
        '''
        df = self.metric_dataframe.copy()
        all_names = df.index.get_level_values(0)
        numeric_columns = [c for c in df.columns if is_numeric_dtype(df.loc[:, c])]
        scaler = StandardScaler()
        scaler.fit(df.loc[(all_names, -1), numeric_columns].values)
        df.loc[:, numeric_columns] = scaler.transform(df.loc[:, numeric_columns])
        return df

    @property
    def filename(self) -> str:
        ''' As a property, it can adapt to whichever CacheAccess context is currently in force.
        '''
        return FileLocations.get_csv_output_dir() / 'calculated_metrics.csv'

    def store_metric_value(
            self,
            fish_name: str,
            assay_label: int,
            column_name: str,
            value: float,
            save_to_file: bool=True):
        ''' Store one column value for one fish, one assay.
        NOTE: this should really only be called by a MetricCalculator.
        '''
        with _MetricLogger.DATAFRAME_SETTER_LOCK:
            self.metric_dataframe.loc[(fish_name, assay_label), column_name] = value
        if save_to_file:
            self.save_metric_dataframe()

    def log_analyzer_metrics(self, analyzer: AbstractMetricAnalyzer):
        from swimfunction.recovery_metrics.metric_analyzers.MetricCalculator import MetricCalculator
        MetricCalculator().calculate_and_log_analyzer_metrics(analyzer)

    def get_metric_values_for_plotting(
            self,
            fish_names: list,
            assay_labels: list,
            metric: str,
            normalize: bool,
            assays_to_strings: bool=True) -> pandas.DataFrame:
        scores = None
        if normalize:
            scores = self.normalized_metric_dataframe.loc[(fish_names, assay_labels), metric]
        else:
            scores = self.metric_dataframe.loc[(fish_names, assay_labels), metric]
        scores = pandas.DataFrame(scores)
        scores.reset_index(inplace=True)
        if assays_to_strings:
            scores.loc[scores['assay'] == -1, 'assay'] = 'Preinjury'
            scores['assay'] = [str(x) for x in scores['assay']]
        return scores

    def get_metric_value(self, fish_name: str, assay_label: int, column_name: str) -> float:
        ''' Get one column value for one fish, one assay.
        '''
        return _get_column_value_from_dataframe(
            self.metric_dataframe, fish_name, assay_label, column_name)

    def get_metric_values(
            self,
            fish_name: str,
            assay_label: int,
            column_names: List[str]) -> List[float]:
        ''' Get multiple column values for one fish, one assay.
        '''
        return [
            _get_column_value_from_dataframe(
                self.metric_dataframe, fish_name, assay_label, column_name)
            for column_name in column_names
        ]

    def has_analyzer_metrics(self, analyzer: AbstractMetricAnalyzer):
        ''' Whether metrics for the analyzer are already in the dataframe
        '''
        has_it = [self.has_metric_values(metric) for metric in analyzer.keys_to_printable]
        return numpy.all(has_it)

    def has_metric_values(self, column_name: str) -> bool:
        ''' Whether the column exists in the dataframe.
        '''
        return column_name in self.metric_dataframe.columns

    def _read_metric_dataframe(self) -> pandas.DataFrame:
        if not self.filename.exists():
            multiindex = pandas.MultiIndex(
                levels=[[],[]],
                codes=[[],[]],
                names=['fish', 'assay'])
            df = pandas.DataFrame(index=multiindex)
        else:
            df = pandas.read_csv(self.filename, header=[0])
            df = df.set_index(['fish', 'assay'])
        return df

    @staticmethod
    def _add_group_column(df) -> pandas.DataFrame:
        for name, assay in df.index:
            if _MetricLogger.group_column in df.columns \
                    and not pandas.isna(df.loc[(name, assay), _MetricLogger.group_column]):
                continue
            df.loc[(name, assay), _MetricLogger.group_column] = data_utils.fish_name_to_group(name)
        return df

    def save_metric_dataframe(self):
        key = config.experiment_name
        if key not in _MetricLogger._dataframes or not _MetricLogger._dataframes[key].size:
            warnings.warn(
                'The metric dataframe is empty. %s will not save an empty dataframe.',
                __name__)
            return
        with _MetricLogger.DATAFRAME_SETTER_LOCK:
            _MetricLogger._dataframes[key].to_csv(self.filename)

MetricLogger = _MetricLogger()
