''' Uses early recovery functional metrics
to predict 8 wpi structural metrics.
'''

from typing import Dict
import numpy
import pandas

from swimfunction.data_access.MetricLogger import MetricLogger
from swimfunction.data_access import data_utils

class RecoveryPredictor:
    ''' Predicts binary, relative recovery outcomes
    (well-recovered, poorly-recovered) according to metric ranks.
    '''

    # Lower compensation is good, so its values are inverted to make high ranks "better"
    PUBLISHED_PREDICTION_METRICS = {
        'rostral_compensation': False,
        'swim_distance': True
    }

    def __init__(self, metrics_for_prediction: Dict[str, bool]):
        '''
        Parameters
        ----------
        metrics_for_prediction: Dict[str, bool]
            keys: metric names (strings)
            values: whether high values of the metric are associated with better health
            (must have a bool for every metric in metrics_for_prediction)
        '''
        self.metrics_for_prediction = metrics_for_prediction

    @staticmethod
    def _ranks_to_per_group_predictions(ranks_df: pandas.DataFrame) -> numpy.ndarray:
        ''' Split ranks across the group's median.
        '''
        r = ranks_df['scores']
        index_groups = numpy.asarray([data_utils.fish_name_to_group(x) for x in r.index])
        predictions = numpy.zeros(r.shape[0], dtype=bool)
        for group in set(index_groups):
            mask = index_groups == group
            if numpy.any(mask):
                predictions[mask] = r[mask] > numpy.median(r[mask])
        return predictions

    @staticmethod
    def _get_metric_df() -> pandas.DataFrame:
        ''' Get a copy of the metric dataframe
        '''
        df = MetricLogger.metric_dataframe.sort_index().reset_index().copy()
        return df

    def _score_by_sum_of_ranks(self, df: pandas.DataFrame) -> pandas.DataFrame:
        '''
        Parameters
        ----------
        df : pandas.DataFrame
            Must contain these columns:
                'fish' (unique ids)
                'assay' (with 1 and 2 wpi)
                'amplitude1t5_suprema_from_preinj'
                'distance_raw'
        '''
        assays_for_prediction = [1, 2]
        df = df.loc[numpy.in1d(df['assay'], assays_for_prediction)].copy()
        for m, keep_sign in self.metrics_for_prediction.items():
            if not keep_sign:
                df.loc[:, m] = -1 * df.loc[:, m]
        return pandas.DataFrame(
            df.groupby('fish')[list(self.metrics_for_prediction.keys())]\
                .mean().rank(axis=0).sum(axis=1, skipna=True),
            columns=['scores']
        )

    def _make_predictions(self, df: pandas.DataFrame) -> pandas.DataFrame:
        ''' Predicts outcomes per-group.
        Ranks each group according to low rostral compensation
        and high distance swum, adds the ranks,
        and splits across the median.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain these columns:
                'fish' (unique ids)
                'assay' (must contain values 1 and 2)
                'amplitude1t5_suprema_from_preinj'
                'distance_raw'

        Returns
        -------
        pandas.DataFrame
            With two columns: (fish, will_recover_best)
        '''
        groups = set([
            data_utils.fish_name_to_group(f)
            for f in df['fish']])
        scores_df = None
        if len(groups) > 1:
            for group in groups:
                mask = [data_utils.fish_name_to_group(f) == group for f in df['fish']]
                df_by_group = df.loc[mask]
                sdf = self._make_predictions(df_by_group)
                scores_df = sdf if scores_df is None else pandas.concat((scores_df, sdf))
            return scores_df
        else:
            scores_df = self._score_by_sum_of_ranks(df)
            scores_df['will_recover_best'] = self._ranks_to_per_group_predictions(scores_df)
        return scores_df.loc[:, 'will_recover_best']

    def add_prediction_to_df(self, df: pandas.DataFrame) -> pandas.DataFrame:
        ''' Adds a column 'will_recover_best' to dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain these columns:
                'fish' (unique ids)
                'assay' (must contain values 1 and 2)
                'amplitude1t5_suprema_from_preinj'
                'distance_raw'

        Returns
        -------
        pandas.DataFrame
            Original DataFrame but now with column "will_recover_best"
        '''
        return pandas.merge(
            df,
            self._make_predictions(df),
            on='fish')

    def predict_by_sum_of_ranks(self) -> pandas.DataFrame:
        ''' Predicts outcomes per-group.
        Assuming you use PUBLISHED_PREDICTION_METRICS:
            Ranks each group according to low rostral compensation and
            high distance swum,
            adds the ranks,
            and splits across the median.

        To make predictions, you must have assay labels 1 and 2 for every fish name.
        You also must have calculated every metric in self.metrics_for_prediction.keys()

        Returns
        -------
        pandas.DataFrame
            With two columns: (fish, will_recover_best)
        '''
        return self._make_predictions(self._get_metric_df())

    @staticmethod
    def predict_as_published() -> pandas.DataFrame:
        ''' Split fish into groups predicted to have high and low
        regeneration outcomes. Prediction is performed exactly as published.
        '''
        return RecoveryPredictor(
            RecoveryPredictor.PUBLISHED_PREDICTION_METRICS
        ).predict_by_sum_of_ranks()
