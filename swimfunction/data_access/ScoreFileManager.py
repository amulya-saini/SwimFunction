''' Manages precalculated scores.
'''
from collections import defaultdict
from threading import Lock
import pathlib
import pandas
from swimfunction.global_config.config import config
from swimfunction import loggers

class ScoreFileManager:
    ''' Loads pre-assigned quantitative scores (quality, endurance, etc.).
    Expects format like this with the header row:
        fish,-1,1,2,3
        M1,5,3,2,1
        F12,5,4,2,4
    Header is formatted thus: first column is fish names, all following columns are assay labels.
    Each row after the header has a fish name and values calculated for the assay.
    Empty values are allowed ",,"
    '''
    ACCESS_LOCK = Lock()
    __slots__ = ['score_file_path', 'scores', 'was_warned_file_not_exist']
    def __init__(self, score_file_path: pathlib.Path):
        self.score_file_path = pathlib.Path(score_file_path)
        self.was_warned_file_not_exist = False
        self.scores = defaultdict(dict)

    def get_score(self, fish_name: str, assay_label: int, missing_value=None):
        ''' Get score from the file
        '''
        with ScoreFileManager.ACCESS_LOCK:
            experiment = config.experiment_name
            key = ScoreFileManager._get_score_key(fish_name, assay_label)
            if not self.scores[experiment]:
                try:
                    self.load_scores()
                except (pandas.errors.EmptyDataError, FileNotFoundError) as e:
                    logger = loggers.get_metric_calculation_logger(__name__)
                    logger.debug(e)
                    if not self.was_warned_file_not_exist:
                        self.was_warned_file_not_exist = True
                        logger.warning('The score file does not exist: %s \
                            If you want to plot these scores, \
                            you need to create this file.',
                            self.score_file_path.as_posix())
            if key not in self.scores[experiment]:
                return missing_value
            return self.scores[experiment][key]

    def load_scores(self):
        ''' Loads manually annotated qualitative scores.
        Expects format like this with the header row:
            fish,-1,1,2,3
            M1,5,3,2,1
            F12,5,4,2,4
        '''
        all_scores = defaultdict(lambda: None)
        fpath = self.score_file_path
        try:
            scores_df = pandas.read_csv(fpath, header=[0], index_col=[0])
        except (pandas.errors.EmptyDataError, FileNotFoundError) as e:
            logger = loggers.get_metric_calculation_logger(__name__)
            logger.debug('Cannot get metrics from empty file: %s', self.score_file_path.as_posix())
            logger.debug(e)
            return all_scores
        for name in scores_df.index:
            for assay in scores_df.columns:
                all_scores[self._get_score_key(name, assay)] = scores_df.loc[name, assay]
        self.scores[config.experiment_name] = all_scores
        return all_scores

    @staticmethod
    def _get_score_key(fish_name, assay):
        ''' Get a key based on the name and assay label

        Returns
        -------
        key: str
            concatenation of fish_name and assay
        '''
        return f'{fish_name}_{assay}wpi'
