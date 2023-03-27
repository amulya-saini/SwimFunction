''' Analyzer for all precalculated metrics.

This allows you to store precalculated metrics in csv files,
then import those csv files into the main metric csv file and DataFrame.

All csv files in the "precalculated" folder in the experiment cache directory
will be imported using this analyzer. The name of the csv file will become
the column name.

Expects format like this with the header row including the word "fish" followed by assay labels:
    fish,-1,1,2,3
    M1,5,3,2,1
    F12,5,4,2,4

The above file has two fish, M1 and F12, with values at assay labels -1, 1, 2, and 3.

Consider making the following files, if you have the data:
    perceived_quality.csv
    glial_bridging.csv
    proximal_axon_regrowth.csv
    distal_axon_regrowth.csv
    endurance.csv
'''
from typing import List
import pathlib
import numpy
from swimfunction.data_models.SwimAssay import SwimAssay
from swimfunction.recovery_metrics.metric_analyzers\
    .AbstractMetricAnalyzer import AbstractMetricAnalyzer
from swimfunction.data_access.ScoreFileManager import ScoreFileManager
from swimfunction import FileLocations

##### Human Assigned Scores
class PrecalculatedMetricsAnalyzer(AbstractMetricAnalyzer):
    ''' Get precalculated metric values.
    '''
    def __init__(self):
        super().__init__()
        self.keys_to_printable = {
            'perceived_quality': 'Quality',
            'endurance': 'Endurance',
            'glial_bridging': '% Glial Bridging',
            'proximal_axon_regrowth': '% Axon Regrowth\n(Proximal)',
            'distal_axon_regrowth': '% Axon Regrowth\n(Distal)'
        }

    def ensure_all_files_exist(self):
        ''' If an expected metric file does not exist,
        make it to prompt the user to add data.
        '''
        for k in self.keys_to_printable:
            fp = self.get_score_file_path(k)
            # If the file does not exist
            if not fp.exists():
                self.logger.info(
                    'File containing %s scores does not exist. Consider making it.', k)
                with open(fp, 'wt'):
                    pass # Create the file.

    def get_value(self, fish_name, assay_label, key, missing_value=numpy.nan):
        ''' Get a specific precalculated value for a fish at an assay.
        The key must match the column name in the csv file.
        '''
        return ScoreFileManager(self.get_score_file_path(key))\
            .get_score(fish_name, assay_label, missing_value)

    def get_score_file_path(self, key: str) -> pathlib.Path:
        ''' Given a key, get the associated file that should exist.
        '''
        return FileLocations.get_precalculated_metrics_dir() / f'{key}.csv'

    def available_keys(self) -> List[pathlib.Path]:
        ''' Keys (file names without extensions) that exist and can be read.
        '''
        return [
            fp.name.replace('.csv', '')
            for fp in FileLocations.get_precalculated_metrics_dir().glob('*.csv')
        ]

    def analyze_assay(self, swim_assay: SwimAssay) -> dict:
        self.ensure_all_files_exist()
        n = swim_assay.fish_name
        a = swim_assay.assay_label
        rv = {
            k: self.get_value(n, a, k) for k in self.available_keys()
        }
        return rv
