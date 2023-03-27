from swimfunction.data_models.SwimAssay import SwimAssay
from swimfunction import loggers

class AbstractMetricAnalyzer:
    ''' Holds a dict from metric key (for pandas column name)
    to a printable name (for plots and such)
    '''
    __slots__ = ['keys_to_printable', 'logger']
    def __init__(self):
        self.logger = loggers.get_metric_calculation_logger(self.__class__.__name__)
        self.keys_to_printable = {}

    def keys_to_self(self) -> dict:
        return {key: self for key in self.keys_to_printable.keys()}

    def analyze_assay(self, swim_assay: SwimAssay) -> dict:
        ''' Calculates the metric, and related required values, for the assay.
        NOTE: this function should NOT affect the calculated_metrics.csv logfile.

        Returns
        -------
        dict
            metric name to value (all relevant measurements for this metric analyzer)
        '''
        raise NotImplementedError()
