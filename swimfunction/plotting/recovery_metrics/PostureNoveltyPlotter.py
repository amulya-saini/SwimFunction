''' Plots percent novelty
'''

from swimfunction.data_access.MetricLogger import MetricLogger
from swimfunction.recovery_metrics.metric_analyzers.PostureNoveltyAnalyzer \
    import PostureNoveltyAnalyzer
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.global_config.config import config
from matplotlib import pyplot as plt
import numpy

WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')

class PostureNoveltyPlotter:
    metric = 'posture_novelty'

    __slots__ = ['analyzer']

    def __init__(self):
        '''
        Parameters
        ----------
        control_group: str, None
            Default is None: all groups.
        control_assay: int
            Default is -1: preinjury/pretreatment.
        '''
        self.analyzer = PostureNoveltyAnalyzer()

    def choose_fish(self, group, fish_to_ignore):
        ''' Selects fish names in the group
        and not in the "fish_to_ignore" list
        '''
        if fish_to_ignore is None:
            fish_to_ignore = []
        names = []
        if group is None:
            names = FDM.get_available_fish_names()
        else:
            names = FDM.get_available_fish_names_by_group()[group]
        names = list(filter(lambda n: n not in fish_to_ignore, names))
        return names

    def get_metric_values(self, names: list, assays: list) -> dict:
        if not MetricLogger.has_analyzer_metrics(self.analyzer):
            MetricLogger.log_analyzer_metrics(self.analyzer)
        name_to_vals = {name: [] for name in names}
        for assay in assays:
            for name in names:
                val = MetricLogger.get_metric_value(name, assay, self.metric)
                if (val is not None) and (not numpy.isnan(val)):
                    name_to_vals[name].append(val)
                else:
                    name_to_vals[name].append(None)
        return name_to_vals

    def plot_posture_novelty_trends(self, group: str=None, assays: list=WPI_OPTIONS, fish_to_ignore: list=None):
        '''
        Parameters
        ----------
        group: str, None
            Default is None: all groups.
        assay: int
            Default is -1: preinjury/pretreatment.
        fish_to_ignore: list, None
            Which fish NOT to include. Default is empty list.
        '''
        names = self.choose_fish(group, fish_to_ignore)
        names_to_scores = self.get_metric_values(names, assays)
        fig, ax = plt.subplots()
        for name in names:
            ax.plot([a if a > 0 else 0 for a in assays], names_to_scores[name], linewidth=1, label=name)
        ax.set_title(f'{len(names)} fish', fontsize=4)
        ax.set_ylabel('% Novel Poses')
        ax.set_xlabel('Assay')
        ax.set_xticks(range(len(assays)))
        ax.set_xticklabels(assays)
        return fig

    def plot_posture_novelty(self, group: str=None, assays: list=WPI_OPTIONS, fish_to_ignore: list=None):
        '''
        Parameters
        ----------
        group: str, None
            Default is None: all groups.
        assay: int
            Default is -1: preinjury/pretreatment.
        fish_to_ignore: list, None
            Which fish NOT to include. Default is empty list.
        '''
        names = self.choose_fish(group, fish_to_ignore)
        names_to_scores = self.get_metric_values(names, assays)
        control_novelty_counts = [
            getattr(x, 'percent_novelty')
            for x in self.analyzer.null_novelty_counts.assay_to_counts.values()]
        novelty_counts = [control_novelty_counts]
        for i in range(len(assays)):
            novelty_counts.append(list(filter(
                lambda x: (x is not None),
                [
                    names_to_scores[name][i]
                    for name in names
                ])))
        fig, ax = plt.subplots()
        ax.set_title(f'{len(names)} fish', fontsize=4)
        ax.boxplot(novelty_counts, positions=range(len(assays) + 1), widths=0.3, showfliers=True)
        ax.set_ylabel('% Novel Poses')
        ax.set_xlabel('Assay')
        ax.set_xticks(range(len(WPI_OPTIONS)+1))
        ax.set_xticklabels(['Tracking Experiment\npreinjury', *WPI_OPTIONS])
        return fig
