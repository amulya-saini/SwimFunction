import functools
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy
import pandas
from scipy.stats.stats import spearmanr
from sklearn.cluster import k_means
from tqdm import tqdm

from swimfunction.data_access import PoseAccess
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access.MetricLogger import MetricLogger
from swimfunction.data_models.Fish import Fish
from swimfunction.pose_processing import pose_filters
from swimfunction.recovery_metrics.metric_analyzers import ScoliosisAnalyzer
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from swimfunction.video_processing.extract_frames import Extractor

ABNORMAL_THRESHOLD = 0.2

def get_correlation_matrix_labels(df: pandas.DataFrame, correlation_fn=spearmanr):
    '''
    Parameters
    -------
    data: dict
        From fish name to tuple of parallel lists: assay labels and scoliosis scores
    '''
    df.set_index(['fish', 'assay'], inplace=True)
    assays = sorted(df.index.get_level_values(1).unique())
    r_matrix = numpy.full((len(assays), len(assays)), numpy.nan)
    p_matrix = r_matrix.copy()
    for i, a1 in enumerate(assays):
        for j, a2 in enumerate(assays):
            names = pandas.unique([n for n in df.index.get_level_values(0) if (n, a1) in df.index and (n, a2) in df.index])
            y1 = df.loc[pandas.MultiIndex.from_product((names, [a1]))].sort_index().values.flatten()
            y2 = df.loc[pandas.MultiIndex.from_product((names, [a2]))].sort_index().values.flatten()
            r, p = correlation_fn(y1, y2)
            r_matrix[i, j] = r
            p_matrix[i, j] = p
    return r_matrix, p_matrix, assays, assays

def get_metric_data_for_heatmap(
        names_to_ignore: list=None,
        column_name: str=ScoliosisAnalyzer.METRIC_COLUMN_NAME,
        normalize: bool=False,
        assays_to_strings: bool=False) -> pandas.DataFrame:
    '''
    Parameters
    ----------
    names_to_ignore: list, None
        Which fish NOT to include. Default is empty list.
    column_name: str
    normalize: bool
        Whether to normalize to preinjury levels.

    Returns
    -------
    dict
        From fish name to tuple of parallel lists: assay labels and metric (e.g., scoliosis) scores
    '''
    if names_to_ignore is None:
        names_to_ignore = []
    names = [n for n in FDM.get_available_fish_names() if n not in names_to_ignore]
    analyzer = ScoliosisAnalyzer.ScoliosisAnalyzer()
    if not MetricLogger.has_analyzer_metrics(analyzer):
        MetricLogger.log_analyzer_metrics(analyzer)
    return MetricLogger.get_metric_values_for_plotting(
        names,
        PoseAccess.WPI_OPTIONS,
        column_name,
        normalize,
        assays_to_strings)

def plot_weekly_correlation(
        names_to_ignore=None,
        column_name=ScoliosisAnalyzer.METRIC_COLUMN_NAME,
        title=''):
    ''' Plots heatmap: weekly scoliosis pairwise correlation matrix.
    '''
    if names_to_ignore is None:
        names_to_ignore = []
    fig, ax = plt.subplots()
    mpl_helpers.plot_heatmap(
        *get_correlation_matrix_labels(get_metric_data_for_heatmap(names_to_ignore, column_name)),
        title=title,
        ax=ax)
    return fig

def set_boxplot_colors(bp, show_fliers: bool):
    '''Set colors for the boxplot according to group.
    Parameters
    ----------
    bp : [type]
        Boxplots as returned by pyplot's boxplot function
    show_fliers : bool
        Whether to show the outliers
    '''
    ngroups = len(bp['boxes'])
    group_colors = mpl_helpers.get_cmap_values('tab10', ngroups)
    for group in range(ngroups):
        plt.setp(bp['boxes'][group], color=group_colors[group])
        plt.setp(bp['caps'][group * 2], color=group_colors[group])
        plt.setp(bp['caps'][group * 2 + 1], color=group_colors[group])
        plt.setp(bp['whiskers'][group * 2], color=group_colors[group])
        plt.setp(bp['whiskers'][group * 2 + 1], color=group_colors[group])
        plt.setp(bp['medians'][group], color=group_colors[group])
        if show_fliers:
            plt.setp(bp['fliers'][group], markeredgecolor=group_colors[group])

def make_boxplot(ax: plt.Axes, feature_dict: dict, show_fliers: bool):
    '''Creates a boxplot with appropriate colors for groups.
    Parameters
    ----------
    ax : plt.Axes
    feature_dict : dict[dict[numpy.ndarray]]
        Maps from wpi -> group -> values
    show_fliers : bool
        Whether to show outliers.
    '''
    groups = sorted(list(next(iter(feature_dict.values())).keys()))
    labels = sorted(list(feature_dict.keys()))
    nlabels = len(feature_dict)
    boxes_width = 1.5 * len(groups)
    for counter, label in enumerate(labels):
        group_data = [feature_dict[label][g] for g in groups]
        if not group_data:
            continue
        positions = [i + (boxes_width * counter) for i in range(len(group_data))]
        set_boxplot_colors(
            ax.boxplot(
                group_data,
                positions=positions,
                widths=0.6,
                showfliers=show_fliers),
            show_fliers)
    # set axes limits and labels
    ax.set_xlim(-1, nlabels*boxes_width)
    ax.set_xticks((numpy.arange(nlabels) * boxes_width) + 2)
    ax.set_xticklabels([f'{label}' if label != -1 else 'Preinj' for label in labels])
    # draw temporary lines and use them to create a legend
    lines = []
    for i in range(len(groups)):
        color = plt.get_cmap('tab10')(i/10)
        line, = plt.plot([1, 1], '-', color=color)
        lines.append(line)
    ax.legend(lines, groups)
    for line in lines:
        line.set_visible(False)

class ScoliosisPlotter:
    ''' Plots scoliosis data
    '''

    __slots__ = ['_scoliosis_data', 'names_to_ignore', 'normalize']
    def __init__(self, names_to_ignore=None, normalize: bool = False):
        if names_to_ignore is None:
            names_to_ignore = []
        self._scoliosis_data = {True: None, False: None}
        self.names_to_ignore = names_to_ignore
        self.normalize = normalize

    @property
    def scoliosis_data(self):
        fix_lists = lambda arr: ([int(x) for x in arr[0]], arr[1])
        if self._scoliosis_data[self.normalize] is None:
            df = get_metric_data_for_heatmap(self.names_to_ignore, normalize=self.normalize)
            self._scoliosis_data[self.normalize] = {
                n: fix_lists(df[df.fish == n].iloc[:, 1:].values.T.tolist())
                for n in df.fish
            }
        return self._scoliosis_data[self.normalize]

    def plot_representative_fish(self, num_to_plot=10, fig_to_use=None):
        ''' Demonstrates scoliosis score by
        plotting a series of frames from low to high scoliosis,
        along with their labeled scoliosis.
        '''
        names = []
        assays = []
        labels = []
        fish_names = [n for n in FDM.get_available_fish_names() if n not in self.names_to_ignore]
        for name in tqdm(fish_names):
            for assay in FDM.get_available_assay_labels(name):
                names.append(name)
                assays.append(assay)
                labels.append(FDM.get_scoliosis(name, assay))
        return ScoliosisPlotter._plot_labeled_frames(
            *ScoliosisPlotter._get_descriptive_subset(
                names, assays, labels, num_to_plot), max(labels), fig_to_use=fig_to_use)

    def plot_by_order(self, week_to_check: int, ax=None, cmap='plasma', **line_kws):
        ''' Plot trendlines for fish, colored by order at week_to_check.

        Parameters
        ----------
        week_to_check: int
            Which week to use for ordered coloring scoliosis.
        '''
        if ax is None:
            _fig, ax = plt.subplots()
        ax.set_title(f'scoliosis < {ABNORMAL_THRESHOLD} at {week_to_check} wpi')
        names = []
        week_vals = []
        for name, (x, y) in self.scoliosis_data.items():
            names.append(name)
            week_vals.append(
                y[x.index(week_to_check)]
                    if week_to_check in x and y[x.index(week_to_check)] is not None
                else numpy.nan)
        order = list(reversed(numpy.argsort(week_vals)))
        names = numpy.take(names, order)
        week_vals = numpy.take(week_vals, order)
        order_counter = 0
        for name, week_val in zip(names, week_vals):
            nongray = (~numpy.isnan(week_vals)).sum()
            x, y = self.scoliosis_data[name]
            color = plt.get_cmap(cmap)(order_counter / nongray)
            if week_val is None or numpy.isnan(week_val):
                color = 'gray'
            else:
                order_counter += 1
            ax.plot(self._set_uninj_zero(x), y, label=name, color=color, alpha=0.7, **line_kws)
        ax.set_xticks(range(9))
        ax.set_xticklabels('Pre,1,2,3,4,5,6,7,8'.split(','))
        return ax

    def plot_normal_and_abnormal_fish(self, week_to_check: int):
        ''' Plot trendlines for well-recovered and poorly-recovered
        fish on separate plots.

        Parameters
        ----------
        week_to_check: int
            Which week to check for abnormal scoliosis.
        '''
        fig, axs = plt.subplots(1, 2)
        self._plot_abnormals(axs[0], week_to_check)
        self._plot_normals(axs[1], week_to_check)
        lims = (min(axs[0].get_ylim()[0], axs[1].get_ylim()[0]),
            max(axs[0].get_ylim()[1], axs[1].get_ylim()[1]))
        axs[0].set_ylim(*lims)
        axs[1].set_ylim(*lims)
        return fig

    def plot_trendlines(self):
        ''' Plots each fish's scoliosis at each week post-injury.
        '''
        fig, ax = plt.subplots()
        for name, (x, y) in self.scoliosis_data.items():
            ax.plot(self._set_uninj_zero(x), y, '', label=name)
        return fig

    def plot_boxplot(self):
        fig, ax = plt.subplots()
        labels = set(functools.reduce(lambda l1, l2: l1 + l2, [a[0] for a in self.scoliosis_data.values()]))
        groups = FDM.get_groups()
        scores_dict = {
            label: {
                group: [y[x.index(label)] for name, (x, y) in self.scoliosis_data.items() if label in x and group in name]
                for group in groups
            }
            for label in labels}
        make_boxplot(ax, scores_dict, show_fliers=True)
        return fig

    def plot_weekly_correlation(self):
        return plot_weekly_correlation(
            self.names_to_ignore,
            column_name=ScoliosisAnalyzer.METRIC_COLUMN_NAME,
            title='Lateral Scoliosis Weekly Correlations')

    def _plot_abnormals(self, ax, week_to_check):
        ax.set_title(f'scoliosis >= {ABNORMAL_THRESHOLD} at {week_to_check} wpi')
        for name, (x, y) in self.scoliosis_data.items():
            if week_to_check in x and y[x.index(week_to_check)] >= ABNORMAL_THRESHOLD:
                ax.plot(self._set_uninj_zero(x), y, label=name)

    def _plot_normals(self, ax, week_to_check):
        ax.set_title(f'scoliosis < {ABNORMAL_THRESHOLD} at {week_to_check} wpi')
        for name, (x, y) in self.scoliosis_data.items():
            if week_to_check in x and y[x.index(week_to_check)] < ABNORMAL_THRESHOLD:
                ax.plot(self._set_uninj_zero(x), y, label=name)

    @staticmethod
    def _plot_labeled_frames(names, assays, labels, frame_nums, label_max, fig_to_use=None):
        ncols = len(labels)
        fig = fig_to_use if fig_to_use is not None else plt.figure(figsize=(11, 1.6))
        gs = mpl.gridspec.GridSpec(1, ncols)
        for i, (n, a, l, fn) in enumerate(zip(names, assays, labels, frame_nums)):
            ax = fig.add_subplot(gs[0, i])
            color = plt.get_cmap('viridis')(l / label_max)
            ax.set_title(f'{l:.2f}', fontsize=10, color=color)
            ax.imshow(ScoliosisPlotter._get_frame(n, a, fn))
            ScoliosisPlotter._setup_img_axis(ax, color)
            print(f'{i} {n} {a} frame {fn} scoliosis: {l}')
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=plt.Normalize(0, max(labels)), cmap='viridis'),
            ax=fig.get_axes(),
            ticks=numpy.around(labels, 2),
            location='bottom',
            aspect=110,
            shrink=1.1,
            pad=0)
        if fig_to_use is None:
            fig.tight_layout()
        return fig

    @staticmethod
    def _get_descriptive_subset(names, assays, labels, num_to_plot):
        take = lambda arr, idx: [arr[i] for i in idx]
        # Sort all arrays by the label
        labels = numpy.asarray(labels)
        order = labels.argsort()
        names, assays, labels = (
            take(names, order),
            take(assays, order),
            take(labels, order)
        )

        labels = numpy.asarray(labels)
        idx = numpy.arange(len(labels), dtype=int)
        clusters = k_means(labels.reshape((-1, 1)), n_clusters=num_to_plot-2)[1]
        rv_names = [names[0], names[-1]]
        rv_assays = [assays[0], assays[-1]]
        rv_labels = [labels[0], labels[-1]]
        for c in numpy.unique(clusters):
            i = int(numpy.median(idx[clusters == c]))
            rv_names.append(names[i])
            rv_assays.append(assays[i])
            rv_labels.append(labels[i])
        rv_frames = [ScoliosisPlotter._get_representative_frame_num(name, assay) for name, assay in zip(rv_names, rv_assays)]
        # Sort result by labels
        rv_labels = numpy.asarray(rv_labels)
        order = rv_labels.argsort()
        rv_names, rv_assays, rv_labels, rv_frames = (
            take(rv_names, order),
            take(rv_assays, order),
            take(rv_labels, order),
            take(rv_frames, order)
        )
        return rv_names, rv_assays, rv_labels, rv_frames

    @staticmethod
    def _get_frame(fish_name, assay_label, frame_num):
        return Extractor(
            fish_name,
            assay_label,
            frame_nums_are_full_video=True
        ).extract_frame(frame_num)

    @staticmethod
    def _get_representative_frame_num(name, assay):
        poses = PoseAccess.get_feature_from_assay(
                Fish(name).load()[assay],
                'smoothed_angles',
                pose_filters.BASIC_FILTERS + [pose_filters.filter_by_distance_from_edges],
                keep_shape=True
            )
        not_nan = numpy.all(~numpy.isnan(poses), axis=1)
        min_i = abs(poses[not_nan] - poses[not_nan].mean(axis=0)).sum(axis=1).argmin()
        frame_num = numpy.where(not_nan)[0][min_i]
        return frame_num

    @staticmethod
    def _setup_img_axis(ax, color):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_color(color)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    @staticmethod
    def _set_uninj_zero(x):
        return [i if i != -1 else 0 for i in x]

if __name__ == '__main__':
    plotter = ScoliosisPlotter()
    plotter.plot_weekly_correlation()
    plotter.plot_trendlines()
    plotter.plot_by_order(8)
    # plotter.plot_normal_and_abnormal_fish(2)
    # plot_weekly_correlation(
    #     names_to_ignore=plotter.names_to_ignore,
    #     column_name=ScoliosisAnalyzer.METRIC_COLUMN_NAME,
    #     title='Body Curvature Weekly Correlations')
    # plotter.plot_representative_fish()
    plt.show()
