import pathlib
from matplotlib import pyplot as plt, ticker as mtick
from scipy import stats as spstats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import numpy

from swimfunction.data_access.assembled_dataframes import get_metric_dataframe
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from swimfunction.plotting.swim_specific_helpers \
    import metric_to_axis_label, metric_to_title
from swimfunction.plotting.recovery_metrics.MetricCorrelationPlotter \
    import choose_metric_columns
from swimfunction import loggers

# Figures
from swimfunction.plotting.constants import \
    FIGURE_WIDTH, FIGURE_ROW_HEIGHT, AXIS_LABELSIZE, \
    AXIS_SMALLTICKSIZE, LEGEND_FONTSIZE, TITLE_SIZE, \
    DOTSIZE

BRIDGING_METRIC = 'glial_bridging'
STRUCTURAL_METRICS = [
    BRIDGING_METRIC,
    'proximal_axon_regrowth',
    'distal_axon_regrowth'
]
ROSTRAL_COMP_METRIC = 'rostral_compensation'
DISTANCE_METRIC = 'swim_distance'
ACTIVITY_METRIC = 'activity'
TIME_METRIC = 'time_against_flow_10_20cms'
BURST_FREQ_METRIC = 'centroid_burst_freq_0cms'
POSE_BURST_FREQ_METRIC = 'pose_burst_freq_0cms'
Y_METRIC = 'mean_y_10_20cms'

def get_logger() -> loggers.logging.Logger:
    ''' Get the default plotting logger.
    '''
    return loggers.get_plotting_logger(__name__)

def linear_regression(x, y):
    ''' Simple linear regression
    '''
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    lr = LinearRegression().fit(x.reshape((-1, 1)), y.reshape((-1, 1)))
    slope, intercept = lr.coef_[0][0], lr.intercept_[0]
    return slope, intercept

def monotonic_regression_constrained_spline(x, y):
    ''' Monotonic regression spline
    See Gi_F and Glorfindel answers to the below:
    https://stats.stackexchange.com/questions/467126/monotonic-splines-in-python
    '''
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    order = x.argsort()
    x = x[order]
    y = y[order]
    monotonic_sign = numpy.sign(linear_regression(x, y)[0])
    reg = HistGradientBoostingRegressor(
        max_depth=1, monotonic_cst=[monotonic_sign],
        loss='squared_error', min_samples_leaf=2)
    reg.fit(x.reshape((-1, 1)), y)
    return x, reg.predict(x.reshape((-1, 1)))

def plot_monotonic_regression(ax, x, y, **line_kws):
    ''' Plots a monotonic regression line
    '''
    xx2, yy2 = monotonic_regression_constrained_spline(x, y)
    ax.plot(xx2, yy2, **line_kws)

class FunctionByStructurePlotter:
    ''' Plots metrics against cellular regeneration (structural) metrics
    '''

    color_kws = dict(zip(
        STRUCTURAL_METRICS,
        [
            dict(color='gray'),
            dict(color='plum'),
            dict(color='powderblue')
        ]))
    marker_kws = dict(zip(
        STRUCTURAL_METRICS,
        [
            dict(marker='+'),
            dict(marker='x'),
            dict(marker=5)
        ]))

    def __init__(self, savedir: pathlib.Path, group: str):
        self.savedir = savedir
        self.group = group

    @property
    def group_str(self):
        ''' Simple group string for filenames
        '''
        return f'{self.group}_' if self.group is not None else ''

    def plot_metric_against_structure(
            self,
            x_metric: str,
            y_structure_metrics: list,
            ax,
            axislabelsize=AXIS_LABELSIZE,
            axistitlesize=TITLE_SIZE,
            offsettextsize=LEGEND_FONTSIZE,
            xticks=None):
        ''' Plot single metric against cellular regeneration
        '''
        yvals = None
        metric_df = get_metric_dataframe(group=self.group, assays=[8])
        if not metric_df.size:
            return
        label_strs = [
            f'n = {metric_df.loc[:, (x_metric, STRUCTURAL_METRICS[0])].dropna().shape[0]}']
        for y_metric in y_structure_metrics:
            if y_metric == x_metric:
                continue
            ax.set_title(metric_to_title(x_metric), fontsize=axistitlesize)
            yvals, xvals = metric_df.loc[:, (y_metric, x_metric)].dropna().values.T.tolist()
            if len(yvals) == 0 or len(xvals) == 0:
                continue
            if x_metric in [ACTIVITY_METRIC, TIME_METRIC, 'posture_novelty']:
                xvals = [x*100 for x in xvals]
            rs, pval = spstats.spearmanr(xvals, yvals)
            scatter_kws = dict(zorder=2, s=DOTSIZE*2, edgecolors=None)
            line_kws = dict(
                zorder=1,
                linewidth=2,
                label=metric_to_title(y_metric),
                alpha=1 if pval < 0.05 else 0)
            ax.scatter(
                xvals, yvals, **scatter_kws,
                **self.marker_kws[y_metric], **self.color_kws[y_metric])
            plot_monotonic_regression(ax, xvals, yvals, **line_kws, **self.color_kws[y_metric])
            ax.set_xlabel(metric_to_axis_label(x_metric), fontsize=axislabelsize)
            ax.set_ylabel(
                '% Regeneration' if len(y_structure_metrics) > 1 \
                    else metric_to_axis_label(y_metric), fontsize=axislabelsize)
            self._set_x_axis_properties(ax, x_metric, offsettextsize)
            label_prefix = f'{metric_to_axis_label(y_metric)}: ' \
                if len(y_structure_metrics) > 1 else ''
            if pval < 0.05:
                label_strs.append(f'{label_prefix}$r_s$ = {numpy.round(rs, 2)}')
            else:
                label_strs.append(f'{label_prefix}p > 0.05')
            if xticks is not None:
                ax.xaxis.set_ticks(xticks)
            mpl_helpers.set_axis_tick_params(ax, labelsize=AXIS_SMALLTICKSIZE)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol=None))
        if yvals is not None:
            ax.set_ylim(0, max(1, max(yvals)))
            text = '\n'.join(label_strs)
            get_logger().info(
                '%s %s vs %s %s',
                self.group,
                x_metric,
                '[' + ','.join(y_structure_metrics) + ']',
                text)

    def plot_function_by_structural_correlations(self, fprefix, structural_metrics):
        ''' Plot all metrics versus cellular regeneration
        '''
        metric_df = get_metric_dataframe(group=self.group, assays=[8])
        metrics = choose_metric_columns(metric_df)
        metrics = [m for m in metrics if metric_df[m].dropna().size]
        ncols = 2
        nrows = int(numpy.ceil(len(metrics) / ncols))
        if nrows == 0:
            return
        fig, axs = mpl_helpers.get_subplots_absolute_sized_axes(
            nrows, ncols, (FIGURE_WIDTH, FIGURE_ROW_HEIGHT*nrows), 2.5, 1.5, sharey=True)
        flat_axs = axs.flatten()
        for ax, x_metric in zip(flat_axs, metrics):
            self.plot_metric_against_structure(x_metric, structural_metrics, ax)
        fname = f'{fprefix}{self.group_str}pairwise.png'
        mpl_helpers.save_fig(fig, self.savedir / fname)
        plt.close(fig)

    def plot_main_function(self):
        ''' Plot one metric from each major class versus cellular regeneration
        '''
        fig, axs = plt.subplots(1, 4, figsize=(FIGURE_WIDTH, 1.75))
        for ax, metric, xtickstep, xtickdefaultmax in zip(
                axs,
                (DISTANCE_METRIC,
                    POSE_BURST_FREQ_METRIC,
                    ROSTRAL_COMP_METRIC,
                    BRIDGING_METRIC),
                (1e5, 30, 0.03, 0.25),
                (5e5, 120, 0.12, 1)):
            metric_df = get_metric_dataframe(group=self.group, assays=[8])
            if not metric_df.size:
                continue
            xmaxval = metric_df.loc[:, (metric)].dropna().max()
            xticks = numpy.arange(0, max(xtickdefaultmax, xmaxval), step=xtickstep)
            y_structure_metrics = [BRIDGING_METRIC]
            if metric == BRIDGING_METRIC:
                y_structure_metrics = STRUCTURAL_METRICS
            self.plot_metric_against_structure(
                metric, y_structure_metrics, ax,
                axislabelsize=8, axistitlesize=10, offsettextsize=8,
                xticks=xticks)
        fig.tight_layout()
        fname = f'{self.group_str}metric_of_each_type.png'
        mpl_helpers.save_fig(fig, self.savedir / fname)
        plt.close(fig)

    def _set_x_axis_properties(self, ax, x_metric, offsettextsize):
        if x_metric == DISTANCE_METRIC:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 5))
            ax.xaxis.offsetText.set_fontsize(offsettextsize)
        else:
            ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        if x_metric == Y_METRIC:
            ax.xaxis.set_ticks(range(0, 1700, 400))
        if x_metric in STRUCTURAL_METRICS:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, symbol=None))
        if x_metric in [ACTIVITY_METRIC, TIME_METRIC, 'posture_novelty']:
            ax.set_xlim(0, 100)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(symbol=None))
            ax.xaxis.set_ticks(range(0, 101, 20))
        if x_metric == ROSTRAL_COMP_METRIC:
            ax.set_xlim(0, get_metric_dataframe(assays=[8])[x_metric].max())
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, -2))
            ax.xaxis.offsetText.set_fontsize(offsettextsize)

def main(savedir: pathlib.Path):
    ''' Plot with groups together and separated
    '''
    for group in ['M', 'F', None]:
        get_logger().info('Group: %s', group)
        plotter = FunctionByStructurePlotter(savedir, group)
        plotter.plot_main_function()
        for sm in STRUCTURAL_METRICS:
            plotter.plot_function_by_structural_correlations(f'{sm}_', [sm])
        plotter.plot_function_by_structural_correlations('all_', STRUCTURAL_METRICS)
