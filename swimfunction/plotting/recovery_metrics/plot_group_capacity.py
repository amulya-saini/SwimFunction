''' Plots capacity metrics for groups of fish
'''
import pathlib
import numpy
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction import FileLocations
from swimfunction.data_access.assembled_dataframes import get_metric_dataframe
from swimfunction.plotting.matplotlib_helpers \
    import shrink_lines_and_dots, save_fig, get_cmap_values
from swimfunction.plotting.swim_specific_helpers \
    import assay_tick_labels, metric_to_axis_label, metric_to_title, \
    make_assay_barplot, make_stripplot
from swimfunction.plotting.constants import FIGURE_WIDTH, \
    FIGURE_ROW_HEIGHT, AXIS_LABELSIZE, AXIS_TICKSIZE, AXIS_SMALLTICKSIZE, \
    LEGEND_FONTSIZE, TITLE_SIZE
from swimfunction.global_config.config import config

WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')

def _set_summary_ax_properties(ax, column_name):
    ''' Sets axis properties.
    '''
    ax.set_xticks(numpy.arange(len(WPI_OPTIONS)))
    ax.set_xticklabels([
        f'{label}' if label != -1 else 'Preinj' for label in WPI_OPTIONS])
    if ax.get_legend() is not None:
        plt.setp(ax.get_legend().get_title(), fontsize=5)
        plt.setp(ax.get_legend().get_texts(), fontsize=4)
        for lh in ax.get_legend().legendHandles:
            lh._sizes = [2]
    ax.set_title(metric_to_title(column_name), fontsize=6)
    ax.set_xlabel('Weeks Post-Injury')
    ax.set_ylim(0, None)
    if column_name in ['distance_raw', 'distance_fa']:
        ax.set_ylabel('px')
    if column_name in ['activity_raw', 'activity_fa']:
        ax.set_ylabel('% frames')
        ax.set_ylim(0, 100)
    if column_name in ['time_against_raw', 'time_against_fa']:
        ax.set_ylabel('% frames')
        ax.set_ylim(0, 100)
    if column_name == 'quality':
        ax.set_ylabel('%')
        ax.set_ylim(0, 1)
    if column_name == 'x_mean':
        ax.set_ylabel('px')
    if column_name == 'y_mean':
        ax.set_ylabel('px')
    if column_name == 'endurance':
        ax.set_ylabel('sec')

def plot_quality_as_percent_bars(quality_ax, df):
    ''' Plots perceived quality as percent bars
    '''
    colors = dict(zip(range(1, 6), get_cmap_values('viridis', 5)))
    nfish = df[df['assay'] == -1]['fish'].size
    if nfish == 0:
        return
    for i, assay in enumerate(sorted(df['assay'].unique())):
        bottom = 0
        for q in reversed(range(1, 6)):
            count = (df[df['assay'] == assay]['perceived_quality'].values == q).sum()
            percent = count / nfish
            quality_ax.bar(
                x=max(0, i),
                height=percent,
                width=0.5,
                bottom=bottom,
                color=colors[q],
                linewidth=0)
            bottom += percent
    quality_ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, symbol=None))
    assay_tick_labels(quality_ax)
    return quality_ax

def plot_capacity_summary(
        axs, column_names,
        as_violin_and_swarm: bool=True,
        group: str=None):
    '''
    Parameters
    ----------
    axs : pyplot.Axes
        Must be enough axes for the number of columns, if provided.
    column_names: list
    as_violin_and_swarm: bool
    group: str, None
    '''
    df = get_metric_dataframe(group=group)
    df['name'] = df['fish']
    if isinstance(axs, numpy.ndarray):
        axs = axs.flatten()
    ax_i = 0
    for column_name in column_names:
        ax = axs[ax_i]
        if column_name == 'perceived_quality':
            plot_quality_as_percent_bars(axs[ax_i], df)
        elif as_violin_and_swarm:
            make_stripplot(ax, df, column_name, include_violin=True)
        else:
            make_assay_barplot(df, group, column_name, ax)
        _set_summary_ax_properties(ax, column_name)
        ax_i += 1

def plot_capacity_metrics_distribution_and_trends(savedir: pathlib.Path, group: str):
    ''' Plot similar to Figure 1
    '''
    group_str = f'{group}_' if group is not None else ''
    outpath = savedir / f'{group_str}capacity_metric_trends.png'
    metrics = [
        'swim_distance',
        'activity',
        'time_against_flow_10_20cms',
        'mean_y_10_20cms',
        'centroid_burst_freq_0cms',
        'perceived_quality'
    ]
    m_titles = [metric_to_title(m) for m in metrics]
    m_labels = [metric_to_axis_label(m) for m in metrics]
    line_fig, line_axs = plt.subplots(2, 3, figsize=(FIGURE_WIDTH, FIGURE_ROW_HEIGHT*2))
    line_axs = line_axs.flatten()
    plot_capacity_summary(
        line_axs,
        metrics,
        as_violin_and_swarm=False,
        group=group)
    for ax in (line_axs[1], line_axs[2]):
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, symbol=None))
        ax.set_ylim(0, 1)
    for ax, title, label in zip(
            line_axs,
            m_titles + m_titles,
            m_labels + m_labels):
        ax.set_ylabel(label, fontsize=AXIS_LABELSIZE)
        ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
        assay_tick_labels(ax)
        shrink_lines_and_dots(ax, linewidth=2)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.tick_params(axis='x', labelsize=AXIS_TICKSIZE)
        ax.tick_params(axis='y', labelsize=AXIS_SMALLTICKSIZE)

    line_axs[0].ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
    line_axs[0].yaxis.offsetText.set_fontsize(LEGEND_FONTSIZE)

    line_fig.tight_layout(h_pad=6, w_pad=4)

    save_fig(line_fig, outpath)
    plt.close(line_fig)

def main():
    ''' Plot groups separately and altogether.
    '''
    savedir = FileLocations.get_capacity_metrics_plot_dir()
    for group in FDM.get_groups() + [None]:
        plot_capacity_metrics_distribution_and_trends(savedir, group)
