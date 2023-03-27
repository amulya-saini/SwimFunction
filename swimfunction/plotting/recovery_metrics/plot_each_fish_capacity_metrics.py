''' Plots capacity metrics for individual fish
'''
import pathlib
import numpy
from matplotlib import pyplot as plt

from swimfunction.data_access import PoseAccess
from swimfunction.data_access.MetricLogger import MetricLogger
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.pose_processing import pose_filters
from swimfunction.recovery_metrics.metric_analyzers\
    .SwimCapacityProfiler import SwimCapacityProfiler
from swimfunction.recovery_metrics.metric_analyzers\
    .SwimCapacityAnalyzer import SwimCapacityAnalyzer
from swimfunction.plotting import swim_specific_helpers as sph
from swimfunction.global_config.config import config
from swimfunction.context_managers.WorkerSwarm import WorkerSwarm

from swimfunction import FileLocations
from swimfunction import loggers
from swimfunction import progress

WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')
ENFORCE_SAME_LENGTH = True

METRICS_HEADER_PARTS = [
    'assay',
    'swim_distance',
    'activity',
    'time_against_flow_10_20cms',
    'y_mean']

COLUMNS_TO_TITLE = SwimCapacityAnalyzer().keys_to_printable

CSV_HEADER = ','.join(['fish'] + METRICS_HEADER_PARTS)

def plot_metric(ax, metric_values, labels, metric, as_percents=False):
    ''' Plot metric values
    '''
    coef = 1
    if as_percents:
        coef = 100
    metric_values = numpy.asarray(metric_values)
    ax.set_title(sph.metric_to_title(metric), fontsize=6)
    ax.set_xlabel('Assay', fontsize=5)
    ax.set_ylabel(sph.metric_to_axis_label(metric), fontsize=5)
    new_labels = labels.copy()
    new_labels[new_labels == -1] = 0
    ax.set_xticks(range(len(WPI_OPTIONS)))
    ax.set_xticklabels(['Pre-injury'] + [str(x) for x in WPI_OPTIONS if x > 0])
    ax.set_xlim(-.5, len(WPI_OPTIONS))
    ax.plot(new_labels, metric_values * coef, c='black', linewidth=1)
    for i, assay_label in enumerate(new_labels):
        ax.scatter(assay_label, metric_values[i] * coef, label=assay_label)
    return metric_values

def plot_y_position(ax, metric_profiler: SwimCapacityProfiler):
    ''' Plot y position during the assay
    '''
    locations = metric_profiler.y_position()
    max_loc = metric_profiler.get_assay_properties(
        metric_profiler.fish_names[0],
        metric_profiler.labels[0]
    ).height
    ax.set_title('Y-position', fontsize=6)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Y')
    ax.vlines(x=70*60*5, ymin=0, ymax=max_loc, color='black', linestyles='dotted', linewidth=2)
    ax.vlines(x=70*60*10, ymin=0, ymax=max_loc, color='black', linestyles='dotted', linewidth=2)
    for i, locs in enumerate(locations):
        ax.plot(range(len(locs)), locs, label=metric_profiler.labels[i], linewidth=0.5)
    ax.set_xlim(0, 70 * 60 * 15.1)
    ax.legend(loc='right')
    return locations

def plot_location_heatmaps(output_dir, metric_profiler: SwimCapacityProfiler):
    ''' Saves location heatmaps for the fish
    '''
    fish_name = metric_profiler.fish_names[0]
    fig, axs = plt.subplots(1, len(WPI_OPTIONS), figsize=(9, 5))
    fig.suptitle('Location Heatmaps', fontsize=8)
    heatmaps = metric_profiler.location_heatmap()
    for i, assay_label in enumerate(WPI_OPTIONS):
        ax = axs[i]
        if assay_label not in metric_profiler.labels:
            ax.set_visible(False)
            continue
        ax.set_title(f'{assay_label}wpi', fontsize=6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.imshow(heatmaps[numpy.where(metric_profiler.labels == assay_label)[0][0]])
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / f'{fish_name}_location_heatmaps.png')
        plt.close(fig)

def plot_capacity_metrics(
        output_dir,
        metric_profiler: SwimCapacityProfiler):
    ''' Basic capacity metrics plotted for a single fish
    '''
    metrics = {
        'swim_distance': False,
        'activity': True,
        'time_against_flow_10_20cms': True,
        'perceived_quality': False
    }
    fish_name = metric_profiler.fish_names[0]
    fig = plt.figure(figsize=(9, 5))
    df = MetricLogger.metric_dataframe.loc[
        fish_name,
        metrics].sort_index()
    labels = df.index.values
    axs = [
        plt.subplot2grid(
            (2, 3), (0, 0),
            colspan=1, fig=fig),
        plt.subplot2grid(
            (2, 3), (0, 1),
            colspan=1, fig=fig),
        plt.subplot2grid(
            (2, 3), (0, 2),
            colspan=1, fig=fig)
    ]
    for i, (metric, as_percent) in enumerate({
        'swim_distance': False,
        'activity': True,
        'time_against_flow_10_20cms': True}.items()):
        plot_metric(
            axs[i],
            df[metric],
            labels,
            metric,
            as_percents=as_percent)

    plot_y_position(
        plt.subplot2grid(
            (2, 3), (1, 0),
            colspan=4, fig=fig), metric_profiler)
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / f'{fish_name}_metrics.png')
        plt.close(fig)

def get_metrics_profiler(fish_name, pose_filters, flow_speed=None):
    '''
    Returns
    -------
    metrics_result
    metric_profiler
    '''
    if pose_filters is None:
        pose_filters = []
    pose_arrays, names, labels = PoseAccess.get_feature(
        [fish_name],
        WPI_OPTIONS,
        'smoothed_coordinates',
        filters=pose_filters, keep_shape=True)
    # Enforce sorted order of labels
    order = numpy.argsort(labels)
    labels = numpy.asarray(labels)[order]
    pose_arrays = [pose_arrays[i] for i in order]
    # Sanity check
    for name in names:
        assert name == fish_name, \
            f'Got features from a fish ({name}) that is not {fish_name}'
    # Plot stuff
    metric_profiler = SwimCapacityProfiler(
        pose_arrays,
        enforce_same_length=ENFORCE_SAME_LENGTH,
        fish_names=names,
        labels=labels,
        flow_speed=flow_speed)
    return metric_profiler

def plot_metrics_one_fish(
        plots_output_dir,
        heatmaps_output_dir,
        metric_profiler: SwimCapacityProfiler):
    ''' Capacity metrics and location heatmap.
    '''
    plot_capacity_metrics(plots_output_dir, metric_profiler)
    if heatmaps_output_dir is not None:
        plot_location_heatmaps(heatmaps_output_dir, metric_profiler)

def plot_metrics_each_fish(
        output_dir: pathlib.Path,
        include_heatmaps: bool,
        pose_filters: list=None):
    ''' Plotting metrics for all fish, assumed precalculated.
    '''
    logger = loggers.get_qc_logger(__name__)
    output_dir = pathlib.Path(output_dir)
    plots_output_dir = output_dir / 'plots'
    plots_output_dir.mkdir(exist_ok=True, parents=True)
    heatmaps_output_dir = None
    if include_heatmaps:
        heatmaps_output_dir = output_dir / 'heatmaps'
        heatmaps_output_dir.mkdir(exist_ok=True, parents=True)
    if list(plots_output_dir.glob('*.png')):
        print(' '.join([
            'Output directory is not empty,',
            'so traditional metrics will not be plotted for all fish.']))
        return
    fish_names = sorted(FDM.get_available_fish_names())

    def task(name):
        plot_metrics_one_fish(
                plots_output_dir,
                heatmaps_output_dir,
                get_metrics_profiler(name, pose_filters))
        progress.increment()

    with progress.Progress(len(fish_names)):
        with WorkerSwarm(logger) as swarm:
            for fish_name in fish_names:
                swarm.add_task(lambda n=fish_name: task(n))

def plot_metrics_for_entire_experiment(force_recalculate: bool, include_heatmaps: bool):
    ''' Plot all fish's metrics in experiment, calculating if necessary
    '''
    analyzer = SwimCapacityAnalyzer()
    if not MetricLogger.has_analyzer_metrics(analyzer) or force_recalculate:
        print('Calculating and logging "traditional metric" values...')
        MetricLogger.log_analyzer_metrics(analyzer)
    output_dir = FileLocations.get_capacity_metrics_plot_dir()
    print(f'Saving traditional metrics plots to {output_dir.as_posix()}')
    print('...')
    plot_metrics_each_fish(output_dir, include_heatmaps, pose_filters=pose_filters.BASIC_FILTERS)

def plot_one_fish_metrics(name: str, include_heatmaps: bool):
    ''' Plot one fish's metrics
    '''
    output_dir = FileLocations.get_capacity_metrics_plot_dir()
    plots_output_dir = output_dir / 'plots'
    plots_output_dir.mkdir(exist_ok=True)
    heatmaps_output_dir = None
    if include_heatmaps:
        heatmaps_output_dir = output_dir / 'heatmaps'
        heatmaps_output_dir.mkdir(exist_ok=True)
    plot_metrics_one_fish(
        plots_output_dir,
        heatmaps_output_dir,
        get_metrics_profiler(name, pose_filters.BASIC_FILTERS))
