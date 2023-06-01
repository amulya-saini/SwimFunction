''' Use this to get basic plots of many metrics
'''
import traceback
from matplotlib import pyplot as plt
from swimfunction import loggers, FileLocations
from swimfunction.data_access.assembled_dataframes import get_metric_dataframe
from swimfunction.plotting.recovery_metrics import plot_each_fish_capacity_metrics, \
    plot_group_capacity
from swimfunction.plotting.recovery_metrics.ScoliosisPlotter import ScoliosisPlotter
from swimfunction.plotting.recovery_metrics.PostureNoveltyPlotter import PostureNoveltyPlotter
from swimfunction.plotting.recovery_metrics \
    import MetricCorrelationPlotter, function_by_structure_correlations
from swimfunction.plotting import eigenfish_figure, cruise_waveform, \
    plot_recovery_prediction, plot_cruise_embedding
from swimfunction.trajectories import metric_tca
from swimfunction.global_config.config import config

# Whether the fish identities carry over from assay to assay.
# For example, did you house the fish separately and keep track
# of which is which for the entire expriment? If so, put it in the config.ini file.
FISH_WERE_TRACKED = config.getboolean('EXPERIMENT DETAILS', 'individuals_were_tracked')

def get_logger() -> loggers.logging.Logger:
    ''' Get the default plotting logger.
    '''
    return loggers.get_plotting_logger(__name__)

def _run_gracefully(fn, *args, **kwargs):
    ''' Runs the provided function
    catching and printing any exceptions.
    Prevents unexpected errors from ending the workflow entirely.
    '''
    rv = None
    try:
        rv = fn(*args, **kwargs)
    except Exception as e:
        get_logger().error(e)
    return rv

def header(msg):
    ''' Log the message loud and clear.
    '''
    get_logger().info('\n------- %s -------\n', msg)

def plot_lateral_scoliosis():
    ''' Plot scoliosis by assay box plots
    and (if fish identities were tracked) week-to-week correlations.
    '''
    save_dir = FileLocations.get_plots_dir()
    header('Plotting Scoliosis')
    plotter = ScoliosisPlotter()
    if FISH_WERE_TRACKED:
        fig = plotter.plot_trendlines()
        fig.savefig(save_dir / 'lateral_scoliosis_weekly_trends.png')
        plt.close(fig)
        fig = plotter.plot_weekly_correlation()
        fig.savefig(save_dir /
            'lateral_scoliosis_weekly_correlations.png')
        plt.close(fig)
    fig = plotter.plot_boxplot()
    if save_dir is not None:
        fig.savefig(save_dir / 'lateral_scoliosis_group_boxplot.png')
        plt.close(fig)

def plot_pose_posture_novelty():
    ''' Plot percent novelty by assay box plots.
    '''
    header('Plotting Posture Novelty')
    fig = PostureNoveltyPlotter().plot_posture_novelty()
    fig.savefig(
        FileLocations.get_plots_dir() / 'percent_pose_novelty_boxplot.png')
    plt.close(fig)

def plot_metrics_vs_structure():
    ''' Plot metrics vs structure
    only if structure (cellular regeneration) has values.
    '''
    df = get_metric_dataframe()
    if 'glial_bridging' in df.columns and df['glial_bridging'].dropna().size:
        _run_gracefully(function_by_structure_correlations.main,
            FileLocations.mkdir_and_return(
                FileLocations.get_plots_dir() / 'metrics_vs_cellular_regeneration')
        )

def plot_predictions_and_tca():
    if not FileLocations.get_outcome_prediction_csv().exists():
        get_logger().info('Predictions do not exist, so they will not be plotted.')
    else:
        header('Plotting predictions and outcomes (if available)')
        _run_gracefully(plot_recovery_prediction.plot_main_figure,
            FileLocations.mkdir_and_return(
                FileLocations.get_plots_dir() / 'outcome_prediction'))
    header('Performing and plotting TCA')
    _run_gracefully(metric_tca.main,
        FileLocations.mkdir_and_return(FileLocations.get_plots_dir() / 'tca'))

def plot_metrics():
    ''' All main plots of interest.
    '''
    header('Plotting eigenfish')
    _run_gracefully(eigenfish_figure.main, FileLocations.get_plots_dir())

    header('Plotting posture novelty and scoliosis')
    _run_gracefully(plot_lateral_scoliosis)
    _run_gracefully(plot_pose_posture_novelty)

    header('Plotting metric correlations')
    _run_gracefully(MetricCorrelationPlotter.main)

    header('Plotting swim capacity metrics')
    _run_gracefully(plot_group_capacity.main)
    _run_gracefully(plot_each_fish_capacity_metrics.plot_metrics_for_entire_experiment,
        force_recalculate=False, include_heatmaps=False)

    header('Plotting cruise waveform')
    _run_gracefully(cruise_waveform.main,
        FileLocations.mkdir_and_return(
            FileLocations.get_plots_dir() / 'cruise_waveform'))

def plotting_main():
    ''' Make all plots.
    '''
    plot_metrics()
    plot_metrics_vs_structure()
    if FISH_WERE_TRACKED:
        plot_predictions_and_tca()
    header('Plotting (calculating too, if needed) cruise embedding...')
    _run_gracefully(plot_cruise_embedding.main)

if __name__ == '__main__':
    plt.switch_backend('agg') # No gui will be created. Safer for distributed processing.
    ARGS = FileLocations.parse_default_args()
    try:
        plotting_main()
    except Exception as _E:
        # Log all exceptions
        get_logger().error('Exception occurred in %s', __name__)
        get_logger().error(_E)
        get_logger().error(traceback.format_exc())
