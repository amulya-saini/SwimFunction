''' Compare all metrics against one another.

As a reminder, we expect these columns to exist after metric calculation:

    SwimCapacityAnalyzer outputs
        'swim_distance'
        'activity'
        'centroid_burst_freq_0cms'
        'pose_burst_freq_0cms'
        'mean_y_10_20cms'
        'time_against_flow_10_20cms'

    StrouhalAnalyzer outputs
        'strouhal_0cms'
        'tail_beat_freq_0cms'
        'tail_beat_freq'

    CruiseWaveformAnalyzer outputs
        'body_frequency'
        'rostral_compensation'

    PostureNoveltyAnalyzer outputs
        'posture_novelty'

    ScoliosisAnalyzer outputs
        'scoliosis'

These optional columns may exist if you supplied
a csv input file with the same name in the "precalculated" folder.

    Likely precalculated, accepted from csv input files.
        'perceived_quality'
        'proximal_axon_regrowth'
        'distal_axon_regrowth'
        'glial_bridging'
        'endurance'
'''
import pathlib
import numpy
import pandas
from matplotlib import pyplot as plt

from swimfunction.data_access import data_utils, assembled_dataframes
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.recovery_metrics.metric_correlation import cluster_metrics, rescale_metrics
from swimfunction.plotting.constants import *
from swimfunction.recovery_metrics.metric_analyzers.MetricCalculator import MetricCalculator
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from swimfunction import FileLocations


BRIDGING_METRIC = 'glial_bridging'

# Using this instead of all columns avoids duplicate metrics
# that report basically the same thing.
# The dict is organized like this:
# name_of_good_metric: [list, of, similar, metrics, to, ignore]
ALIAS_METRICS = {
    'tail_beat_freq': ['tail_beat_freq_0cms', 'strouhal_0cms', 'body_frequency']
}
# This dict goes from an alias to the preferred metric
REVERSE_ALIAS = {}
for _PREFERRED, _LIST_OF_ALIAS in ALIAS_METRICS.items():
    for _A in _LIST_OF_ALIAS:
        REVERSE_ALIAS[_A] = _PREFERRED

# In the Tracking Experiment, these metrics
# were higher at 1 wpi than they were before injury on average.
METRICS_THAT_GO_UP_AFTER_INJURY = [
    'scoliosis',
    'posture_novelty',
    'rostral_compensation',
    'tail_beat_freq',
]

MTP = MetricCalculator.get_metrics_to_printable_dict()

### Data acquisition

def metric_to_printable(m):
    ''' Get the printable metric name if defined.
    Otherwise return metric with _ replaced by spaces
    and the first word is capitalized.
    '''
    printable = MTP.get(m, m)
    if '_' in printable:
        printable.replace('_', ' ')
    return ' '.join([s.capitalize() for s in printable.split(' ')])

def choose_metric_columns(df: pandas.DataFrame):
    ''' Select all columns that are not "fish", "assay", "group", or an alias metric.
    If the preferred metric does not exist, the alias is kept.
    '''
    possible_cols = list(filter(lambda c: c not in ['fish', 'assay', 'group'], df.columns))
    cols = []
    for col in possible_cols:
        if col not in REVERSE_ALIAS \
                or REVERSE_ALIAS[col] not in possible_cols:
            cols.append(col)
    return cols

def standardize_signs(df: pandas.DataFrame) -> pandas.DataFrame:
    ''' We standardize all metrics so they decrease with acute injury (preinjury to 1 wpi).
    If the values go up with injury, we multiply by -1 to make sure the direction
    matches all other metrics.
    This is critical so the correlation matrix clusters correctly.
    '''
    for c in df.columns:
        if c in METRICS_THAT_GO_UP_AFTER_INJURY:
            df.loc[:, c] = df.loc[:, c] * -1
    return df

def metrics_to_printable_with_signs(metrics) -> dict:
    ''' Add a (-) in front of the metric if it went up with injury.
    '''
    return {
        m: metric_to_printable(m) \
            if m not in METRICS_THAT_GO_UP_AFTER_INJURY \
            else f'(-) {metric_to_printable(m)}'
        for m in metrics
    }

def plot_metric_clustermap(
        savedir: pathlib.Path,
        df: pandas.DataFrame,
        group: str=None,
        assay: int=None):
    print('Rescaling DataFrame')
    scaled_df, _scaler = rescale_metrics.rescale_df(df)
    print('Plotting')
    mtp = metrics_to_printable_with_signs(df.columns)
    scaled_df.columns = [mtp[c] for c in scaled_df.columns]
    cg, cor_df = cluster_metrics.plot_spearman_clustermap(
        scaled_df,
        right_triangle_only=True,
        title='',
        numbersize=DOTSIZE,
        ticksize=DOTSIZE,
        tickrotation=90,
        figsize=(5.5, 5.5),
        p_vals_on_top=False)
    if cg is None:
        return
    print('Done!')
    group_str = f'{group}' if group is not None else ''
    group_title = 'All Fish' if group is None else 'Males' if group == 'M' else 'Females' if group == 'F' else group
    annot_type = 'Spearman R'
    cg.ax_heatmap.tick_params(axis='both', which='both', labelsize=DOTSIZE)
    cg.ax_cbar.set_title(annot_type, fontsize=AXIS_TICKSIZE, pad=15)
    cg.ax_cbar.tick_params(axis='both', which='major', labelsize=DOTSIZE)
    xlim = (
        cg.ax_cbar.get_xlim()[0],
        cg.ax_cbar.get_xlim()[0] + abs(numpy.subtract(*cg.ax_cbar.get_ylim())) / 4)
    cg.ax_cbar.set_xlim(*xlim)
    cg.ax_cbar.set_aspect('equal')
    cg.figure.subplots_adjust(left=0.02, bottom=0.12, right=0.88, top=0.96)
    cg.figure.suptitle(f'{group_title} {assay}wpi', fontsize=TITLE_SIZE)
    outpath = savedir / f'{group_str}{assay}wpi_metrics_clustermap_spearman.png'
    cor_df.to_csv(outpath.with_suffix('.csv'), mode='w')
    print('Saving', outpath)
    mpl_helpers.save_fig(cg.figure, outpath)
    plt.close(cg.figure)

def match_group(names: numpy.ndarray, group):
    if group is None:
        return numpy.full(len(names), True)
    return [data_utils.fish_name_to_group(n) == group for n in names]

def remove_pointless_metrics(metrics_df: pandas.DataFrame):
    ''' Remove columns that are completely uniform
    (as those will break the downstream calculations.)
    '''
    cols_to_keep = []
    for c in metrics_df.columns:
        if len(metrics_df[c].unique()) > 1:
            cols_to_keep.append(c)
    return metrics_df[cols_to_keep]

### Main
def main():
    ''' Plot a correlation matrix for each assay,
    all groups together and separated between groups.
    '''
    savedir = FileLocations.mkdir_and_return(
        FileLocations.get_plots_dir() / 'correlation_clustermaps')
    metrics_df = assembled_dataframes.get_metric_dataframe()
    assays = metrics_df['assay'].unique()
    preferred_index = list(filter(
        lambda c: c in metrics_df.columns,
        ['fish', 'assay', 'group']))
    metrics_df = metrics_df.set_index(preferred_index)
    metrics = choose_metric_columns(metrics_df)
    metrics_df = metrics_df.loc[:, metrics]
    metrics_df = standardize_signs(metrics_df)
    for group in [None] + FDM.get_groups():
        for assay in assays:
            subdf = metrics_df[
                (metrics_df.index.get_level_values('assay') == assay)
                & (match_group(metrics_df.index.get_level_values('fish'), group))]
            subdf = remove_pointless_metrics(subdf)
            if not subdf.size:
                continue
            print(subdf.columns)
            plot_metric_clustermap(
                savedir=savedir,
                df=subdf,
                group=group,
                assay=assay)

if __name__ == '__main__':
    FileLocations.parse_default_args()
    main()
