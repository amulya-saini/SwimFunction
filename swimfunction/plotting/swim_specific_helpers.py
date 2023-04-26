''' Helpers specific to swim function.
See matplotlib_helpers for general plotting helpers.
'''
from collections import defaultdict
import numpy
import pandas
from scipy import stats as spstats
import seaborn

from swimfunction.plotting.constants \
    import AXIS_TICKSIZE, TITLE_SIZE, AXIS_LABELSIZE, WIDER_DASH
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from swimfunction.global_config.config import config

WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')

# WPI_COLORS only really works for 1 to 8 wpi and control.
#   All other assays will be colored black, same as control, -1, or 0.
#   If your experiment has assays beyond 8 wpi, it may misbehave.
WPI_COLORS = defaultdict(lambda: 'black')
for _W, _C in zip(range(1, 9), mpl_helpers.get_cmap_values('coolwarm_r', 8)):
    WPI_COLORS[_W] = _C
    WPI_COLORS[str(_W)] = _C

def metric_to_title(metric: str) -> str:
    ''' Get the metric's title
    '''
    predefined = {
        'swim_distance': 'Swim Distance',
        'activity': 'Centroid Time Active',
        'centroid_burst_freq_0cms': 'Centroid Burst Frequency 0 cm/s',
        'pose_burst_freq_0cms': 'Posture Burst Frequency 0 cm/s',
        'mean_y_10_20cms': 'Mean-Y 10, 20 cm/s',
        'time_against_flow_10_20cms': 'Time Against Flow 10, 20 cm/s',
        'strouhal_0cms': 'Strouhal 0 cm/s',
        'tail_beat_freq_0cms': 'Tail Beat Frequency 0 cm/s',
        'tail_beat_freq': 'Tail Beat Frequency',
        'body_frequency': 'Body Frequency',
        'rostral_compensation': 'Rostral Compensation',
        'posture_novelty': '% Posture Novelty',
        'scoliosis': 'Lateral Scoliosis',
        'perceived_quality': 'Quality',
        'proximal_axon_regrowth': '% Axon Regrowth (Proximal)',
        'distal_axon_regrowth': '% Axon Regrowth (Distal)',
        'glial_bridging': '% Glial Bridging',
        'endurance': 'Time to Exhaustion'
    }
    return predefined.get(metric, metric.replace('_', ' ').capitalize())

def metric_to_axis_label(metric: str) -> str:
    ''' Get the metric's axis label
    '''
    predefined = {
        'swim_distance': 'swim distance (px)',
        'activity': 'time active (%)',
        'centroid_burst_freq_0cms': 'centroid bursts/min',
        'pose_burst_freq_0cms': 'pose bursts/min',
        'mean_y_10_20cms': 'mean y-position (px)',
        'time_against_flow_10_20cms': 'time against flow (%)',
        'strouhal_0cms': 'Strouhal',
        'tail_beat_freq_0cms': 'tail beat frequency (Hz)',
        'tail_beat_freq': 'tail beat frequency (Hz)',
        'body_frequency': 'frequency (Hz)',
        'rostral_compensation': 'rostral compensation (Δradians)',
        'posture_novelty': 'novel poses (%)',
        'scoliosis': 'scoliosis (radians)',
        'perceived_quality': 'count (%)',
        'proximal_axon_regrowth': 'proximal axon regrowth (%)',
        'distal_axon_regrowth': 'distal axon regrowth (%)',
        'glial_bridging': 'bridging (%)',
        'endurance': 'endurance time (sec)'
    }
    return predefined.get(metric, metric)

def plot_waveform_features(
        ax,
        df_waveform: pandas.DataFrame,
        color,
        label,
        **kwargs):
    ''' Plots dimension against normalized value.
    Each WPI gets its own line.
    x is position on the fish

    Parameters
    ----------
    ax
    df_waveform: pandas.DataFrame
    color
    kwargs: dict
        Passed onto ax.errorbar
    '''
    if df_waveform is None or not df_waveform.size:
        return
    ax.errorbar(
        x=numpy.arange(df_waveform.shape[1]) + 1,
        y=numpy.nanmean(df_waveform, axis=0),
        c=color,
        yerr=spstats.sem(df_waveform, axis=0, nan_policy='omit'),
        capsize=2,
        barsabove=True,
        linestyle=(0, (2, 3)) if label[0] == '8' else '-',
        elinewidth=1,
        ecolor='black',
        linewidth=2,
        label=f'{label}, n={df_waveform.shape[0]}',
        **kwargs
    )

def plot_waveform(df_waveform: pandas.DataFrame, feature_ax, feature):
    '''
    Feature must be in [amplitudes, frequencies]
    '''
    from swimfunction.data_access.fish_manager import DataManager as FDM
    if df_waveform is None or not df_waveform.size:
        return
    group_palette_soft = mpl_helpers.df_to_group_palette(
        pandas.DataFrame({'group': FDM.get_groups()}), soft_colors=True)
    group_palette = mpl_helpers.df_to_group_palette(
        pandas.DataFrame({'group': FDM.get_groups()}), soft_colors=False)
    for group in df_waveform.index.get_level_values('group').unique():
        for assay in df_waveform.index.get_level_values('assay').unique():
            df = df_waveform.loc[
                (df_waveform.index.get_level_values('assay') == assay) \
                    & (df_waveform.index.get_level_values('group') == group)]
            plot_waveform_features(
                feature_ax,
                df,
                group_palette[group] if assay > 0 else group_palette_soft[group],
                f'{assay} WPI')
    feature_ax.legend()
    feature_ax.set_ylabel(feature)
    feature_ax.set_xticks(numpy.arange(df_waveform.shape[1]) + 1)
    ymax = feature_ax.get_ylim()[1]
    if 'amplitude' in feature:
        rostral_caudal_cutoff = 5.5
        feature_ax.vlines(
            x=rostral_caudal_cutoff, ymin=0, ymax=ymax,
            linewidth=2, color='black', linestyles=WIDER_DASH)
    else:
        body_cutoff = 1.5
        feature_ax.vlines(
            x=body_cutoff, ymin=0, ymax=ymax,
            linewidth=2, color='black', linestyles=WIDER_DASH)
    feature_ax.set_ylim(0, ymax)
    feature_ax.set_xlabel('Position Along Fish', fontsize=AXIS_LABELSIZE)
    if feature == 'amplitudes':
        feature_ax.set_title('Mean Angular Amplitude', fontsize=TITLE_SIZE)
        feature_ax.set_ylabel('radians', fontsize=AXIS_LABELSIZE)
    elif feature == 'frequencies':
        feature_ax.set_title('Mean Angular Frequencies', fontsize=TITLE_SIZE)
        feature_ax.set_ylabel('Hz', fontsize=AXIS_LABELSIZE)
    mpl_helpers.shrink_lines_and_dots(feature_ax, linewidth=2)

def assay_tick_labels(ax, x_axis=True):
    ''' Set the axis to have assay labels.
    The first assay is assumed to be control.
    '''
    labels = ['control'] + [f'{x}' for x in range(1, len(WPI_OPTIONS))]
    if x_axis:
        ax.set_xticks(list(range(len(WPI_OPTIONS))))
        ax.set_xticklabels(labels, fontsize=AXIS_TICKSIZE)
        ax.set_xlabel('')
    else:
        ax.set_yticks(list(range(len(WPI_OPTIONS))))
        ax.set_yticklabels(labels, fontsize=AXIS_TICKSIZE)
        ax.set_ylabel('')
    mpl_helpers.set_axis_tick_params(ax)

def make_stripplot(
        ax, df: pandas.DataFrame, column_name: str, include_violin: bool=True):
    '''Creates a violinplot (with swarm overlay) with appropriate colors for groups.
    '''
    if df is None or not df.size:
        return
    group_palette = mpl_helpers.df_to_group_palette(df, soft_colors=True)
    black_palette = {g: 'black' for g in group_palette}
    seaborn.stripplot(
        x='assay',
        y=column_name,
        hue='group',
        data=df,
        dodge=True,
        palette=black_palette,
        size=2,
        zorder=2,
        ax=ax)
    if include_violin:
        seaborn.violinplot(
            x='assay',
            y=column_name,
            hue='group',
            data=df,
            inner=None,
            linewidth=0,
            dodge=True,
            palette=group_palette,
            zorder=1,
            ax=ax)
    ngroups = df['group'].unique().size
    current_handles, current_labels = ax.get_legend_handles_labels()
    ax.legend(current_handles[:ngroups], current_labels[:ngroups])
    mpl_helpers.shrink_lines_and_dots(ax, linewidth=2)

def _get_sem_safe(df, column_name, hue_column, hue_val, x_column, x_val):
    ''' Ensure that the SEM does not calculate for fewer than 2 values.
    Avoids a RuntimeWarning
    '''
    vals = df[(df[hue_column] == hue_val) & (df[x_column] == x_val)][column_name]
    sem = numpy.nan
    if vals.size > 1:
        sem = spstats.sem(vals, nan_policy='omit')
    return sem

def _get_mean_safe(df, column_name, hue_column, hue_val, x_column, x_val):
    ''' Ensures that the mean is calculated for greater than 0 values.
    Avoids a RuntimeWarning
    '''
    subdf = df[(df[hue_column] == hue_val) & (df[x_column] == x_val)]
    mean = numpy.nan
    if subdf.size > 0:
        mean = subdf[column_name].mean()
    return mean

def make_assay_barplot(
        df,
        group_or_condition,
        column_name,
        ax,
        dodge=True,
        hue='group',
        palette=None):
    ''' Make a barplot (x axis is assay) on the given matplotlib axis.
    '''
    if df is None or not df.size:
        return
    if palette is None:
        palette = mpl_helpers.df_to_group_palette(df, soft_colors=True)
    _df = df.copy()
    # Select group
    if 'group' in _df.columns and group_or_condition in _df['group'].values:
        _df = _df[_df['group'] == group_or_condition]
    x_attr = 'assay'
    _df = _df.loc[:, list(set(('fish', 'assay', x_attr, column_name, hue)))].dropna()
    if not _df.size:
        return
    seaborn.barplot(
        data=_df,
        x=x_attr,
        y=column_name,
        dodge=dodge,
        hue=hue,
        palette=palette,
        errorbar=None,
        ax=ax)

    err_dict = {
        'x_pos' : [_bar.xy[0] + (_bar._width / 2) for _bar in ax.patches],
        'mean': [
            _get_mean_safe(_df, column_name, hue, h, x_attr, xl)
                for h in sorted(_df[hue].unique())
                for xl in sorted(_df[x_attr].unique())],
        'sem': [
            _get_sem_safe(_df, column_name, hue, h, x_attr, xl)
                for h in sorted(_df[hue].unique())
                for xl in sorted(_df[x_attr].unique())]
    }
    if not numpy.all(numpy.isnan(err_dict['sem'])):
        ax.errorbar(x=err_dict['x_pos'], y=err_dict['mean'],
            yerr=err_dict['sem'], fmt='none', c='black', capsize=2)
    else:
        ax.scatter(x=err_dict['x_pos'], y=err_dict['mean'], c='black')
    title = metric_to_title(column_name)
    label = metric_to_axis_label(column_name)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_ylabel(label, fontsize=AXIS_LABELSIZE)
    mpl_helpers.shrink_lines_and_dots(ax, linewidth=2)
    assay_tick_labels(ax)
