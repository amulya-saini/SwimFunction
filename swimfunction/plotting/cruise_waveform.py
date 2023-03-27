# New change: not showing all vs >20%, showing <20% vs >20%
''' Plots like Figure 2 in the paper

# TODO use ugly but obvious colors instead for the public repo.
'''

from matplotlib import pyplot as plt

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from swimfunction.data_access.assembled_dataframes \
    import get_metric_dataframe, get_waveform_dataframe, BRIDGING_SELECTION
from swimfunction.plotting.constants import AXIS_LABELSIZE, WIDER_DASH
from swimfunction.plotting.matplotlib_helpers \
    import shrink_lines_and_dots, set_axis_tick_params, save_fig

# Specific plotting
from swimfunction.plotting.swim_specific_helpers import make_assay_barplot, plot_waveform

ROSTRAL_COMP_METRIC = 'rostral_compensation'

def plot_frequency_and_strouhal(tail_beat_ax, strouhal_ax, group):
    ''' Plot frequency and strouhal on their axes
    '''
    df = get_metric_dataframe(group=group)
    make_assay_barplot(
        df,
        group,
        'tail_beat_freq',
        tail_beat_ax,
        dodge=False,
        hue='group',
        palette=None)
    tail_beat_ax.set_ylim(0, 20.5)
    tail_beat_ax.set_yticks((0, 10, 20))
    tail_beat_ax.set_ylabel('tail beat frequency Hz', fontsize=AXIS_LABELSIZE, labelpad=15)

    make_assay_barplot(
        df,
        group,
        'strouhal_0cms',
        strouhal_ax,
        dodge=False,
        hue='group',
        palette=None)
    strouhal_ax.set_ylim(0, 0.41)
    strouhal_ax.set_ylabel('Strouhal 0 cm/s flow', fontsize=AXIS_LABELSIZE, labelpad=15)
    xlim = strouhal_ax.get_xlim()
    strouhal_ax.hlines(
        y=0.2, xmin=xlim[0], xmax=xlim[1], linewidth=2, color='black', linestyles=WIDER_DASH)
    strouhal_ax.hlines(
        y=0.4, xmin=xlim[0], xmax=xlim[1], linewidth=2, color='black', linestyles=WIDER_DASH)
    strouhal_ax.set_xlim(*xlim) # reinforce original limits despine new hlines

    for ax in [tail_beat_ax, strouhal_ax]:
        ax.set_xlabel('')
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        shrink_lines_and_dots(ax, linewidth=2)

def plot_amplitude_by_bridging(
        savedir, group, bridging_threshold: float=0.2, assays_to_show_waveform=None):
    ''' Plot cruise waveform split by bridging outcome
    '''
    if assays_to_show_waveform is None:
        assays_to_show_waveform = FDM.get_available_assay_labels()

    group_str = f'{group}' if group is not None else 'all_fish'
    figpath = savedir / f'{group_str}_cruise_amplitude_by_bridging.png'

    df_low_bridging = get_metric_dataframe(
        group=group, bridging_selection=BRIDGING_SELECTION.low)
    df_high_bridging = get_metric_dataframe(
        group=group, bridging_selection=BRIDGING_SELECTION.high)
    df_waveform_low = get_waveform_dataframe(
        'amplitudes', group=group, assays=assays_to_show_waveform,
        bridging_selection=BRIDGING_SELECTION.low)
    df_waveform_high = get_waveform_dataframe(
        'amplitudes', group=group, assays=assays_to_show_waveform,
        bridging_selection=BRIDGING_SELECTION.high)

    # count fish with non-nan amplitude metric
    n_high_male = df_high_bridging[df_high_bridging['group'] == 'M']\
        .loc[:, ('fish', ROSTRAL_COMP_METRIC)].dropna()['fish'].unique().size
    n_high_female = df_high_bridging[df_high_bridging['group'] == 'F']\
        .loc[:, ('fish', ROSTRAL_COMP_METRIC)].dropna()['fish'].unique().size

    n_low_male   = df_low_bridging[df_low_bridging['group'] == 'M']\
        .loc[:, ('fish', ROSTRAL_COMP_METRIC)].dropna()['fish'].unique().size
    n_low_female = df_low_bridging[df_low_bridging['group'] == 'F']\
        .loc[:, ('fish', ROSTRAL_COMP_METRIC)].dropna()['fish'].unique().size

    print(group, 'counts')
    print('  Low')
    for wpi in assays_to_show_waveform:
        print(
            '   ', wpi, 'wpi:',
            df_low_bridging.loc[df_low_bridging['assay'] == wpi,
                                ROSTRAL_COMP_METRIC].dropna().shape[0])
    print('  High')
    for wpi in assays_to_show_waveform:
        print(
            '   ', wpi, 'wpi:',
            df_high_bridging.loc[df_high_bridging['assay'] == wpi,
            ROSTRAL_COMP_METRIC].dropna().shape[0])

    fig, axs = plt.subplots(3, 2, figsize=(5, 6.5))
    axs = axs.flatten()
    worst_amplitude_ax = axs[0]
    best_amplitude_ax = axs[1]
    worst_rostral_compensation_ax = axs[2]
    best_rostral_compensation_ax = axs[3]
    tail_beat_ax = axs[4]
    strouhal_ax = axs[5]

    group_tag = group if group is not None else 'M+F'
    plot_waveform(df_waveform_low, worst_amplitude_ax, 'amplitudes')
    make_assay_barplot(
        df_low_bridging,
        f'{group_tag}_low_bridging',
        ROSTRAL_COMP_METRIC,
        worst_rostral_compensation_ax,
        dodge=False,
        hue='group',
        palette=None)

    plot_waveform(df_waveform_high, best_amplitude_ax, 'amplitudes')
    make_assay_barplot(
        df_high_bridging,
        f'{group_tag}_high_bridging',
        ROSTRAL_COMP_METRIC,
        best_rostral_compensation_ax,
        dodge=False,
        hue='group',
        palette=None)

    plot_frequency_and_strouhal(tail_beat_ax, strouhal_ax, group)

    for ax in axs:
        set_axis_tick_params(ax)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    n_str = f'n=({n_low_male}m, {n_low_female}f)' \
        if group is None else f'n={n_low_male}' if group == 'M' else f'n={n_low_female}'
    worst_amplitude_ax.set_title(
        f'Amplitude\nBridging < {bridging_threshold:.1f}, {n_str}', fontsize=14)
    worst_rostral_compensation_ax.set_title(
        f'Rostral Compensation\nBridging < {bridging_threshold:.1f}, {n_str}', fontsize=14)
    n_str = f'n=({n_high_male}m, {n_high_female}f)' \
        if group is None else f'n={n_high_male}' if group == 'M' else f'n={n_high_female}'
    best_amplitude_ax.set_title(
        f'Amplitude\nBridging > {bridging_threshold:.1f}, {n_str}', fontsize=14)
    best_rostral_compensation_ax.set_title(
        f'Rostral Compensation\nBridging > {bridging_threshold:.1f}, {n_str}', fontsize=14)

    for ax in [worst_amplitude_ax, best_amplitude_ax]:
        ax.set_yticks((0, 0.1, 0.2))

    mpl_helpers.same_axis_lims(
        (worst_amplitude_ax, best_amplitude_ax), False, True)
    mpl_helpers.same_axis_lims(
        (worst_rostral_compensation_ax, best_rostral_compensation_ax), False, True)

    fig.tight_layout()
    save_fig(fig, figpath)
    plt.close(fig)

def plot_amplitude_ignore_bridging(savedir, group, assays_to_show_waveform=None):
    ''' Plot cruise waveform but not split by bridging outcome
    '''
    if assays_to_show_waveform is None:
        assays_to_show_waveform = FDM.get_available_assay_labels()

    group_str = f'{group}' if group is not None else 'all_fish'
    figpath = savedir / f'{group_str}_cruise_amplitude.png'

    df = get_metric_dataframe(group=group)
    df_waveform = get_waveform_dataframe(
        'amplitudes', group=group, assays=assays_to_show_waveform)

    # count fish with non-nan amplitude metric
    n_male = df[df['group'] == 'M']\
        .loc[:, ('fish', ROSTRAL_COMP_METRIC)].dropna()['fish'].unique().size
    n_female = df[df['group'] == 'F']\
        .loc[:, ('fish', ROSTRAL_COMP_METRIC)].dropna()['fish'].unique().size

    print(group, 'counts')
    for wpi in assays_to_show_waveform:
        print(
            '   ', wpi, 'wpi:',
            df.loc[df['assay'] == wpi,
            ROSTRAL_COMP_METRIC].dropna().shape[0])

    fig, axs = plt.subplots(2, 2, figsize=(5, 6.5))
    axs = axs.flatten()
    amplitude_ax = axs[0]
    rostral_comp_ax = axs[1]
    tail_beat_ax = axs[2]
    strouhal_ax = axs[3]

    group_tag = group if group is not None else 'M+F'
    plot_waveform(df_waveform, amplitude_ax, 'amplitudes')
    make_assay_barplot(
        df,
        f'{group_tag}',
        ROSTRAL_COMP_METRIC,
        rostral_comp_ax,
        dodge=False,
        hue='group',
        palette=None)

    plot_frequency_and_strouhal(tail_beat_ax, strouhal_ax, group)

    for ax in axs:
        set_axis_tick_params(ax)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    n_str = f'n=({n_male}m, {n_female}f)' \
        if group is None else f'n={n_male}' if group == 'M' else f'n={n_female}'
    amplitude_ax.set_title(
        f'Amplitude n={n_str}', fontsize=14)
    rostral_comp_ax.set_title(
        f'Rostral Compensation n={n_str}', fontsize=14)

    amplitude_ax.set_yticks((0, 0.1, 0.2))

    fig.tight_layout()
    save_fig(fig, figpath)
    plt.close(fig)

def main(savedir):
    ''' Plot according to whether glial bridging is available.
    '''
    df = get_metric_dataframe().reset_index()
    has_bridging = 'glial_bridging' in df and df['glial_bridging'].dropna().size
    for group in ['M', 'F', None]:
        if has_bridging:
            plot_amplitude_by_bridging(savedir, group)
        else:
            plot_amplitude_ignore_bridging(savedir, group)
