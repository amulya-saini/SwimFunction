''' Plot prediction outcomes
'''
import pathlib
import numpy
import pandas
import seaborn
from scipy import stats as spstats
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick

from swimfunction import FileLocations
from swimfunction.data_access import data_utils
from swimfunction.data_access.assembled_dataframes import get_metric_dataframe
from swimfunction.plotting.constants import FIGURE_WIDTH, FIGURE_ROW_HEIGHT
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from swimfunction.plotting import swim_specific_helpers as sph

DISTANCE_METRIC = 'swim_distance'
ROSTRAL_COMP_METRIC = 'rostral_compensation'
BRIDGING_METRIC = 'glial_bridging'
STRUCTURAL_METRICS = [
    BRIDGING_METRIC,
    'proximal_axon_regrowth',
    'distal_axon_regrowth'
]

def get_sig_str(pval: str):
    ''' Get significance star string for the pvalue
    '''
    s = 'n.s.'
    if pval < 0.001:
        s = '***'
    elif pval < 0.01:
        s = '**'
    elif pval < 0.05:
        s = '*'
    return s

def get_full_metric_lims(metric, assays: list=None, names_to_ignore: list=None):
    ''' Get axis lims for the given metric.
    '''
    df = get_metric_dataframe(assays=assays, names_to_ignore=names_to_ignore)
    return df[metric].min(), df[metric].max()

def plot_distance_by_compensation(
        ax,
        df: pandas.DataFrame,
        hue_column: str,
        cmap: str):
    ''' Plot distance versus rostral compensation
    '''
    x_metric = DISTANCE_METRIC
    y_metric = ROSTRAL_COMP_METRIC
    ax.set_xlabel(sph.metric_to_axis_label(x_metric))
    ax.set_ylabel(sph.metric_to_axis_label(y_metric))
    msize = 14
    lwidth = 0.25
    DARK_GOOD = 'mediumturquoise'
    DARK_BAD = 'orange'
    df = df[numpy.in1d(df['assay'], [1, 2])].groupby('fish').mean()
    cmapper = None
    if numpy.issubdtype(df[hue_column], bool):
        cmapper = lambda x: DARK_GOOD if x else DARK_BAD
    else:
        cmapper = lambda x: plt.get_cmap(cmap)(x) if not numpy.isnan(x) else 'gray'
    # plot df and reference x vs y metrics colored by experiment, marker by experiment

    groups = numpy.asarray([data_utils.fish_name_to_group(n) for n in df.index])
    for group in set(groups):
        d = df[x_metric][groups == group]
        a = df[y_metric][groups == group]
        c = df[hue_column][groups == group]
        cvals = c.apply(cmapper).values
        pred_mask = df['will_recover_best'][groups == group] \
            if 'will_recover_best' in df.columns else numpy.ones(df.shape[0])
        ax.scatter(
            d[pred_mask], a[pred_mask],
            edgecolors='k', linewidths=lwidth, s=msize,
            marker='D', facecolors=cvals[pred_mask])
        ax.scatter(
            d[~pred_mask], a[~pred_mask],
            edgecolors='k', linewidths=lwidth, s=msize,
            marker='o', facecolors=cvals[~pred_mask])
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel('Mean Swim Distance (px)', fontsize=10)
    if y_metric == ROSTRAL_COMP_METRIC:
        ax.set_ylabel('Avg Rostral Comp (Δradians)', fontsize=10)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 5))
    ax.xaxis.offsetText.set_fontsize(8)
    mpl_helpers.shrink_lines_and_dots(ax, linewidth=lwidth, dotsize=msize)

def plot_final_sorting(ax, outcomes_df: pandas.DataFrame, y_column: str):
    ''' Plot final outcomes
    '''
    LIGHT_GOOD = 'paleturquoise'
    LIGHT_BAD = 'bisque'
    outcomes_df.loc[:, 'outcome'] = ['Good' \
        if x else 'Poor' for x in outcomes_df.loc[:, 'will_recover_best']]
    outcomes_df = outcomes_df.sort_values(
        by=['will_recover_best', 'group'], ascending=[False, True])
    outcomes_df.loc[outcomes_df['group'] == 'F', 'group'] = \
        f'Females (n={(outcomes_df["group"] == "F").sum()})'
    outcomes_df.loc[outcomes_df['group'] == 'M', 'group'] = \
        f'Males (n={(outcomes_df["group"] == "M").sum()})'
    seaborn.stripplot(
        x='group',
        y=y_column,
        hue='outcome',
        hue_order=('Poor', 'Good'),
        data=outcomes_df,
        dodge=True,
        palette={'Male': 'black', 'Female': 'black', 'Good': 'black', 'Poor': 'black'},
        size=2,
        zorder=2,
        ax=ax)
    violins = seaborn.violinplot(
        x='group',
        y=y_column,
        hue='outcome',
        hue_order=('Poor', 'Good'),
        data=outcomes_df,
        inner=None,
        linewidth=0,
        dodge=True,
        palette={'Male': '#aaaaff', 'Female': '#ffaaaa', 'Good': LIGHT_GOOD, 'Poor': LIGHT_BAD},
        zorder=1,
        ax=ax)

    top = violins.viewLim.y1
    outcomes_df['outcome_bool'] = outcomes_df['outcome'] == 'Good'
    for group, textx in zip(outcomes_df['group'].unique(), (0, 1)):
        _t, p = spstats.ttest_ind(
            outcomes_df.loc[
                (outcomes_df['group'] == group) \
                & outcomes_df['outcome_bool'].values.flatten(), y_column],
            outcomes_df.loc[
                (outcomes_df['group'] == group) \
                & ~outcomes_df['outcome_bool'].values.flatten(), y_column])
        ax.hlines(y=top, xmin=textx-0.2, xmax=textx+0.2, color='black')
        ax.text(
            x=textx, y=top,
            s=f'{group}\n{get_sig_str(p)} p = {p:.3f}',
            fontsize=8)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol=None))
    ax.set_xlabel('Groups by Ranked 1 & 2 WPI Average', fontsize=10)
    ax.set_ylabel(sph.metric_to_axis_label(y_column), fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, symbol=None))
    mpl_helpers.shrink_lines_and_dots(ax, dotsize=2)

def plot_final_sorting_multiple_y(outcomes_df: pandas.DataFrame):
    LIGHT_GOOD = 'paleturquoise'
    LIGHT_BAD = 'bisque'
    if not outcomes_df.size:
        return None, None
    outcomes_df.loc[:, 'outcome'] = [
        'Good' if x else 'Poor' for x in outcomes_df.loc[:, 'will_recover_best']]
    outcomes_df = outcomes_df.sort_values(
        by=['will_recover_best', 'group'], ascending=[False, True])
    outcomes_df.loc[outcomes_df['group'] == 'F', 'group'] = \
        f'Females (n={(outcomes_df["group"] == "F").sum()})'
    outcomes_df.loc[outcomes_df['group'] == 'M', 'group'] = \
        f'Males (n={(outcomes_df["group"] == "M").sum()})'
    outcomes_df = outcomes_df.loc[:, ['group', 'outcome'] + STRUCTURAL_METRICS]\
        .melt(['group', 'outcome'])
    if not outcomes_df.size:
        return None, None
    black_palette = {'Good': 'black', 'Poor': 'black'}
    color_palette = {'Good': LIGHT_GOOD, 'Poor': LIGHT_BAD}
    catplot_kwargs = dict(
        data=outcomes_df,
        x='outcome',
        y='value',
        col='variable',
        hue='outcome',
        dodge=True,
        hue_order=('Poor', 'Good'))
    catplot_kwargs['order'] = ('Poor', 'Good')
    violin = seaborn.catplot(
        kind='violin', inner=None, linewidth=0, palette=color_palette, **catplot_kwargs)
    strip = seaborn.catplot(kind='strip', size=2, palette=black_palette, **catplot_kwargs)

    outcomes_df['outcome_bool'] = outcomes_df['outcome'] == 'Good'
    print('Testing whether outcome prediction was significant:')
    for metric in STRUCTURAL_METRICS:
        group_selector = numpy.ones_like(outcomes_df.shape[0])
        bad = outcomes_df.loc[
            group_selector \
            & (outcomes_df['variable'] == metric) \
            & outcomes_df['outcome_bool'].values.flatten(), 'value']
        good = outcomes_df.loc[
            group_selector \
            & (outcomes_df['variable'] == metric)\
            & ~outcomes_df['outcome_bool'].values.flatten(), 'value']
        _t, p = spstats.ttest_ind(bad, good)
        sigtext = f'{metric}\n{get_sig_str(p)} p = {p:.4f}'
        print(sigtext)

    mpl_helpers.same_axis_lims(
        violin.figure.get_axes() + strip.figure.get_axes(),
        same_x_lims=False, same_y_lims=True)
    for ax in violin.figure.get_axes() + strip.figure.get_axes():
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol=None))
        ax.set_xlabel('Groups by Ranked 1 & 2 WPI Average', fontsize=10)
        ax.set_ylabel('regeneration (%)', fontsize=10)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, symbol=None))
        mpl_helpers.shrink_lines_and_dots(ax, dotsize=2)
    return violin.figure, strip.figure

def add_nope_8wpi_prediction_group(df, only_keep_structural_metrics=True):
    ''' Adds prediction group to the 8 WPI assays
    '''
    predictions = pandas.read_csv(
        FileLocations.get_outcome_prediction_final_name_key_csv(),
        header=0,
        index_col='fish')
    predictions['will_recover_best'] = predictions['predicted_well']
    if not only_keep_structural_metrics:
        return pandas.merge(predictions, df, on='fish')
    return pandas.merge(
        predictions,
        df[df['assay'] == 8].set_index('fish').loc[:, STRUCTURAL_METRICS + ['group']], on='fish')

def add_predictions(df):
    ''' Adds predictions to the 1,2 WPI assays
    '''
    predictions = pandas.read_csv(
        FileLocations.get_outcome_prediction_csv(),
        header=[0], index_col=[0])
    return pandas.merge(df, predictions['will_recover_best'], on='fish')

def plot_main_figure(savedir: pathlib.Path):
    ''' Plot outcome prediction details.
    '''
    fig, nope_axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH/2, FIGURE_ROW_HEIGHT))

    df = get_metric_dataframe()
    early_df = add_predictions(df[numpy.in1d(df['assay'], (1, 2))])
    early_df['will_recover_best'] = early_df['will_recover_best'].astype(bool)
    early_df_f = early_df[[data_utils.fish_name_to_group(f) == 'F' for f in early_df['fish']]]
    early_df_m = early_df[[data_utils.fish_name_to_group(f) == 'M' for f in early_df['fish']]]
    plot_distance_by_compensation(nope_axs[0], early_df_f, 'will_recover_best', 'RdBu')
    nope_axs[0].set_title('Female')
    plot_distance_by_compensation(nope_axs[1], early_df_m, 'will_recover_best', 'RdBu')
    nope_axs[1].set_title('Male')
    nopevplot, nopesplot = plot_final_sorting_multiple_y(
        add_nope_8wpi_prediction_group(df[df['assay'] == 8]))

    if nopevplot is not None:
        for snsplot in (nopevplot, nopesplot):
            snsplot.set_size_inches(FIGURE_WIDTH * 0.5, FIGURE_ROW_HEIGHT)
            snsplot.tight_layout()
        for ax in fig.get_axes() \
                + nopevplot.get_axes() \
                + nopesplot.get_axes():
            mpl_helpers.set_axis_tick_params(ax, labelsize=8)
            mpl_helpers.shrink_lines_and_dots(ax, linewidth=2)
        mpl_helpers.save_fig(nopevplot, savedir / 'outcome_prediction_final_structure_violinplot.png')
        mpl_helpers.save_fig(nopesplot, savedir / 'outcome_prediction_final_structure_stripplot.png')
        plt.close(nopevplot)
        plt.close(nopesplot)

    # Finalize axes
    nope_axs[1].set_xlabel(None)
    nope_axs[1].set_ylabel(None)
    mpl_helpers.same_axis_lims(nope_axs)

    fig.suptitle('Ranked Metrics', fontsize=16)

    mpl_helpers.save_fig(fig, savedir / 'outcome_prediction_early_ranks.png')
    plt.close(fig)

