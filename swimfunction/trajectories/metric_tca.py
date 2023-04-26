''' Tensor decomposition analysis and plots
'''
import pathlib
import json
import numpy
import pandas
import seaborn
import tensortools

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import spatial
from sklearn.cluster import AgglomerativeClustering

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.context_managers.CacheContext import CacheContext
from swimfunction.data_access.assembled_dataframes import get_metric_dataframe
from swimfunction.recovery_metrics.metric_correlation import cluster_metrics, rescale_metrics
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from swimfunction.plotting.recovery_metrics.MetricCorrelationPlotter \
    import metric_to_printable, standardize_signs, metrics_to_printable_with_signs
from swimfunction.plotting.constants import FIGURE_WIDTH, FIGURE_ROW_HEIGHT

class TensorIndex:
    ''' Keeps track of the index names for each dimension.
    '''
    valid_axis_names = ['fish', 'metrics', 'assays']
    axis_name_to_title = {
        'fish': 'Fish',
        'metrics': 'Functional Measurements',
        'assays': 'Assays'
    }
    def __init__(self, a, name_a, b, name_b, c, name_c):
        self.index_values = (a, b, c)
        self.names = [name_a, name_b, name_c]
        if numpy.any(~numpy.in1d(self.names, TensorIndex.valid_axis_names)):
            raise ValueError('Axis names must be in TensorIndex.valid_axis_names')

    def moveaxis(self, source, destination):
        ''' Shift the tensor axes around like numpy.moveaxis
        '''
        kws = {}
        position_args = ['a', 'b', 'c']
        name_args = ['name_a', 'name_b', 'name_c']
        for i, (pos_arg, name_arg) in enumerate(zip(position_args, name_args)):
            if i not in destination:
                kws[pos_arg] = self[i]
                kws[name_arg] = self.names[i]
            else:
                to_take_i = source[destination.index(i)]
                kws[pos_arg] = self[to_take_i]
                kws[name_arg] = self.names[to_take_i]
        return TensorIndex(**kws)

    @property
    def nfish(self) -> int:
        ''' Number of fish in the tensor index
        '''
        return len(self.index_values[self.names.index('fish')])

    def __getitem__(self, idx):
        return self.index_values[idx]

    def __len__(self):
        return len(self.names)

    def __str__(self) -> str:
        parts = []
        for n, t in self.axis_name_to_title.items():
            parts.append(f'{t}: {self[self.names.index(n)]}')
        return ''.join(parts)

    def as_dict(self) -> dict:
        ''' Become a dict for safe pickling
        '''
        return {
            'a': self.index_values[0],
            'b': self.index_values[1],
            'c': self.index_values[2],
            'name_a': self.names[0],
            'name_b': self.names[1],
            'name_c': self.names[2],
        }

    @staticmethod
    def from_dict(d: dict):
        ''' Initialize from a dict
        '''
        if isinstance(d, TensorIndex):
            return d
        return TensorIndex(d['a'], d['name_a'], d['b'], d['name_b'], d['c'], d['name_c'])

# Not euclidean. We care about direction of change, not amount of change.
# As long as recovery is in the same direction, recovery trajectory is similar.
CLUSTERING_DISTANCE = 'cosine'

LINKAGE_METHOD = 'average'

MAX_RANKS_TO_PRECALCULATE = 90
NUM_ENSEMBLE_REPS = 4
ENSEMBLE_METHODS = (
    'cp_als',    # fits unconstrained tensor decomposition.
    'ncp_bcd',   # fits nonnegative tensor decomposition.
    'ncp_hals',  # fits nonnegative tensor decomposition.
)

BRIDGING_METRIC = 'glial_bridging'
STRUCTURAL_METRICS = [
    BRIDGING_METRIC,
    'proximal_axon_regrowth',
    'distal_axon_regrowth'
]
DISTANCE_METRIC = 'swim_distance'
ACTIVITY_METRIC = 'activity'
TIME_METRIC = 'time_against_flow_10_20cms'
BURST_FREQ_METRIC = 'centroid_burst_freq_0cms'
POSE_BURST_FREQ_METRIC = 'pose_burst_freq_0cms'
Y_METRIC = 'mean_y_10_20cms'

def get_fish_names_ordered_by_metric(assay: int, metric=BRIDGING_METRIC):
    ''' Returns fish sorted by metric at certain assay
    If a fish does not exist at that assay, it will not be returned.
    So, this also cuts out missing fish.
    '''
    return get_metric_dataframe(
        assays=[assay]).reset_index().sort_values(by=metric)['fish']

def get_ordered_metrics():
    ''' Metrics likely to be available.
    It will keep as many as truly exist.
    '''
    return [
        'endurance',
        'swim_distance',
        'activity',
        'time_against_flow_10_20cms',
        'scoliosis',
        'posture_novelty',
        'centroid_burst_freq_0cms',
        'pose_burst_freq_0cms',
        'mean_y_10_20cms',
        'perceived_quality',
        'rostral_compensation',
        'tail_beat_freq'
    ]

def drop_fish_with_nan_from_tensor(tensor, t_index: TensorIndex) -> tuple:
    if not t_index.nfish:
        return tensor, t_index
    fish_index = t_index.names.index('fish')
    which_fish_bad = numpy.unique(numpy.where(numpy.isnan(tensor))[fish_index])
    which_fish_good = numpy.asarray([
        i for i in range(t_index[fish_index].size) if i not in which_fish_bad],
        dtype=which_fish_bad.dtype)
    print(' '.join([
        f'WARNING: dropping {which_fish_bad.size} of',
        f'{t_index[fish_index].size} fish from the tensor!!!!!']))
    if fish_index == 0:
        return tensor[which_fish_good, :, :], TensorIndex(
            t_index[0][which_fish_good], t_index.names[0],
            t_index[1], t_index.names[1],
            t_index[2], t_index.names[2])
    if fish_index == 1:
        return tensor[:, which_fish_good, :], TensorIndex(
            t_index[0], t_index.names[0],
            t_index[1][which_fish_good], t_index.names[1],
            t_index[2], t_index.names[2])
    if fish_index == 2:
        return tensor[:, :, which_fish_good], TensorIndex(
            t_index[0], t_index.names[0],
            t_index[1], t_index.names[1],
            t_index[2][which_fish_good], t_index.names[2])
    raise RuntimeError('This dumb hacked together function failed.')

def reorder_tensor_as_fish_assay_metrics(tensor, t_index: TensorIndex):
    old_order = [
        t_index.names.index('fish'),
        t_index.names.index('assays'),
        t_index.names.index('metrics')]
    new_order = [0, 1, 2]
    return numpy.moveaxis(tensor, old_order, new_order), t_index.moveaxis(old_order, new_order)

def get_metric_tensor(
        group: str,
        wpis: list,
        ordered_fish_names: list,
        ordered_metrics: list) -> numpy.ndarray:
    '''
    Returns
    -------
    numpy.ndarray
    '''
    df = get_metric_dataframe(group=group, assays=wpis).set_index(['fish', 'assay'])
    ordered_fish_names = [f for f in ordered_fish_names if f in df.index.get_level_values('fish')]
    ordered_metrics = list(filter(
        lambda m: m in df.columns and df[m].dropna().size, ordered_metrics))
    if 'strouhal_0cms' in ordered_metrics:
        ordered_metrics.remove('strouhal_0cms')
    df = df.loc[:, ordered_metrics]
    df = standardize_signs(df) # Make sure metrics all go down with injury
    columns_to_printable = metrics_to_printable_with_signs(ordered_metrics)
    df = rescale_metrics.rescale_df(df)[0] # Scale after changing any signs
    tensor = []
    for assay in wpis:
        assay_matrix = []
        for fish in ordered_fish_names:
            fish_row = []
            for metric in ordered_metrics:
                if (fish, assay) not in df.index or metric not in df.columns:
                    fish_row.append(numpy.nan)
                else:
                    fish_row.append(df.loc[(fish, assay), metric])
            assay_matrix.append(fish_row)
        tensor.append(assay_matrix)
    tensor, t_index = numpy.asarray(tensor), TensorIndex(
        numpy.asarray(wpis),
        'assays',
        numpy.asarray(ordered_fish_names),
        'fish',
        [columns_to_printable[m] for m in ordered_metrics],
        'metrics')
    tensor, t_index = drop_fish_with_nan_from_tensor(tensor, t_index)
    return tensor, t_index

def get_reconstruction_error(recon, original):
    ''' Reconstruction error
    '''
    nnan = ~numpy.isnan(original)
    return numpy.sum((recon[nnan] - original[nnan])**2) / original[nnan].size

def plot_tca_component_result(fprefix, savedir, group, tensor, t_index: TensorIndex, model):
    group_str = f'{group}' if group is not None else ''
    fig, axs = mpl_helpers.get_figure_cols_sharex(
        model.factors.rank,
        model.factors.ndim,
        figsize=(FIGURE_WIDTH, FIGURE_ROW_HEIGHT*3))
    fig.suptitle('tensortools')
    for comp in range(model.factors.rank):
        for prop in range(model.factors.ndim):
            axs[0, prop].set_title(
                f'{TensorIndex.axis_name_to_title[t_index.names[prop]]} ({tensor.shape[prop]})')
            yy = model.factors[prop][:, comp]
            if t_index.names[prop] == 'assays':
                axs[comp, prop].plot(numpy.arange(len(yy)), yy)
            else:
                axs[comp, prop].bar(numpy.arange(len(yy)), yy)
    row = axs.shape[0] - 1
    # axs[row, 0].set_xlabel(f'{tensor.shape[0]} {t_index.names[0]}')
    # axs[row, 1].set_xlabel(f'{tensor.shape[1]} {t_index.names[1]}')
    # axs[row, 2].set_xlabel(f'{tensor.shape[2]} {t_index.names[2]}')
    for col in range(axs.shape[1]):
        axs[row, col].set_xticks(range(len(t_index[col])))
        rotation = 45 if t_index.names[col] != 'assays' else None
        labels = [l for l in t_index[col]]
        if t_index.names[col] == 'assays':
            labels[0] = 'control'
        axs[row, col].set_xticklabels(labels, rotation=rotation)
    for ax in axs[:-1, :].flatten():
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.tight_layout()
    mpl_helpers.save_fig(fig, savedir / f'{fprefix}{group_str}_metric_TCA.png', include_svg=True)
    plt.close(fig)

# Subplots with factors
def plot_tiny_tca_component_result(
        fprefix,
        savedir,
        group,
        tensor,
        t_index: TensorIndex,
        model,
        fish_clusters,
        color_by_cluster=True):
    df = get_metric_dataframe(assays=[8])
    fish_clusters_ordered = numpy.argsort([
        df.loc[numpy.in1d(df['fish'], c), BRIDGING_METRIC].mean()
        for c in fish_clusters])
    fish_to_rgb = {}

    cmap_colors = mpl_helpers.get_cmap_values('plasma', len(fish_clusters))

    assert len(cmap_colors) == len(fish_clusters)
    cluster_to_rgb = {}
    for i, cluster_i in enumerate(fish_clusters_ordered):
        color = cmap_colors[i]
        cluster_to_rgb[cluster_i] = color
        for fish in fish_clusters[cluster_i]:
            fish_to_rgb[fish] = color

    if color_by_cluster:
        fish_colors = [fish_to_rgb[f] for f in t_index.index_values[t_index.names.index('fish')]]
    else:
        fish_colors = ['black' for f in t_index.index_values[t_index.names.index('fish')]]

    group_str = f'{group}' if group is not None else ''
    figheight = 2 if model.factors.rank == 1 else FIGURE_ROW_HEIGHT * 2
    fig, axs = mpl_helpers.get_figure_cols_sharex(
        model.factors.rank,
        model.factors.ndim,
        figsize=(FIGURE_WIDTH / 2, figheight))
    fig.suptitle('TCA Factors')
    for comp in range(model.factors.rank):
        for prop in range(model.factors.ndim):
            axs[0, prop].set_title(
                f'{TensorIndex.axis_name_to_title[t_index.names[prop]]} ({tensor.shape[prop]})')
            yy = model.factors[prop][:, comp]
            if t_index.names[prop] == 'assays':
                axs[comp, prop].plot(numpy.arange(len(yy)), yy, color='black')
            elif t_index.names[prop] == 'fish':
                axs[comp, prop].bar(numpy.arange(len(yy)), yy, color=fish_colors)
            else:
                axs[comp, prop].bar(numpy.arange(len(yy)), yy, color='black')
    row = axs.shape[0] - 1
    # axs[row, 0].set_xlabel(f'{tensor.shape[0]} {t_index.names[0]}')
    # axs[row, 1].set_xlabel(f'{tensor.shape[1]} {t_index.names[1]}')
    # axs[row, 2].set_xlabel(f'{tensor.shape[2]} {t_index.names[2]}')
    for col in range(axs.shape[1]):
        if t_index.names[col] != 'assays':
            plt.setp(axs[row, col].get_xticklabels(), visible=False)
            continue
        axs[row, col].set_xticks(range(len(t_index[col])))
        labels = [l for l in t_index[col]]
        if t_index.names[col] == 'assays':
            labels[0] = 'control'
        axs[row, col].set_xticklabels(labels)
    print('Ordered metrics:', t_index.index_values[t_index.names.index('metrics')])
    for ax in axs[:-1, :].flatten():
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.tight_layout()
    mpl_helpers.save_fig(fig, savedir / f'{fprefix}{group_str}_metric_TCA.png', include_svg=True)
    plt.close(fig)

    fig, ax = plt.subplots()
    for i in range(len(fish_clusters)):
        ax.scatter(0, 0, label=f'c{i+1}', color=cluster_to_rgb[i])
    ax.legend()
    mpl_helpers.save_fig(
        fig,
        savedir / f'{fprefix}{group_str}_cluster_colors.png',
        include_svg=True)
    plt.close(fig)

def plot_tca_clustering(fprefix, savedir, group, tensor, t_index, model):
    group_str = f'{group}' if group is not None else ''
    fig, axs = plt.subplots(3, 3, figsize=(FIGURE_WIDTH, FIGURE_ROW_HEIGHT*3))
    fig.suptitle('TCA Clustering')
    axs[0, 0].set_title(f'{tensor.shape[0]} {t_index.names[0]}')
    axs[0, 1].set_title(f'{tensor.shape[1]} {t_index.names[1]}')
    axs[0, 2].set_title(f'{tensor.shape[2]} {t_index.names[2]}')
    name_to_cmap = dict(zip(['assays', 'fish', 'metrics'], ['plasma', 'hsv', 'tab20']))
    for prop, prop_name in enumerate(t_index.names):
        cmap = name_to_cmap[prop_name]
        yy = model.factors[prop]
        colors = {
            val: mpl_helpers.get_cmap_values(cmap, len(t_index[prop]))[i]
            for i, val in enumerate(sorted(t_index[prop]))}
        if prop_name == 'assays':
            colors = {
                val: mpl_helpers.get_cmap_values(cmap, len(t_index[prop]))[i]
                for i, val in enumerate(range(1, 9))}
            colors['control'] = 'gray'
        for i, label in enumerate(t_index[prop]):
            if prop_name == 'assays' and label == -1:
                label = 'control'
            axs[0, prop].scatter(yy[i, 0], yy[i, 1], label=str(label), color=colors[label])
            axs[0, prop].set_xlabel('Dim 1')
            axs[0, prop].set_xlabel('Dim 2')
            axs[1, prop].scatter(yy[i, 0], yy[i, 2], label=str(label), color=colors[label])
            axs[1, prop].set_xlabel('Dim 1')
            axs[1, prop].set_xlabel('Dim 3')
            axs[2, prop].scatter(yy[i, 1], yy[i, 2], label=str(label), color=colors[label])
            axs[2, prop].set_xlabel('Dim 2')
            axs[2, prop].set_xlabel('Dim 3')
        axs[0, prop].legend()

        distances = spatial.distance.squareform(
            spatial.distance.pdist(model.factors[prop], metric=CLUSTERING_DISTANCE))
        distance_df = pandas.DataFrame(distances, index=t_index[prop], columns=t_index[prop])
        cg = cluster_metrics.plot_clustering_and_significance(
            distance_df,
            method=LINKAGE_METHOD,
            distance_metric='precomputed',
            title=prop_name,
            annot=distance_df,
            color_annot=distance_df,
            numbersize=5,
            titlesize=6,
            ticksize=5,
            tickrotation=90,
            cmap='RdBu',
            center=0)
        cg.ax_cbar.set_title(prop_name, fontsize=8, pad=15)
        cg.ax_cbar.tick_params(axis='both', which='major', labelsize=3)
        xlim = (
            cg.ax_cbar.get_xlim()[0],
            cg.ax_cbar.get_xlim()[0] + abs(numpy.subtract(*cg.ax_cbar.get_ylim())) / 4)
        cg.ax_cbar.set_xlim(*xlim)
        cg.ax_cbar.set_aspect('equal')
        cg.figure.subplots_adjust(left=0.02, bottom=0.12, right=0.88, top=0.96)
        outpath = savedir / f'{fprefix}{group_str}_{prop_name}_clustermap.png'
        print('Saving', outpath)
        mpl_helpers.save_fig(cg.figure, outpath, include_svg=False)
        plt.close(cg.figure)
    outpath = savedir / f'{fprefix}{group_str}_dotplot.png'
    print('Saving', outpath)
    mpl_helpers.save_fig(fig, outpath, include_svg=False)
    plt.close(fig)

def save_cluster_fish_names(outpath, clusters: list):
    ''' Keep those fish names! Keep those clusters!
    '''
    listified = [
        c.tolist() if isinstance(c, numpy.ndarray) else c
        for c in clusters
    ]
    with open(outpath, 'wt') as ofh:
        ofh.write(json.dumps(listified))

def read_fish_clusters(cpath):
    ''' Read any existing fish clusters, if already saved
    '''
    clusters = []
    if not cpath.exists():
        return clusters
    with open(cpath, 'rt') as ifh:
        contents = ifh.read().strip()
        if contents:
            clusters = json.loads(contents)
            clusters = [numpy.asarray(c) for c in clusters]
    return clusters

def plot_threshold_choice_get_fish_clusters(savedir, t_index, model, cluster_fpath):
    ''' Cluster the fish, save the threshold selection plots and fish clusters.
    '''
    prop = t_index.names.index('fish')
    yy = model.factors[prop]

    ac_kwargs = dict(n_clusters=None, affinity='euclidean', linkage='ward')
    tt = numpy.linspace(start=0.001, stop=5, num=1000)
    nlab = []
    mcs = []
    for t in tt:
        ac = AgglomerativeClustering(distance_threshold=t, **ac_kwargs)
        ac.fit(yy)
        nlab.append(len(numpy.unique(ac.labels_)))
        mcs.append(numpy.min([(ac.labels_ == l).sum() for l in ac.labels_]))
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 0.4, 1.5))
    ax.plot(tt, nlab, '-', linewidth=2, color='black', label='Number of Clusters')
    ax.plot(tt, mcs, '--', linewidth=2, color='black', label='Min Cluster Size')
    ax.set_xlabel('Distance Threshold', fontsize=8)
    ax.set_ylabel('Number of Fish', fontsize=8)
    ax.legend(fontsize=8)
    outpath = savedir / 'choosing_dist_threshold.png'
    print('Saving', outpath)
    mpl_helpers.save_fig(fig, outpath, include_svg=True)
    plt.close(fig)

    chosen_thresold = 1.7

    ac = AgglomerativeClustering(distance_threshold=chosen_thresold, **ac_kwargs)
    ac.fit(yy)

    fish_names = numpy.asarray(t_index.index_values[t_index.names.index('fish')])
    fish_clusters = [
        fish_names[ac.labels_ == l]
        for l in numpy.unique(ac.labels_)
    ]
    save_cluster_fish_names(cluster_fpath, fish_clusters)
    return fish_clusters

# Subplots with theshold choice and cluster outcomes
def plot_tca_fish_clustering(
        plot_fpath: pathlib.Path,
        cluster_fpath: pathlib.Path,
        t_index: TensorIndex,
        model,
        all_outcomes: bool):
    ''' Plot the fish clusters
    '''
    fish_clusters = read_fish_clusters(cluster_fpath)
    if not fish_clusters:
        fish_clusters = plot_threshold_choice_get_fish_clusters(
            plot_fpath.parent, t_index, model, cluster_fpath)
    print(f'Found {len(fish_clusters)} fish clusters')
    df = get_metric_dataframe(assays=[8])
    y_metrics = STRUCTURAL_METRICS + ['scoliosis', 'rostral_compensation']
    if all_outcomes:
        y_metrics = STRUCTURAL_METRICS + get_ordered_metrics()
        y_metrics = list(filter(lambda m: m in df.columns and df[m].dropna().size, y_metrics))
    mat_values = numpy.asarray([
        [
            df.loc[numpy.in1d(df['fish'], c), m].mean()
            for c in fish_clusters
        ]
        for m in y_metrics
    ])
    order = numpy.argsort(mat_values[0, :])
    cluster_labels = numpy.asarray([
            f'c{i+1} (n={len(c)})' for i, c in enumerate(fish_clusters)
        ])[order]
    mat_values = mat_values[:, order]
    row_max_values = numpy.asarray([df.loc[:, m].max() for m in y_metrics]).reshape(-1, 1)
    row_normalized = mat_values / row_max_values
    print(row_max_values)
    fig, ax = plt.subplots(figsize=(4, 2.75 if all_outcomes else 2))
    fontsizes = 10 if all_outcomes else 12
    to_be_percents = STRUCTURAL_METRICS + [ACTIVITY_METRIC, TIME_METRIC, 'posture_novelty']
    valstrs = [['' for _ in range(mat_values.shape[1])] for _ in range(len(y_metrics))]
    annot_kws = dict(va='center', ha='center', fontsize=fontsizes)
    for i in range(mat_values.shape[1]):
        for j, m in enumerate(y_metrics):
            val = mat_values[j, i]
            if m in to_be_percents:
                val *= 100
                valstr = f'{int(val)}%'
            elif m == DISTANCE_METRIC:
                valstr = f'{val:.1E}'
            elif m == 'scoliosis':
                valstr = f'{val:.1f}'
            else:
                valstr = f'{val:.2f}'
            valstrs[j][i] = valstr
    seaborn.heatmap(
        data=row_normalized,
        vmin=0, vmax=1,
        cmap='Blues',
        annot=numpy.asarray(valstrs),
        cbar=False,
        fmt='', annot_kws=annot_kws, ax=ax)
    ax.set_title('8 wpi Cluster Averages', fontsize=14)
    ax.set_yticks(numpy.arange(len(y_metrics)) + 0.5)
    ax.set_yticklabels(
        [metric_to_printable(m) for m in y_metrics],
        fontsize=fontsizes, rotation='horizontal')
    ax.set_xticks(numpy.arange(len(fish_clusters)) + 0.5)
    ax.set_xticklabels(cluster_labels, fontsize=12)
    print('Saving', plot_fpath)
    mpl_helpers.save_fig(fig, plot_fpath, include_svg=True)
    plt.close(fig)
    return fish_clusters

def plot_metric_tca(fprefix, savedir, group, tensor, t_index, model):
    plot_tca_component_result(fprefix, savedir, group, tensor, t_index, model)
    plot_tca_clustering(fprefix, savedir, group, tensor, t_index, model)

def select_from_ensemble(ensemble: tensortools.Ensemble, max_rank: int) -> tensortools.Ensemble:
    en = tensortools.Ensemble(
        ensemble._nonneg,
        ensemble._fit_method,
        ensemble._fit_options)
    en.results = {
        r: ensemble.results[r]
        for r in ensemble.results if r <= max_rank
    }
    return en

def load_ensembles_and_best_model_of_rank(
        ensembles_dir, group, rank_for_best, overwrite_cache):
    ''' Load ensembles of models, or calculate and save them if necessary.
    '''
    best_model = None
    ensembles = {}
    group_str = group if group is not None else 'all'
    fname = '_'.join([
        f'{group_str}',
        f'{MAX_RANKS_TO_PRECALCULATE}maxrank',
        f'{NUM_ENSEMBLE_REPS}reps_ensembles.pickle'])
    ensembles_path = ensembles_dir / fname
    with CacheContext(ensembles_path) as cache_handle:
        ensembles = cache_handle.getContents(default_val={})
        if not ensembles or overwrite_cache:
            _tensor, _t_index = get_metric_tensor(
                group,
                sorted(FDM.get_available_assay_labels()),
                ordered_fish_names=get_fish_names_ordered_by_metric(8, BRIDGING_METRIC),
                ordered_metrics=get_ordered_metrics())
            if not _tensor.size:
                return ensembles, best_model
            _tensor, _t_index = reorder_tensor_as_fish_assay_metrics(_tensor, _t_index)
            ensembles['tensor'] = _tensor
            ensembles['t_index'] = _t_index
            for m in ENSEMBLE_METHODS:
                ensembles[m] = tensortools.Ensemble(fit_method=m, fit_options=dict(tol=1e-4))
                ensembles[m].fit(
                    ensembles['tensor'],
                    ranks=range(1, MAX_RANKS_TO_PRECALCULATE+1),
                    replicates=NUM_ENSEMBLE_REPS)
        if not isinstance(ensembles['t_index'], dict):
            ensembles['t_index'] = ensembles['t_index'].as_dict()
        cache_handle.saveContents(ensembles)
    ensembles['t_index'] = TensorIndex.from_dict(ensembles['t_index'])
    if rank_for_best is not None:
        models = ensembles['cp_als'].results[rank_for_best]
        models = sorted(
            models,
            key=lambda m: get_reconstruction_error(m.factors.full(), ensembles['tensor']))
        best_model = models[0]
    return ensembles, best_model

def get_ensemble_to_plot(ensembles: dict, method: str, max_rank: list) -> tensortools.Ensemble:
    ensemble_to_plot = ensembles[method]
    if max_rank is not None:
        ensemble_to_plot = select_from_ensemble(ensemble_to_plot, max_rank)
    return ensemble_to_plot

def check_multiple_tca_models(
        savedir: pathlib.Path,
        group: str,
        max_rank: int,
        overwrite_cache: bool=False,
        ensembles_dir: pathlib.Path=None,
        plot_factors: bool=True):
    ''' Mostly identical to
    https://github.com/ahwillia/tensortools/blob/main/examples/cpd_ensemble.py
    Methods:
        ncp_hals nonnegative decomp
        ncp_bcd  nonnegative decomp
        cp_als   unrestricted decomp
        mcp_als  not sure
    '''
    # Fit ensembles of tensor decompositions.
    ensembles, _ = load_ensembles_and_best_model_of_rank(
        ensembles_dir, group, None, overwrite_cache)

    # Plotting options for the unconstrained and nonnegative models.
    diagnostic_plot_options = {
        'cp_als': {
            'line_kw': {
                'color': 'black',
                'label': 'cp_als'
            },
            'scatter_kw': {
                'color': 'black',
                's': 1
            }
        },
        'ncp_hals': {
            'line_kw': {
                'color': 'blue',
                'alpha': 0.5,
                'label': 'ncp_hals'
            },
            'scatter_kw': {
                'color': 'blue',
                'alpha': 0.5,
                's': 1
            }
        },
        'ncp_bcd': {
            'line_kw': {
                'color': 'red',
                'alpha': 0.5,
                'label': 'ncp_bcd'
            },
            'scatter_kw': {
                'color': 'red',
                'alpha': 0.5,
                's': 1
            }
        }
    }

    factor_plot_options = {
        'cp_als': {
            'line_kw': {
                'color': 'black',
                'label': 'cp_als'
            },
            'scatter_kw': {
                'color': 'black',
                's': 1
            },
            'bar_kw': {
                'color': 'black'
            }
        },
        'ncp_hals': {
            'line_kw': {
                'color': 'blue',
                'alpha': 0.5,
                'label': 'ncp_hals'
            },
            'scatter_kw': {
                'color': 'blue',
                'alpha': 0.5,
                's': 1
            },
            'bar_kw': {
                'color': 'blue',
                'alpha': 0.5
            }
        },
        'ncp_bcd': {
            'line_kw': {
                'color': 'red',
                'alpha': 0.5,
                'label': 'ncp_bcd'
            },
            'scatter_kw': {
                'color': 'red',
                'alpha': 0.5,
                's': 1
            },
            'bar_kw': {
                'color': 'red',
                'alpha': 0.5
            }
        }
    }

    # Plot similarity and error plots.
    diag_fig, axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*0.4, FIGURE_ROW_HEIGHT * 0.6))
    for m in ENSEMBLE_METHODS:
        ensemble_to_plot = get_ensemble_to_plot(ensembles, m, max_rank)
        # plot reconstruction error as a function of num components.
        tensortools.plot_objective(ensemble_to_plot, ax=axs[0], **diagnostic_plot_options[m])
        # plot model similarity as a function of num components.
        tensortools.plot_similarity(ensemble_to_plot, ax=axs[1], **diagnostic_plot_options[m])
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylim(0, None)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].legend()
    diag_fig.tight_layout()
    group_str = group if group is not None else 'all'
    name_base = f'{group_str}_{max_rank}maxrank_{NUM_ENSEMBLE_REPS}reps_'
    mpl_helpers.save_fig(diag_fig, savedir / f'{name_base}diagnostics.png')
    plt.close(diag_fig)

    if plot_factors:
        factor_fig, axs = plt.subplots(2, 3)
        for m in ENSEMBLE_METHODS:
            ensemble_to_plot = get_ensemble_to_plot(ensembles, m, max_rank)
            # Plot the low-d factors for an example model,
            # e.g. rank-2, first optimization run / replicate.
            num_components = 2
            replicate = 0
            tensortools.plot_factors(
                ensemble_to_plot.factors(num_components)[replicate],
                fig=factor_fig,
                plots=['bar', 'line', 'bar'],
                **factor_plot_options[m])  # plot the low-d factors
        mpl_helpers.save_fig(factor_fig, savedir / f'{name_base}factors.png')
        plt.close(factor_fig)
    return ensembles

def arr_to_csv_line(arr):
    return ','.join([
        f'"{x}"' if isinstance(x, str) else f'{x}' for x in arr
    ])

def save_model(model, t_index: TensorIndex, outpath: pathlib.Path):
    ''' Save the factors as pseudo csv
    '''
    with open(outpath, 'wt') as fh:
        for prop in range(model.factors.ndim):
            fh.write(f'{t_index.names[prop]}\n')
            fh.write(arr_to_csv_line(t_index.index_values[prop]).replace('\n',' '))
            fh.write('\n')
        for comp in range(model.factors.rank):
            fh.write(f'\tTC{comp + 1}\n')
            for prop in range(model.factors.ndim):
                fh.write(f'{t_index.names[prop]}\n')
                fh.write(arr_to_csv_line(model.factors[prop][:, comp]))
                fh.write('\n')

def main(savedir: pathlib.Path):
    available_assays = FDM.get_available_assay_labels()
    if len(available_assays) < 2:
        return
    group = None
    chosen_rank = 7
    savedir.parent.mkdir(exist_ok=True, parents=False)
    savedir.mkdir(exist_ok=True, parents=False)
    ensembles_dir = savedir / 'ensembles'
    ensembles_dir.mkdir(exist_ok=True, parents=False)
    ensembles, best_model = load_ensembles_and_best_model_of_rank(
        ensembles_dir, group, chosen_rank, overwrite_cache=False)
    if ensembles is None or 't_index' not in ensembles or not ensembles['t_index'].nfish:
        return
    save_model(
        best_model,
        ensembles['t_index'],
        savedir / f'best_model_rank_{best_model.factors.rank}.csv')
    cluster_fpath = savedir / 'fish_clustered_by_tca.txt'
    plotpath = savedir / 'fish_cluster_outcomes.png'
    fish_clusters = plot_tca_fish_clustering(
        plotpath, cluster_fpath, ensembles['t_index'], best_model, all_outcomes=False)
    plotpath = savedir / 'fish_cluster_all_outcomes.png'
    fish_clusters = plot_tca_fish_clustering(
        plotpath, cluster_fpath, ensembles['t_index'], best_model, all_outcomes=True)
    check_multiple_tca_models(
        savedir, group, 90, overwrite_cache=False, ensembles_dir=ensembles_dir, plot_factors=False)
    check_multiple_tca_models(
        savedir, group, 10, overwrite_cache=False, ensembles_dir=ensembles_dir, plot_factors=False)
    plot_tiny_tca_component_result(
        'TCA_tiny_', savedir, group, ensembles['tensor'], ensembles['t_index'],
        best_model, fish_clusters, False)
    plot_tiny_tca_component_result(
        'rank1_TCA_tiny_', savedir, group, ensembles['tensor'], ensembles['t_index'],
        ensembles['cp_als'].results[1][0], fish_clusters, False)
