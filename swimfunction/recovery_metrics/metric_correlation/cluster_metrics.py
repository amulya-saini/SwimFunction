''' Pairwise correlation of DataFrame columns
with cluster significance tested.
'''
import pathlib
import numpy
import pandas
import subprocess
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from scipy import spatial, cluster as spcluster, stats as spstats

def i_to_str(i: int) -> str:
    ''' Integer to unique string of alpha characters
    '''
    return ''.join([chr(i % 26 + 65)] * ((i // 26) + 1))

def get_rsafe_dict(vals):
    ''' R has requirements about what are and are not allowed in column names.
    This solves any issue by defining a dictionary from columns to "R-safe" letters.
    '''
    return {v: i_to_str(i) for i, v in enumerate(vals)}

def from_rsafe(rsafe_val, val_to_rsafe: dict):
    ''' R has requirements about what are and are not allowed in column names.
    This converts our "R-safe" letters back to original column names.
    '''
    inverse = {v:k for k,v in val_to_rsafe.items()}
    return inverse[rsafe_val]

def to_rsafe(val, val_to_rsafe: dict):
    ''' R has requirements about what are and are not allowed in column names.
    This solves any issue by making everything "R-safe" letters.
    '''
    return val_to_rsafe[val]

def _without_R_clustering_and_significance(
        df: pandas.DataFrame,
        distance_metric: str,
        linkage_method: str):
    ''' Plot with heirarchical clustering,
    but no cluster signficance tested.
    '''
    distances = None
    if distance_metric == 'precomputed':
        distances = spatial.distance.squareform(df.values)
    elif distance_metric == 'cor':
        distances = spatial.distance.pdist(df.T, metric=lambda x, y: 1 - spstats.spearmanr(x, y)[0])
    else:
        distances = spatial.distance.pdist(df.T, metric=distance_metric)
    linkage = spcluster.hierarchy.linkage(distances, method=linkage_method)
    return (
        linkage,
        numpy.full(df.shape[1], numpy.nan),
        pandas.DataFrame(
            spatial.distance.squareform(distances),
            index=df.columns,
            columns=df.columns),
        df.columns)

def get_clustering_and_significance(
        df: pandas.DataFrame,
        distance_metric: str,
        linkage_method: str):
    '''
    Parameters
    ----------
    df
    distance_metric
        precomputed, cor (interpreted as spearman in my forked repo), euclidean,
    '''
    if df.shape[0] < 10 or df.shape[1] < 10:
        print(' '.join([
            'Data frame too small for sigclust2,',
            'so linkage will be returned without p-values.',
            'This is safest with metric "precomputed".'
        ]))
        return _without_R_clustering_and_significance(df, distance_metric, linkage_method)
    tmp_dir = pathlib.Path('/tmp/swimfunction_for_sigclust2')
    tmp_dir.mkdir(exist_ok=True)
    metric_csv = tmp_dir / 'CLUSTERING_all_metrics_scaled.csv'
    _df = df.copy()
    if isinstance(_df.index[0], tuple) and len(_df.index[0]) == 2:
        _df.index = [f'{a[0]}_{a[1]}' for a in _df.index]
    index_dict = get_rsafe_dict(_df.index)
    column_dict = get_rsafe_dict(_df.columns)
    _df.index = [to_rsafe(x, index_dict) for x in _df.index]
    _df.columns = [to_rsafe(x, column_dict) for x in _df.columns]
    _df.to_csv(metric_csv)
    call_path = pathlib.Path(__file__).parent / 'cluster_data_with_stats.R'
    cmd = [
        'Rscript', call_path.as_posix(),
        metric_csv.as_posix(),
        '-m', distance_metric,
        '-l', linkage_method]
    print(' '.join(cmd))
    subprocess.call(cmd)
    linkage_pvals = pandas.read_csv(tmp_dir / 'CLUSTERING_all_metrics_col_linkage.csv', header=0, index_col=0)
    linkage_pvals.index = linkage_pvals.index + len(linkage_pvals.index)
    distances = pandas.read_csv(tmp_dir / 'CLUSTERING_all_metrics_col_dist.csv', header=0, index_col=False)
    distances.columns = [from_rsafe(x, column_dict) for x in distances.columns]
    distances.index = distances.columns
    ordered_labels = pandas.read_csv(tmp_dir / 'CLUSTERING_all_metrics_col_ordered_labels.csv', header=None, index_col=None).values.flatten().tolist()
    ordered_labels = [from_rsafe(x, column_dict) for x in ordered_labels]
    return linkage_pvals.loc[:, ('i1', 'i2', 'heights', 'obs')], linkage_pvals['pvals'], distances, ordered_labels

def annotate_p(clustergrid, Z, cluster_i, label, color, p_vals_on_top: bool):
    ''' Add p-values and color for significant clusters.
    '''
    dend_heights = [numpy.max(j) for j in clustergrid.dendrogram_col.dependent_coord]

    # Get dendrogram arm
    ranked_heights = numpy.empty(Z.shape[0])
    ranked_heights[numpy.argsort(Z['heights'])] = numpy.arange(Z.shape[0])
    target_rank =ranked_heights.tolist().index(Z.index.get_loc(cluster_i))

    dend_heights = [max(x) for x in clustergrid.dendrogram_col.dependent_coord]
    ranked_dend_heights = numpy.empty_like(ranked_heights)
    ranked_dend_heights[numpy.argsort(dend_heights)] = numpy.arange(Z.shape[0])
    dend_i = ranked_dend_heights.tolist().index(target_rank)

    # Color dendrogram arm
    clustergrid.ax_col_dendrogram.plot(
        clustergrid.dendrogram_col.independent_coord[dend_i],
        clustergrid.dendrogram_col.dependent_coord[dend_i],
        linewidth=1,
        color=color)
    # Invert x and y for side.
    clustergrid.ax_row_dendrogram.plot(
        clustergrid.dendrogram_col.dependent_coord[dend_i],
        clustergrid.dendrogram_col.independent_coord[dend_i],
        linewidth=1,
        color=color)
    if p_vals_on_top:
        clustergrid.ax_col_dendrogram.text(
            numpy.mean(clustergrid.dendrogram_col.independent_coord[dend_i]) + 1,
            numpy.max(clustergrid.dendrogram_col.dependent_coord[dend_i]) + 0.1,
            label,
            fontsize=10)
    else:
        clustergrid.ax_row_dendrogram.text(
            numpy.max(clustergrid.dendrogram_col.dependent_coord[dend_i]) + 0.1,
            numpy.mean(clustergrid.dendrogram_col.independent_coord[dend_i]) + 1,
            label,
            fontsize=10)

def plot_clustering_and_significance(
        df: pandas.DataFrame,
        method: str,
        color_annot: pandas.DataFrame=None,
        annot: pandas.DataFrame=None,
        cmap: str='RdBu',
        mask_diagonal: bool=False,
        right_triangle_only: bool=False,
        distance_metric: str='cor',
        p_vals_on_top: bool=True,
        title: str='',
        **kwargs):
    '''
    Parameters
    ----------
    df: pandas.DataFrame
    method: str, "complete"
        linkage method
    color_annot: pandas.DataFrame, None
        Values to use when coloring the heatmap. Default is to use df if distance_metric=precomputed else distance matrix.
    annot: pandas.DataFrame, None
    cmap: str, 'RdBu'
    mask_diagonal: bool, False
    right_triangle_only: bool=False
        NOT YET IMPLEMENTED
    distance_metric: str, "cor"
    p_vals_on_top: bool=True
    title: str, ''
    **kwargs: dict
        Passed to matplotlib_helpers.plot_clustermap
    '''
    print('Running clustering and stats through R')
    Z, p_vals, dist_df, ordered_labels = get_clustering_and_significance(
        df, distance_metric, method)

    mask = kwargs.get('mask', numpy.zeros_like(dist_df.values, dtype=bool))
    if mask_diagonal:
        mask = mask + numpy.identity(dist_df.shape[0], dtype=bool)

    if color_annot is None:
        color_annot = dist_df if distance_metric != 'precomputed' else df

    clustergrid = mpl_helpers.plot_clustermap(
        color_annot,
        annot=dist_df if annot is None else annot,
        title=title,
        row_linkage=Z,
        col_linkage=Z,
        cmap=cmap,
        method=method,
        **kwargs)

    alpha = 0.05
    if numpy.all(~numpy.isnan(p_vals)):
        for i, p in enumerate(p_vals):
            if p <= 1:
                color = 'red' if p < alpha else 'orange'
                label = f'{p:.2E}' if p < alpha else 'n.s.'
                annotate_p(clustergrid, Z, Z.index[i], label, color, p_vals_on_top)

    if right_triangle_only:
        for i in range(mask.shape[0]):
            mask[i, :i] = 1
    return clustergrid

def get_correlation_matrix(df_one_assay_only_metric_columns: pandas.DataFrame):
    ''' Metric pairwise correlation matrix
    Parameters
    ----------
    df_one_assay_only_metric_columns : pandas.DataFrame
        DataFrame's columns must only be metrics
        for example, it cannot have "fish", "assay", or "group.
        Throw those into the index, if you wish.
    '''
    columns = df_one_assay_only_metric_columns.columns
    cormat = pandas.DataFrame(
        numpy.identity(len(columns), dtype=float),
        index=columns, columns=columns)
    for i, j in numpy.asarray(numpy.meshgrid(columns, columns)).T.reshape((-1, 2)):
        if i == j:
            continue
        a, b = df_one_assay_only_metric_columns.loc[:, (i, j)].dropna().T.values
        rs, _p = spstats.spearmanr(a.astype(float), b.astype(float))
        cormat.loc[i, j] = rs
        cormat.loc[j, i] = rs
    return cormat

def plot_spearman_clustermap(
        scaled_df: pandas.DataFrame,
        right_triangle_only: bool=False,
        title: str='', **kwargs):
    '''
    Parameters
    ----------
    scaled_df
    right_triangle_only
    title
    **kwargs : dict
        Passed to plot_clustering_and_significance then to matplotlib_helpers.plot_clustermap
    '''

    predrop_count = scaled_df.shape[0]
    # drop missing values
    scaled_df = scaled_df.dropna()
    if scaled_df.shape[0] < 2:
        return None, None
    r_df = get_correlation_matrix(scaled_df)

    print('Metric matrix shape: ', scaled_df.shape)
    n_dropped = predrop_count - scaled_df.shape[0]
    print(f'Dropout rate: {n_dropped} of {predrop_count} assays dropped.')

    spearman_clustergrid = plot_clustering_and_significance(
        scaled_df, annot=r_df,
        color_annot=r_df,
        distance_metric='cor',
        method='average',
        title=title,
        cmap='RdBu',
        mask_diagonal=True,
        right_triangle_only=right_triangle_only,
        center=0,
        **kwargs)
    return spearman_clustergrid, r_df
