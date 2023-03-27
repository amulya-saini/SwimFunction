''' Small helpers for matplotlib and plotting in general
'''
from typing import Tuple
import pathlib

from matplotlib import pyplot as plt, gridspec
import numpy
import pandas
import seaborn

from swimfunction.plotting.constants import AXIS_TICKSIZE, DOTSIZE

class SingleColorPalette(dict):
    def __init__(self, single_color):
        self.color = single_color
    def __getitem__(self, __key) -> str:
        return self.color

def get_cmap_values(cmap: str, nvals: int) -> list:
    ''' Get values from a colormap
    If qualitative cmap:
        in order
        nvals must be no larger than the max number of colors
    If continuous colormap:
        spaced evenly across the map
        nvals can be any positive non-zero size
    '''
    if nvals == 1:
        it = [0]
    else:
        it = numpy.linspace(0, 1, num=nvals)
    if cmap in ['tab20', 'tab20b', 'tab20c']:
        it = numpy.arange(20)[:nvals]
    if cmap in ['Paired', 'Set3']:
        it = numpy.arange(12)[:nvals]
    if cmap in ['tab10']:
        it = numpy.arange(10)[:nvals]
    if cmap in ['Pastel1', 'Set1']:
        it = numpy.arange(9)[:nvals]
    if cmap in ['Pastel2', 'Accent', 'Dark2', 'Set2']:
        it = numpy.arange(8)[:nvals]
    return [plt.get_cmap(cmap)(x) for x in it]

def df_to_group_palette(df, soft_colors: bool):
    ''' Map from group to color. Prefers M: blue, F: red (lighter if soft_colors=False),
    otherwise uses tab10 cmap for soft_colors=False,
    or the lighter odd-numbered set of tab20 cmap for soft_colors=True.
    '''
    if soft_colors:
        colors = get_cmap_values('tab20', 20)[1::2]
    else:
        colors = get_cmap_values('tab10', 10)
    group_palette = {
        group: colors[i] for i, group in enumerate(df['group'].unique())
    }
    # Prefer blue and red for sexes, otherwise use tab10
    if soft_colors:
        group_palette.update({'M': '#aaaaff', 'F': '#ffaaaa'})
    else:
        group_palette.update({'M': 'blue', 'F': 'red'})
    return group_palette

def plot_heatmap(
        r_mat: list,
        p_mat: list,
        x_labels: list,
        y_labels: list,
        title: str,
        numbersize=5,
        titlesize=6,
        ticksize=5,
        tickrotation=90,
        left_triangle_only=False,
        right_triangle_only=False,
        ax=None,
        **kwargs):
    '''Plots a heatmap given by the matrix.
    Plots on the axis if provided, otherwise a new Matplotlib figure.

    Parameters
    ----------
    r_mat : list
        Matrix of r values (correlation coefficients)
    p_mat : list
        Matrix of p values (significance)
    x_labels : list
        Horizontal human-readable string labels (will become tick labels)
    y_labels : list
        Vertical human-readable string labels (will become tick labels)
    title : str
        title of the figure
    numbersize : int
    titlesize : int
    ticksize : int
    tickrotation : int
    ax
        Pyplot axis to use for plotting
    '''
    cmap = 'RdBu'
    if ax is None:
        _fig, ax = plt.subplots()
    mask = kwargs.pop('mask', numpy.zeros_like(r_mat))
    if left_triangle_only:
        for i in range(len(mask)):
            mask[i, i:] = 1
    if right_triangle_only:
        for i in range(len(mask)):
            mask[i, :i] = 1
    seaborn.heatmap(
        r_mat,
        annot=r_mat,
        fmt='.2f',
        square=True,
        annot_kws=dict(fontsize=numbersize),
        cbar=False,
        # cbar_kws=dict(location='right', use_gridspec=False),
        cmap=cmap,
        center=0,
        mask=mask,
        ax=ax, **kwargs) #, linewidths=.5)
    # ax.collections[0].colorbar.ax.tick_params(labelsize=numbersize)
    ax.set_xticklabels(x_labels, fontsize=ticksize, rotation=tickrotation)
    ax.set_yticklabels(y_labels, fontsize=ticksize, rotation=0)
    ax.set_title(title, fontsize=titlesize)
    return ax

def plot_clustermap(
        df: pandas.DataFrame,
        annot: pandas.DataFrame,
        title: str,
        labels: list=None,
        numbersize=5,
        titlesize=6,
        ticksize=5,
        tickrotation=90,
        **kwargs):
    '''Plots a clustermap given by the matrix.
    Plots on a new Matplotlib figure.

    Parameters
    ----------
    df : pandas.DataFrame
        Matrix of values
    annot: pandas.DataFrame
    title : str
        title of the figure
    labels: list, default=None
    numbersize : int
    titlesize : int
    ticksize : int
    tickrotation : int
    kwargs : dict
        Passed on to seaborn.clustermap
    '''
    new_df = False
    if isinstance(df, numpy.ndarray):
        df = pandas.DataFrame(df, index=labels, columns=labels)
        new_df = True
    clustermap = seaborn.clustermap(
        data=df,
        fmt='.2f',
        annot_kws=dict(fontsize=numbersize),
        annot=annot,
        **kwargs)

    if not new_df and labels is not None:
        clustermap.ax_heatmap.set_xticklabels(labels, fontsize=ticksize, rotation=tickrotation)
        clustermap.ax_heatmap.set_yticklabels(labels, fontsize=ticksize, rotation=0)
    clustermap.figure.suptitle(title, fontsize=titlesize)
    return clustermap

def get_subplots_absolute_sized_axes(
        nrows: int, ncols: int,
        figsize: Tuple[float, float],
        axis_width: float, axis_height: float,
        sharex: bool=False, sharey: bool=False) -> Tuple[plt.Figure, numpy.ndarray]:
    ''' Create axes with exact sizes.

    Spaces axes as far from each other and the figure edges as possible
    within the grid defined by nrows, ncols, and figsize.

    Allows you to share y and x axes, if desired.
    '''
    fig = plt.figure(figsize=figsize)
    figwidth, figheight = figsize
    # spacing on each left and right side of the figure
    h_margin = (figwidth - (ncols * axis_width)) / figwidth / ncols / 2
    # spacing on each top and bottom of the figure
    v_margin = (figheight - (nrows * axis_height)) / figheight / nrows / 2
    row_addend = 1 / nrows
    col_addend = 1 / ncols
    inner_ax_width = axis_width / figwidth
    inner_ax_height = axis_height / figheight
    axes = []
    sharex_ax = None
    sharey_ax = None
    for row in range(nrows):
        bottom = (row * row_addend) + v_margin
        for col in range(ncols):
            left = (col * col_addend) + h_margin
            if not axes:
                axes.append(fig.add_axes(
                    [left, bottom, inner_ax_width, inner_ax_height]))
                if sharex:
                    sharex_ax = axes[0]
                if sharey:
                    sharey_ax = axes[0]
            else:
                axes.append(fig.add_axes(
                    [left, bottom, inner_ax_width, inner_ax_height],
                    sharex=sharex_ax, sharey=sharey_ax))
    return fig, numpy.flip(numpy.asarray(list(axes)).reshape((nrows, ncols)), axis=0)

def get_figure_cols_sharex(nrows, ncols, figsize):
    gs = gridspec.GridSpec(nrows, ncols)
    fig = plt.figure(figsize=figsize)
    axs = [[
        fig.add_subplot(gs[0, i])
        for i in range(ncols)
    ]]
    for i in range(1, nrows):
        row = []
        for j in range(ncols):
            row.append(fig.add_subplot(gs[i, j], sharex=axs[0][j]))
        axs.append(row)
    return fig, numpy.asarray(axs)

def fig_to_img(fig):
    fig.canvas.draw()
    img = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img

def save_fig(fig, path: pathlib.Path, include_svg=True):
    ''' Save a figure as png and svg
    '''
    fig.savefig(path.with_suffix('.png'), dpi=300, format='png')
    if include_svg:
        fig.savefig(path.with_suffix('.svg'), dpi=300, format='svg')

# Setting plot properties

def same_axis_lims(
        axs: list,
        same_x_lims: bool=True,
        same_y_lims: bool=True,
        same_z_lims: bool=False):
    ''' Extends all axes to the max xlims and ylims.
    '''
    xlim = [
        min(ax.get_xlim()[0] for ax in axs),
        max(ax.get_xlim()[1] for ax in axs)
    ]
    ylim = [
        min(ax.get_ylim()[0] for ax in axs),
        max(ax.get_ylim()[1] for ax in axs)
    ]
    if same_z_lims:
        zlim = [
            min(ax.get_zlim()[0] for ax in axs),
            max(ax.get_zlim()[1] for ax in axs)
    ]
    for ax in axs:
        if same_x_lims:
            ax.set_xlim(xlim)
        if same_y_lims:
           ax.set_ylim(ylim)
        if same_z_lims:
           ax.set_zlim(zlim)

def set_axis_tick_params(ax, labelsize=AXIS_TICKSIZE, set_z=False):
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    if set_z:
        ax.tick_params(axis='z', labelsize=labelsize)

def shrink_lines_and_dots(ax, linewidth=3, dotsize=DOTSIZE):
    for line in ax.get_lines():
        line._linewidth = linewidth
        line._markersize = dotsize

if __name__ == '__main__':
    plt.figure()
    for n in range(2, 11):
        plt.scatter(range(n), [n] * n, c=get_cmap_values('viridis', n))
    plt.show()