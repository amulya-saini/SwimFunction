''' Plot eigenfish for preinjury, 1 wpi, and 8 wpi poses.
'''
import pathlib
from matplotlib import pyplot as plt, gridspec

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import data_utils
from swimfunction.plotting.eigenfish import EigenfishPlotter
from swimfunction.plotting.constants import *
from swimfunction.plotting import matplotlib_helpers as mpl_helpers

def plot_eigenfish(group, assay_label, subfig):
    pca_result = data_utils.calculate_pca(
        group,
        EigenfishPlotter._FEATURE,
        assay_label=assay_label,
        force_recalculate=False,
        verbose=False)
    print(group, assay_label, pca_result.get_npcs_for_required_variance(0.98))
    if subfig is None:
        return
    vp_kws = dict(
        percent_text=False,
        legend=False,
        cumulative_plot_kws=dict(
            color='black',
            linewidth=2,
            markersize=4),
        bar_kws=dict(color='black')
    )
    title_parts = []
    if group is None:
        title_parts.append('All Fish')
    elif group == 'M':
        title_parts.append('Males')
    elif group == 'F':
        title_parts.append('Females')
    if assay_label is None:
        title_parts.append('All Assays')
    elif assay_label == -1:
        title_parts.append('Preinjury')
    else:
        title_parts.append(f'{assay_label} WPI')
    EigenfishPlotter(pca_result).eigenfish_smear(
        5,
        max_coefficient=3,
        title=' '.join(title_parts),
        var_plot_kws=vp_kws,
        fig=subfig)

def big_plot(outdir):
    assays = FDM.get_available_assay_labels()
    print('The eigenfish plot includes preinjury (-1), 1 wpi (1), and 8 wpi (8) assays.')
    print('It also requires groups "M" and "F".')
    groups = FDM.get_available_fish_names_by_group().keys()
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_MAX_HEIGHT-2))
    gs = gridspec.GridSpec(nrows=5, ncols=4)
    plot_eigenfish(None, None, fig.add_subfigure(gs[0, 1:3]))
    plot_eigenfish('M', None, fig.add_subfigure(gs[1, :2]))
    plot_eigenfish('F', None, fig.add_subfigure(gs[1, 2:]))
    if -1 in assays:
        if 'M' in groups:
            plot_eigenfish('M', -1, fig.add_subfigure(gs[2, :2]))
        if 'F' in groups:
            plot_eigenfish('F', -1, fig.add_subfigure(gs[2, 2:]))
    if 1 in assays:
        if 'M' in groups:
            plot_eigenfish('M', 1, fig.add_subfigure(gs[3, :2]))
        if 'F' in groups:
            plot_eigenfish('F', 1, fig.add_subfigure(gs[3, 2:]))
    if 8 in assays:
        if 'M' in groups:
            plot_eigenfish('M', 8, fig.add_subfigure(gs[4, :2]))
        if 'F' in groups:
            plot_eigenfish('F', 8, fig.add_subfigure(gs[4, 2:]))
    mpl_helpers.save_fig(fig, outdir / 'eigenfish.png')
    plt.close(fig)

def report_required_pcs_for_98_percent_variance():
    print('group, assay, n pcs for 98% variance explained')
    for i in (None, -1, 1, 8):
        if i is not None and i not in FDM.get_available_assay_labels():
            continue
        plot_eigenfish(None, i, None)
        for group in FDM.get_available_fish_names_by_group().keys():
            plot_eigenfish(group, i, None)

def main(savedir: pathlib.Path):
    report_required_pcs_for_98_percent_variance()
    big_plot(savedir)
