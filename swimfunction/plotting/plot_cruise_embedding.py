''' Plot like Figure 4
'''

import pathlib
import numpy
import seaborn
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from swimfunction.plotting.constants import FIGURE_WIDTH, FIGURE_ROW_HEIGHT
from swimfunction.plotting.swim_specific_helpers import WPI_OPTIONS, WPI_COLORS, assay_tick_labels
from swimfunction.plotting.eigenfish import EigenfishPlotter
from swimfunction.pose_processing import pose_filters
from swimfunction.data_access import data_utils, PoseAccess
from swimfunction.data_models.Fish import Fish
from swimfunction.trajectories.cruise_embedding import cruise_embedding
from swimfunction import progress, FileLocations

INCLUDE_SCOLIOTIC_FISH = False
N_NEIGHBORS = 20
N_COMPONENTS = 20

class EmbeddingPlotter:
    def __init__(self, savedir: pathlib.Path):
        strategy = cruise_embedding.UmapStrategy(n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS)
        self.savedir = savedir
        self.interface = cruise_embedding.EmbeddingInterface(
            include_scoliotic_fish=INCLUDE_SCOLIOTIC_FISH,
            all_fish_names=FDM.get_available_fish_names(),
            strategy=strategy)

    def plot_exemplars_PCA(self, group):
        '''
        a) PCA of exemplars preinj and 1wpi
        '''
        if not self.interface.embedding.size:
            return
        names = self.interface.fish_names
        if group is not None:
            names = [n for n in names if data_utils.fish_name_to_group(n) == group]
        fig, ax = plt.subplots(figsize=(1.5, 1.5), subplot_kw=dict(projection='3d'))
        full_pca_result = data_utils.calculate_pca(
            feature=EigenfishPlotter._FEATURE, force_recalculate=False, verbose=False)
        progress.init(len(names))
        for name in names:
            progress.increment(name, len(names))
            fish = Fish(name).load()
            for wpi in [-1, 1]:
                if wpi not in fish.swim_keys():
                    continue
                poses = PoseAccess.get_feature_from_assay(
                    fish[wpi], 'smoothed_angles', pose_filters.BASIC_FILTERS, keep_shape=True)
                exemplar_episodes = self.interface.get_exemplar_indices(name, wpi)
                for index_arr in exemplar_episodes:
                    burst = full_pca_result.decompose(poses[index_arr])
                    ax.plot(
                        burst[:, 0], burst[:, 1], zs=burst[:, 2],
                        linewidth=0.1 , color=WPI_COLORS[wpi], alpha=0.3, clip_on=False)
        progress.finish()
        ax.set_title(f'{len(names)} fish', fontsize=14)
        ax.view_init(elev=45, azim=45)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.set_zlabel('PC3', fontsize=8)
        mpl_helpers.shrink_lines_and_dots(ax, linewidth=0.2)
        mpl_helpers.set_axis_tick_params(ax, labelsize=8, set_z=True)
        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-0.4, 0.4)
        ax.set_zlim(-0.3, 0.2)
        group_str = f'{group}' if group is not None else ''
        mpl_helpers.save_fig(fig, self.savedir / f'{group_str}_exemplars_pca.png')
        plt.close(fig)

    def plot_exemplars_umap_scatter(self, group):
        '''
        b) high bridging
        c) low bridging
        '''
        if not self.interface.embedding.size:
            return
        fig, axs = plt.subplots(1, 2, figsize=(4, 1.5), subplot_kw=dict(projection='3d'))
        # names with bridging
        names = [
            n for n in self.interface.fish_names
            if FDM.get_final_percent_bridging(n, missing_val=None) is not None]
        # names in group
        if group is not None:
            names = [n for n in names if data_utils.fish_name_to_group(n) == group]
        # names separated by bridging
        low_names = [n for n in names if FDM.get_final_percent_bridging(n, missing_val=0) <= 0.2]
        high_names = [n for n in names if FDM.get_final_percent_bridging(n, missing_val=0) > 0.2]

        for wpi in tqdm(WPI_OPTIONS):
            label = f'{wpi}wpi' if wpi > 0 else 'control'
            best_embedding = self.interface.get_exemplars_embedded(high_names, wpi)
            worst_embedding = self.interface.get_exemplars_embedded(low_names, wpi)
            if not len(best_embedding):
                continue
            scatter_kws = dict(linewidths=0, s=1, label=label, color=WPI_COLORS[wpi], clip_on=False)
            axs[0].scatter(
                best_embedding[:, 0], best_embedding[:, 1], zs=best_embedding[:, 2],
                **scatter_kws)
            axs[1].scatter(
                worst_embedding[:, 0], worst_embedding[:, 1], zs=worst_embedding[:, 2],
                **scatter_kws)

        axs[0].set_title(f'High Bridging (> 20%) {len(high_names)} fish', fontsize=14)
        axs[1].set_title(f'Low Bridging (≤ 20%) {len(low_names)} fish', fontsize=14)
        for ax in axs:
            ax.view_init(azim=166, elev=-57)
            ax.set_xlabel('UMAP1', fontsize=8)
            ax.set_ylabel('UMAP2', fontsize=8)
            ax.set_zlabel('UMAP3', fontsize=8)
            mpl_helpers.shrink_lines_and_dots(ax, linewidth=2)
            mpl_helpers.set_axis_tick_params(ax, labelsize=8, set_z=True)
        group_str = f'{group}' if group is not None else ''
        mpl_helpers.save_fig(fig, self.savedir / f'{group_str}_exemplars_umap_scatter.png')
        plt.close(fig)

    def plot_exemplars_umap(self, group):
        '''
        d/e) decomposed cruise space (gaussian kde, per week)
        '''
        if not self.interface.embedding.size:
            return
        fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_ROW_HEIGHT*2), constrained_layout=True)
        gs = fig.add_gridspec(nrows=10, ncols=6, hspace=0.4)
        axs = numpy.asarray([
            fig.add_subplot(gs[5:7, 0]), # Top
            fig.add_subplot(gs[5:7, 1]), # Top
            fig.add_subplot(gs[5:7, 2]), # Top
            fig.add_subplot(gs[5:7, 3]), # Top
            fig.add_subplot(gs[5:7, 4]), # Top
            fig.add_subplot(gs[5:7, 5]), # Top
            fig.add_subplot(gs[7:, 0]), # Bottom
            fig.add_subplot(gs[7:, 1]), # Bottom
            fig.add_subplot(gs[7:, 2]), # Bottom
            fig.add_subplot(gs[7:, 3]), # Bottom
            fig.add_subplot(gs[7:, 4]), # Bottom
            fig.add_subplot(gs[7:, 5]) # Bottom
        ])
        tops = axs[:6]
        bottoms = axs[6:]

        # names with bridging
        names = [
            n for n in self.interface.fish_names
            if FDM.get_final_percent_bridging(n, missing_val=None) is not None]
        # names in group
        if group is not None:
            names = [n for n in names if data_utils.fish_name_to_group(n) == group]
        # names separated by bridging
        low_names = [n for n in names if FDM.get_final_percent_bridging(n, missing_val=0) <= 0.2]
        high_names = [n for n in names if FDM.get_final_percent_bridging(n, missing_val=0) > 0.2]

        best_means, best_stds = self._plot_weekly_kde_return_means_stds(high_names, bottoms[:3])
        worst_means, worst_stds = self._plot_weekly_kde_return_means_stds(low_names, bottoms[3:])
        self._plot_weekly_means_stds(best_means, best_stds, tops[:3])
        self._plot_weekly_means_stds(worst_means, worst_stds, tops[3:])

        tops[1].set_title(f'High Bridging', fontsize=14)
        tops[4].set_title(f'Low Bridging', fontsize=14)

        for ax in numpy.concatenate((tops, bottoms)):
            mpl_helpers.set_axis_tick_params(ax, labelsize=8)
            mpl_helpers.shrink_lines_and_dots(ax, linewidth=2)

        for a, b in zip(bottoms[:3], bottoms[3:]):
            mpl_helpers.same_axis_lims([a, b], same_y_lims=False)
        for t, b in zip(tops, bottoms):
            t.set_xlim(*b.get_xlim())
        group_str = f'{group}' if group is not None else ''
        mpl_helpers.save_fig(fig, self.savedir / f'{group_str}_exemplars_umap_kde.png')
        plt.close(fig)

    def _plot_weekly_kde_return_means_stds(self, names, axs):
        means = []
        stds = []
        for wpi in tqdm(WPI_OPTIONS):
            label = f'{wpi}wpi' if wpi > 0 else 'control'
            embedding = self.interface.get_exemplars_embedded(names, wpi)
            if not len(embedding):
                continue
            means.append(numpy.mean(embedding, axis=0))
            stds.append(numpy.std(embedding, axis=0))
            seaborn.kdeplot(
                x=embedding[:, 0], color=WPI_COLORS[wpi],
                ax=axs[0], fill=True, alpha=0.1, label=label)
            seaborn.kdeplot(
                x=embedding[:, 1], color=WPI_COLORS[wpi],
                ax=axs[1], fill=True, alpha=0.1, label=label)
            seaborn.kdeplot(
                x=embedding[:, 2], color=WPI_COLORS[wpi],
                ax=axs[2], fill=True, alpha=0.1, label=label)
        axs[0].set_xlabel('UMAP1', fontsize=8)
        axs[1].set_xlabel('UMAP2', fontsize=8)
        axs[2].set_xlabel('UMAP3', fontsize=8)
        axs[0].set_ylabel('Density', fontsize=8)
        means = numpy.asarray(means)
        stds = numpy.asarray(stds)
        return means, stds

    def _plot_weekly_means_stds(self, means, stds, axs):
        for i in range(len(WPI_OPTIONS)-1):
            barkws = dict(
                solid_capstyle='butt',
                elinewidth=0.75,
                color=WPI_COLORS[i+1],
                ecolor='black',#WPI_COLORS[i]
            )
            axs[0].errorbar(
                means[i:i+2, 0],
                [i, i+1],
                xerr=stds[i:i+2, 0],
                **barkws)
            axs[1].errorbar(
                means[i:i+2, 1],
                [i, i+1],
                xerr=stds[i:i+2, 1],
                **barkws)
            axs[2].errorbar(
                means[i:i+2, 2],
                [i, i+1],
                xerr=stds[i:i+2, 2],
                **barkws)
        for i, ax in enumerate(axs):
            assay_tick_labels(ax, x_axis=False)
            ax.vlines(means[3, i], 0, 9, colors='black', linestyles='--', linewidth=1)
        axs[0].set_xlabel('Mean UMAP1', fontsize=8)
        axs[1].set_xlabel('Mean UMAP2', fontsize=8)
        axs[2].set_xlabel('Mean UMAP3', fontsize=8)

    def embedding_to_csv(self):
        if not self.interface.embedding.size:
            return
        fpath = FileLocations.get_csv_output_dir() / 'embedded_cruise_exemplar_umap_values.csv'
        pandas.DataFrame(
            self.interface.embedding,
            index=pandas.MultiIndex.from_tuples(self.interface.keys, names=['fish', 'assay'])
        ).sort_index().to_csv(fpath, mode='w')

def main():
    plotter = EmbeddingPlotter(
        FileLocations.get_cruise_embedding_dir(INCLUDE_SCOLIOTIC_FISH))
    plotter.embedding_to_csv()
    for group in FDM.get_groups() + [None]:
        plotter.plot_exemplars_PCA(group)
        plotter.plot_exemplars_umap_scatter(group)
        plotter.plot_exemplars_umap(group)

if __name__ == '__main__':
    FileLocations.parse_default_args()
    main(FileLocations.get_cruise_embedding_dir(INCLUDE_SCOLIOTIC_FISH))
