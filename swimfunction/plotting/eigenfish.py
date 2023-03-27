''' Plots all eigenfish oscillating
between each side of the mean.
'''
from swimfunction.data_access import PoseAccess, data_utils
from swimfunction.data_models.PCAResult import PCAResult
from swimfunction.pose_processing import pose_filters, pose_conversion
from matplotlib import gridspec, pyplot as plt
from os import PathLike

import matplotlib as mpl
import imageio
from swimfunction import FileLocations
import numpy
from tqdm import tqdm

DEBUG = print

class EigenfishPlotter:
    ''' Plots oscillating eigenpose gifs
    '''
    _FEATURE = 'smoothed_angles'

    __slots__ = ['fps', 'pca_result', 'nframes']
    def __init__(
            self,
            pca_result: PCAResult = None,
            fps=30):
        self.nframes = 100
        self.fps = fps
        self.pca_result = pca_result
        if self.pca_result is None:
            self.pca_result = data_utils.calculate_pca(
                feature=EigenfishPlotter._FEATURE, force_recalculate=False, verbose=False)

    def setup_fish_ax(self, ax, dim):
        ''' Set limits and features of the axis.
        '''
        var_explained = 100 * self.pca_result.variances[dim] / sum(self.pca_result.variances)
        ax.set_title(f'PC{dim}: {int(var_explained)}%', fontsize=7, loc='left')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-11, 1)
        ax.set_aspect('equal')
        ax.set_axis_off()

    def rotate_pose(self, pose):
        # TODO if have time, make so the fish looks like it's swimming forward.
        return pose

    def plot_fish(self, fish_ax, pose, dim):
        fish_ax.clear()
        pose = self.rotate_pose(pose)
        fish_ax.plot(pose[:, 0], pose[:, 1])
        self.setup_fish_ax(fish_ax, dim)

    def plot_bar(self, bar_ax, possibilities, y):
        bar_ax.clear()
        bar_ax.plot(numpy.zeros_like(possibilities), possibilities, '.', color='blue', markersize=1)
        bar_ax.plot(0, y, '.', color='red', markersize=5)
        bar_ax.get_xaxis().set_visible(False)
        bar_ax.spines['bottom'].set_visible(False)

    def eigenfish_smear(
            self,
            ndims_to_plot,
            max_coefficient,
            eigenshape_xpad=4,
            title='PCA',
            cmap='gray_r',
            var_plot_kws: dict=None,
            fig=None):
        ''' Plots eigenfish gif-like image with blurred fish poses
        Oscillates between three standard deviations (+- 3 * norm_pc)
            from the mean pose.
        '''
        if var_plot_kws is None:
            var_plot_kws = {}
        if fig is None:
            fig = plt.figure(figsize=(4, 2))
        ndims_to_plot = min(ndims_to_plot, self.pca_result.pcs.shape[0])
        nrows = 1
        ncols = 3 + ndims_to_plot
        fig.suptitle(title, fontsize=12)
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
        scree_ax = fig.add_subplot(gs[:, :2])
        eigen_ax = fig.add_subplot(gs[:, 2:])
        plt.subplots_adjust(top=0.7, left=0.15, bottom=0.3)

        vp_kws = dict(legend=False)
        vp_kws.update(var_plot_kws)
        self.pca_result.plot_variance(ax=scree_ax, title=None, **var_plot_kws)

        coeffs = numpy.linspace(0, max_coefficient, num=100)
        rotation_angle = -1 * numpy.pi / 2
        prev_max_x = 0
        eigen_min_y = 0
        eigen_min_x = 0
        for dim in range(ndims_to_plot):
            poses = [pose_conversion.angles_pose_to_points(
                self.pca_result.mean + coeff*self.pca_result.norm_pcs[dim, :],
                rotation_angle=rotation_angle) for coeff in coeffs]
            min_x = min([p[:, 0].min() for p in poses])
            max_x = max([p[:, 0].max() for p in poses])
            x_shift = prev_max_x - min_x + eigenshape_xpad
            for coeff, pose in zip(coeffs, poses):
                eigen_ax.plot(pose[:, 0] + x_shift, pose[:, 1], linewidth=0.5, color=plt.get_cmap(cmap)(coeff / max_coefficient))
            eigen_ax.text(poses[0][0, 0] + x_shift - 2, poses[0][0, 1] + 1, f'PC{dim+1}', fontsize=8)
            prev_max_x = max_x + x_shift
            if pose[:, 0].min() + x_shift < eigen_min_x:
                eigen_min_x = pose[:, 0].min() + x_shift
            eigen_min_y = min(eigen_min_y, pose[:, 1].min())
        eigen_ax.set_ylim(eigen_min_y, 0)
        eigen_ax.set_xlim(eigen_min_x, None)
        eigen_ax.set_aspect('equal')
        eigen_ax.set_axis_off()

        scree_ax.get_yaxis().labelpad = 0
        return fig

    def eigenfish_gif(self):
        ''' Renders the animation.
        Oscillates between three standard deviations (+- 3 * norm_pc)
            from the mean pose.
        '''
        cols_per_plot = 4
        fig = plt.figure(figsize=(8, 4))
        nrows = 2
        ncols = 4
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols * cols_per_plot)
        ndims = self.pca_result.norm_pcs.shape[0]

        rotation_angle = -1 * numpy.pi / 2
        mean_pose = pose_conversion.angles_pose_to_points(
            self.pca_result.mean,
            rotation_angle=rotation_angle)

        counter_to_percent = lambda x: 3 * numpy.sin(numpy.pi * (x / 180))
        counters = numpy.arange(360, step=int(360/self.nframes))
        counter_to_variance = lambda x, dim: counter_to_percent(x) * self.pca_result.variances[dim]
        possibilities = [
            [counter_to_variance(x, d) for x in counters]
            for d in range(ndims)
        ]
        bar_axs = []
        fish_axs = []
        for dim in range(ndims):
            row = dim // ncols
            col = cols_per_plot * (dim % ncols)
            bar_ax = fig.add_subplot(gs[row, col])
            fish_ax = fig.add_subplot(gs[row, col+1:col+cols_per_plot])
            bar_axs.append(bar_ax)
            self.plot_bar(bar_ax, possibilities[dim], 0)
            bar_ax.plot(0, 0, '.', color='red', markersize=5)
            fish_axs.append(fish_ax)
            self.plot_fish(fish_ax, mean_pose, dim)

        fig.tight_layout()

        imgs = []
        for counter in tqdm(counters):
            percent = counter_to_percent(counter)
            for dim in range(ndims):
                pose = pose_conversion.angles_pose_to_points(
                    self.pca_result.mean + percent*self.pca_result.norm_pcs[dim, :],
                    rotation_angle=rotation_angle)
                self.plot_fish(fish_axs[dim], pose, dim)
                y = counter_to_variance(counter, dim)
                self.plot_bar(bar_axs[dim], possibilities[dim], y)
            fig.canvas.draw()
            img = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
            imgs.append(img)
        return imgs

    def save_gif(self, outfile: PathLike):
        ''' Writes the animation to the file.
        '''
        DEBUG('Saving gif')
        outfile = FileLocations.as_absolute_path(outfile)
        imageio.mimsave(outfile, self.eigenfish_gif(), 'GIF', fps=self.fps)
        DEBUG('Done')

def get_pca_result_by_behavior(behavior):
    ''' Get a pca result from a specific behavior.
    '''
    if behavior is None:
        return None
    poses_list, _names, _assays = PoseAccess.get_feature(
        filters=[pose_filters.TEMPLATE_filter_by_behavior(
            behavior, data_utils.AnnotationTypes.all)],
        keep_shape=False)
    return PCAResult().pca(numpy.concatenate(poses_list))

if __name__ == '__main__':
    ARGS = FileLocations.parse_default_args(
        lambda parser: parser.add_argument(
            '-o', '--outfile', type=str, default='~/Desktop/tmp_eigenfish.gif'),
        lambda parser: parser.add_argument(
            '-b', '--behavior',
            help='Symbol of behavior to get. "|" is cruise.', type=str, default='')
    )
    PCA = None
    if ARGS.behavior:
        PCA = get_pca_result_by_behavior(ARGS.behavior)
    else:
        PCA = data_utils.calculate_pca(force_recalculate=False)
    EigenfishPlotter(ARGS.feature, PCA).save_gif(ARGS.outfile)
