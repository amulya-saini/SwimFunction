''' Plots
    Box plot length of cruises
    % cruise frames for group
    % rest frames for group
    Behaviors as image
'''
import pathlib
from matplotlib import gridspec, pyplot as plt
import numpy

from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import data_utils, PoseAccess
from swimfunction.data_models import PCAResult
from swimfunction.data_models.Fish import Fish
from swimfunction.data_models.SwimAssay import SwimAssay
from swimfunction.data_access.data_utils import AnnotationTypes
from swimfunction.pose_processing import pose_filters
from swimfunction.global_config.config import config
from swimfunction import loggers
from swimfunction import FileLocations
from swimfunction import progress

BEHAVIOR_COLORS = config.getfloatlist('BEHAVIORS', 'colors')
BEHAVIOR_COLORS_DICT = config.getfloatdict('BEHAVIORS', 'symbols', 'BEHAVIORS', 'colors')
BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')

FEATURE_FOR_PC_QC = 'smoothed_angles'

def plot_behaviors_as_image(behaviors: numpy.ndarray, assay_label, ax=None, maxwidth=1300):
    ''' Make an image where each pixel block is a frame, color is the behavior.
    '''
    width = max(int(numpy.sqrt(behaviors.size)), maxwidth)
    height = int(numpy.ceil(behaviors.size / width))
    img = numpy.full((width*height, 4), 0, dtype=float)
    img[:len(behaviors), :] = numpy.asarray([BEHAVIOR_COLORS_DICT[b] for b in behaviors])
    img = img.reshape((height, width, 4))
    img = numpy.repeat(img, 15, axis=0) # Interleaf duplications for visualization purposes
    for i in range(10, 15):
        img[i::15, :, :3] = 0
    if ax is None:
        _fig, ax = plt.subplots()
    ax.set_title(f'{assay_label}', fontsize=6)
    ax.imshow(img)
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)

def _plot_pc_space(swim: SwimAssay, ax, pca_result: PCAResult.PCAResult):
    ''' (for one assay)
    Plot behavior episode traces in PCA colored by the behavior.
    '''
    poses = PoseAccess.get_feature_from_assay(swim,
        feature=FEATURE_FOR_PC_QC,
        filters=pose_filters.BASIC_FILTERS,
        keep_shape=True)
    decomposed_poses = pca_result.decompose(poses)
    # It displays better in reverse order of behaviors.
    for behavior in BEHAVIORS[::-1]:
        for index_arr in PoseAccess.get_behavior_episodes_from_assay(
                swim, behavior, filters=pose_filters.BASIC_FILTERS):
            burst = decomposed_poses[index_arr]
            ax.plot(burst[:, 0], burst[:, 1], linewidth=1, color=BEHAVIOR_COLORS_DICT[behavior])

def plot_behaviors_in_pc_space(fish: Fish, pca_result: PCAResult.PCAResult, fpath):
    ''' (for all of a fish's assays)
    Plot behavior episode traces in PCA colored by the behavior.
    '''
    fig, axs = plt.subplots(3, 3)
    axs = axs.flatten()
    fig.suptitle(f'{fish.name} in Behavior Space')
    for _wi, wpi in enumerate(WPI_OPTIONS):
        axs[_wi].set_title(wpi)
        if wpi not in fish:
            continue
        _plot_pc_space(fish[wpi], axs[_wi], pca_result)
    if not fpath.parent.exists():
        fpath.parent.mkdir()
    fig.savefig(fpath.as_posix())
    plt.close(fig=fig)

def plot_cruise_rest_behavior_image(fish: Fish, fpath):
    ''' Make the main QC plot with % cruise and % rest compared.
    '''
    fig = plt.figure(tight_layout=True)
    fig.suptitle(f'{fish.name} Predicted Behaviors QC', fontsize=8)
    gs = gridspec.GridSpec(2,2)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[:, 1])
    ]
    axs[0].set_title('Percents', fontsize=6)
    axs[1].set_title('Cruise Lengths', fontsize=6)
    assays = []
    percent_rests = []
    percent_cruises = []
    cruise_lengths = []
    for wpi in fish.swim_keys():
        assays.append(wpi if wpi > 0 else 0)
        behaviors = PoseAccess.get_behaviors(fish[wpi], AnnotationTypes.predicted)
        nframes = len(behaviors)
        percent_cruises.append(100 * (behaviors == BEHAVIORS.cruise).sum() / nframes)
        percent_rests.append(100 * (behaviors == BEHAVIORS.rest).sum() / nframes)
        where_is_cruise = sorted(numpy.where(behaviors == BEHAVIORS.cruise)[0])
        cruise_lengths.append(list(map(
            len,
            PoseAccess.split_into_consecutive_indices(where_is_cruise))))
    axs[0].set_title('% Cruise and % Rest')
    axs[0].plot(
        assays, percent_cruises,
        ls='-', marker='.', label='% cruise',
        color=BEHAVIOR_COLORS_DICT[BEHAVIORS.cruise])
    axs[0].plot(
        assays, percent_rests,
        ls='-', marker='.', label='% rest',
        color=BEHAVIOR_COLORS_DICT[BEHAVIORS.rest])
    axs[0].set_xticks(range(len(WPI_OPTIONS)))
    axs[0].set_xticklabels(WPI_OPTIONS)
    axs[0].set_ylim(0, 100)
    axs[0].set_ylabel('Percent')
    axs[0].set_xlabel('Assay')
    axs[0].legend()
    axs[1].set_title('Cruise Lengths')
    axs[1].boxplot(cruise_lengths, positions=assays, widths=0.3, showfliers=True)
    axs[1].set_xticks(range(len(WPI_OPTIONS)))
    axs[1].set_xticklabels(WPI_OPTIONS)
    axs[1].set_ylabel('Frame Count')
    axs[1].set_xlabel('Assay')
    random_assay = numpy.random.choice(fish.swim_keys())
    axs[2].set_title(f'{random_assay} Behaviors (random assay for QC)', fontsize=6)
    plot_behaviors_as_image(
        fish[random_assay].predicted_behaviors,
        random_assay,
        axs[2],
        maxwidth=800)
    if not fpath.parent.exists():
        fpath.parent.mkdir()
    fig.savefig(fpath.as_posix())
    plt.close(fig=fig)

def qc_fish(
        fish_name,
        pca_result: PCAResult.PCAResult,
        output_dir: pathlib.Path,
        overwrite_existing: bool,
        swarm: WorkerSwarm):
    ''' Basic behavior QC for specific fish.
    '''
    fish = Fish(name=fish_name).load()
    print(fish.name)

    print('    pc space')
    fpath = output_dir / 'behavior_pc_space' / f'{fish.name}.png'
    if not fpath.exists() or overwrite_existing:
        plot_behaviors_in_pc_space(fish, pca_result, fpath)

    def swarm_task():
        print(' cruise length, % cruise, % rest, behavior image')
        fpath = output_dir / 'predicted_behaviors' / f'{fish.name}.png'
        if not fpath.exists() or overwrite_existing:
            plot_cruise_rest_behavior_image(fish, fpath)
        fish.delete()
        progress.increment()
    swarm.add_task(swarm_task)

def qc(overwrite_existing: bool=False):
    ''' Basic behavior QC for all fish.
    '''
    logger = loggers.get_qc_logger(__name__)
    print('Loading...')
    output_dir = FileLocations.get_qc_outputs_dir() / 'behavior_qc'
    output_dir.mkdir(exist_ok=True, parents=True)
    pca_result = data_utils.calculate_pca(feature=FEATURE_FOR_PC_QC)
    print('Loaded.')
    with progress.Progress(len(FDM.get_available_fish_names())):
        with WorkerSwarm(logger) as swarm:
            for name in FDM.get_available_fish_names():
                qc_fish(name, pca_result, output_dir, overwrite_existing, swarm)

if __name__ == '__main__':
    plt.switch_backend('agg') # No gui will be created. Safer for distributed processing.
    FileLocations.parse_default_args()
    qc(overwrite_existing=True)
