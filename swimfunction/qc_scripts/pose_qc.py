''' Reports quality of DeepLabCut predicted poses.
Plots
    pose quality,
    binary pose quality,
    high-quality-box-lengths (histogram), and
    weekly high quality pose counts.
'''
from matplotlib import pyplot as plt
import numpy
import threading
import traceback

from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models import Fish
from swimfunction.global_config.config import config

from swimfunction import FileLocations
from swimfunction import loggers
from swimfunction import progress

WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')

QUALITY_CUTOFF = 0.5

def _get_binary_likelihoods(swim):
    ''' Returns numpy array of 0 or 1
    Applies threshold to binarize likelihoods and summarizes \
        such that if one keypoint is below the threshold, \
        the binary score of the pose is 0.
    '''
    mins = swim.likelihoods.min(axis=1)
    binary_likelihoods = numpy.zeros_like(mins, dtype=numpy.int8)
    binary_likelihoods[mins < QUALITY_CUTOFF] = 0
    binary_likelihoods[mins >= QUALITY_CUTOFF] = 1
    return binary_likelihoods

def _count_quality_spans(swim):
    ''' A quality span is a group of adjacent high-quality poses.
    We want to know the length distribution of quality spans.
    There may be one quality span which covers the entire video,
    or there may be tons of short quality spans interrupted by
    frequent low-quality frames.
    We want to know the distribution of high-quality frame blocks.
    '''
    binary_likelihoods = _get_binary_likelihoods(swim)
    spans = []
    current_span = 0
    for x in binary_likelihoods:
        if x == 0:
            if current_span > 0:
                spans.append(current_span)
            current_span = 0
        if x == 1:
            current_span += 1
    if current_span > 0:
        spans.append(current_span)
    return numpy.asarray(spans)

def plot_keypoint_quality(fish, output_dir):
    ''' Plots each keypoint's quality scores over time
    '''
    fig, axs = plt.subplots(3, 3)
    fig.tight_layout()
    for i, week in enumerate(WPI_OPTIONS):
        if week not in fish:
            continue
        swim = fish[week]
        row = i // 3
        col = i % 3
        for keypoint in range(swim.likelihoods.shape[1]):
            axs[row, col].plot(numpy.arange(swim.likelihoods.shape[0]), swim.likelihoods[:, keypoint], label=keypoint, linewidth=1)
        axs[row, col].set_title(f'{week}wpi')
    fig.savefig((output_dir / f'{fish.name}_quality.png').as_posix())
    plt.close(fig)

def plot_binary_quality(fish, output_dir):
    ''' Plots result of _get_binary_likelihoods,
        which is whether or not the pose has any low-likelihood keypoints.
    '''
    fig, axs = plt.subplots(3, 3)
    fig.tight_layout()
    for i, week in enumerate(WPI_OPTIONS):
        binary_likelihoods = numpy.array([])
        if week in fish:
            swim = fish[week]
            binary_likelihoods = _get_binary_likelihoods(swim)
        row = i // 3
        col = i % 3
        if len(binary_likelihoods) == 0:
            continue
        axs[row, col].plot(numpy.arange(binary_likelihoods.shape[0]), binary_likelihoods, linewidth=1.5)
        axs[row, col].set_title(f'{week}wpi')
    fig.savefig((output_dir / f'{fish.name}_binary_quality.png').as_posix())
    plt.close(fig)

def plot_high_quality_block_histograms(fish, output_dir):
    ''' Plots histogram of lengths of high-quality runs in the data.
    A high-quality run is a block of video where every frame has a pose with no low-quality keypoints.
    '''
    fig, axs = plt.subplots(3, 3)
    fig.tight_layout()
    for i, week in enumerate(WPI_OPTIONS):
        quality_spans = []
        if week in fish:
            swim = fish[week]
            quality_spans = _count_quality_spans(swim)
        row = i // 3
        col = i % 3
        if len(quality_spans) == 0:
            continue
        axs[row, col].hist(quality_spans, bins=min((4, len(quality_spans))))
        axs[row, col].vlines(x=quality_spans.max(), ymin=0, ymax=axs[row, col].get_ylim()[1], color='red')
        axs[row, col].set_title(f'{week}wpi')
    fig.savefig((output_dir / f'{fish.name}_quality_pose_block_lengths.png').as_posix())
    plt.close(fig)

def _get_quality_counts(fish):
    counts = []
    for swim_i in WPI_OPTIONS:
        if swim_i not in fish:
            counts.append(numpy.nan)
            continue
        swim = fish[swim_i]
        lmins = swim.likelihoods.min(axis=1)
        quality_pose_count = lmins[lmins>=QUALITY_CUTOFF].shape[0]
        # q1 = lmins[lmins>=0.5].shape[0]
        # q2 = lmins[lmins>=0.05].shape[0]
        # quality_counts[fish_name].append(q2 - q1)
        counts.append(quality_pose_count)
    return counts

def plot_num_quality_poses_by_week(quality_counts, output_dir):
    fig, ax = plt.subplots()
    fig.suptitle(f'Number of high quality (>{QUALITY_CUTOFF}) poses from week to week')
    for fish_name in quality_counts:
        ax.set_xlabel('weeks post-injury')
        ax.set_ylabel('Number of frames with high quality')
        ax.plot(WPI_OPTIONS, quality_counts[fish_name], label=fish_name, linewidth=1)
        ax.scatter(WPI_OPTIONS, quality_counts[fish_name], label=fish_name)
    fig.savefig((output_dir / 'weekly_high_quality_pose_counts.png').as_posix())
    plt.close(fig)
    with open(output_dir / 'weekly_high_quality_pose_counts.csv', 'wt') as fh:
        fh.write('Fish,-1,1,2,3,4,5,6,7,8\n')
        for fish_name in quality_counts:
            quality_string = ','.join([str(x) for x in quality_counts[fish_name]])
            fh.write(f'{fish_name},{quality_string}\n')

def qc(overwrite_existing=False):
    logger = loggers.get_qc_logger(__name__)
    print(f'For binary quality plots, using quality cutoff of {QUALITY_CUTOFF}')
    output_dir = FileLocations.get_qc_outputs_dir() / 'pose_quality_by_fish'
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f'Saving pose QC plots to {output_dir.as_posix()}')
    print('...')
    dict_access_lock = threading.Lock()
    quality_counts = {}
    def handle_fish(fish_name):
        progress.increment(fish_name)
        fish = Fish.Fish(name=fish_name).load()
        with dict_access_lock:
            try:
                quality_counts[fish_name] = _get_quality_counts(fish)
            except Exception as e:
                print(e)
                print(traceback.format_exc())
        plot_keypoint_quality(fish, output_dir)
        plot_binary_quality(fish, output_dir)
        plot_high_quality_block_histograms(fish, output_dir)
        fish.delete()

    fish_names = sorted(FDM.get_available_fish_names())
    with progress.Progress(len(fish_names)):
        with WorkerSwarm(logger) as swarm:
            for fish_name in fish_names:
                if not overwrite_existing and list(output_dir.glob(f'{fish_name}*')):
                    swarm.logger.warning(f'{fish_name} already processed, so pose qc will terminate.')
                else:
                    swarm.add_task(lambda name=fish_name: handle_fish(name))
    if len(quality_counts) == len(fish_names):
        plot_num_quality_poses_by_week(quality_counts, output_dir)
    print('Done!')

if __name__ == '__main__':
    plt.switch_backend('agg') # No gui will be created. Safer for distributed processing.
    args = FileLocations.parse_default_args(
        lambda parser: parser.add_argument(
            '-o', '--overwrite_existing',
            help='Whether to overwrite existing output. Default is to stop the script if a file already exists.',
            default=False, action='store_true')
    )
    qc(args.overwrite_existing)
