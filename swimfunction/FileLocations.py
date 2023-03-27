''' Contains locations of important files.
    Please do not use white space (" ") in file names.
    If you use spaces, then you are responsible for escaping the characters appropriately.
    You are fairly warned.
'''
import threading
import traceback
import argparse
from typing import List
from collections import defaultdict
import pathlib

from swimfunction.data_access.data_utils import AnnotationTypes, parse_details_from_filename
from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
from swimfunction.context_managers.CacheContext import CacheContext
from swimfunction.global_config.config import config, CacheAccessParams
from swimfunction.video_processing import fp_ffmpeg
from swimfunction import progress

TESTING_MODE = config.getboolean('TEST', 'test')

def as_absolute_path(location) -> pathlib.Path:
    ''' Get the location as a Path expanded and resolved.
    '''
    return pathlib.Path(location).expanduser().resolve()

SWIMFUNCTION_ROOT = as_absolute_path(__file__).parent.parent

##################  IMPORTANT! UPDATE THESE PATHS FOR YOUR ENVIRONMENT!  ##################
#*******  You choose these locations. Make them whatever is most convenient!  *******#

##################  DeepLabCut(DLC) and Pre-DLC Analysis  ##################

HOME = as_absolute_path('~')

##################  Post-DLC Analysis  ##################

def mkdir_and_return(path) -> pathlib.Path:
    ''' Ensure the path is absolute,
    then create the directory if necessary and return the result.
    '''
    path = as_absolute_path(path)
    if not TESTING_MODE:
        path.mkdir(parents=True, exist_ok=True)
    return path

def get_cache_root():
    ''' Access cache root directory
    '''
    return mkdir_and_return(config.access_params.cache_root)

def get_test_root():
    '''
    Returns
    -------
        SWIMFUNCTION_ROOT / 'tests'
    '''
    return mkdir_and_return(SWIMFUNCTION_ROOT / 'tests')

def get_experiment_name() -> str:
    ''' Access experiment name
    '''
    return config.experiment_name

def get_experiment_cache_root() -> pathlib.Path:
    ''' Access experiment root directory
    '''
    return mkdir_and_return(get_cache_root() / config.experiment_name)

def get_videos_cache_root() -> pathlib.Path:
    ''' Location to save intermediate and final video files.
    Returns
    -------
        get_experiment_cache_root() / 'videos'
    '''
    return get_experiment_cache_root() / 'videos'

#### NOTE 1: THESE FILES MUST EXIST IF RELEVANT TO YOUR WORK.
def get_qualitative_score_file() -> pathlib.Path:
    ''' Comma separated, includes header, human-assigned swim quality scores.
    Format: fish_name,assay1_score,assay2_score,...
    Example:
        fish,-1,1,2,3,4,5,6,7,8
        M1,5,3,1,2,1,1,1,0,0
        M2,5,3,2,2,1,1,1,1,1
    '''
    return get_videos_cache_root() / 'qualitative_recovery.csv'

def get_endurance_times_file() -> pathlib.Path:
    ''' Comma separated, includes header, endurance times.
    Format: fish_name,assay1_score,assay2_score,...
    Example:
        fish,-1,1,2,3,4,5,6,7,8
        M1,5,3,1,2,1,1,1,0,0
        M2,5,3,2,2,1,1,1,1,1
    '''
    return get_experiment_cache_root() / 'endurance_times.csv'

def get_fish_to_assays_csv() -> pathlib.Path:
    ''' Comma separated, no header, which assays are available.
    Format: fish_name,assay1,assay2,assay3,...
    Example:
        F6,-1,1,2,3,4,5,6,7,8
        F7,-1
        F8,-1,1,2,3,4,5,6,7,8
    '''
    return get_videos_cache_root() / 'available_assays.csv'

def get_precalculated_metrics_dir() -> pathlib.Path:
    ''' Location of csv files containing precalculated metrics
    (e.g., quality, glial bridging, etc.)
    See ScoreFileManager for details.
    '''
    return mkdir_and_return(get_experiment_cache_root() / 'precalculated')

#### END NOTE 1

### Video directories

def get_original_videos_dir() -> pathlib.Path:
    ''' Note: see find_video_files() to search for videos within this directory.
    '''
    return mkdir_and_return(get_videos_cache_root() / 'originals')

def get_median_frames_dir(videos_dir) -> pathlib.Path:
    '''
    Directory which will contain video median files
    called "median_{video_fname}.png"
    (these are created by get_median_frames.py)
    '''
    return mkdir_and_return(as_absolute_path(videos_dir) / 'median_frames')

def get_normalized_videos_dir() -> pathlib.Path:
    ''' Directory to save final videos (likely cropped, resized, normalized)
    '''
    return mkdir_and_return(get_videos_cache_root() / 'normalized')

### DeepLabCut directories
def get_training_root_dir() -> pathlib.Path:
    ''' Get path to pose annotation training images cache
    Returns
    -------
        cache_root / 'train_images'
    '''
    return mkdir_and_return(get_cache_root() / 'train_images')

def get_dlc_outputs_dir() -> pathlib.Path:
    ''' Get path to dlc h5/csv cache
    '''
    return mkdir_and_return(get_experiment_cache_root() / 'raw_pose_annotations')

def get_dlc_caudal_fin_outputs_dir() -> pathlib.Path:
    ''' Get path to dlc h5/csv cache FOR CAUDAL FINS SPECIFICALLY
    Note: Does not create the directory. It may return a non-existing path.
    '''
    return get_dlc_outputs_dir() / 'caudal_fin_annotations'

def get_dlc_body_outputs_dir() -> pathlib.Path:
    ''' Get path to dlc h5/csv cache FOR BODY CENTERLINE KEYPOINTS SPECIFICALLY
    Note: Does not create the directory. It may return a non-existing path.
    '''
    return get_dlc_outputs_dir() / 'body_annotations'

### Data and QC cache directories
def get_qc_outputs_dir() -> pathlib.Path:
    '''
    Returns
    -------
        cache_root / experiment_name / 'qc'
    '''
    return mkdir_and_return(get_experiment_cache_root() / 'qc')

def get_processed_pose_cache_dir() -> pathlib.Path:
    ''' Path to pose-related cache files (poses in numpy format, pca, etc)
    Returns
    -------
        cache_root / experiment_name / 'processed_pose_data'
    '''
    return mkdir_and_return(get_experiment_cache_root() / 'processed_pose_data')

def get_local_pca_cache() -> pathlib.Path:
    ''' Location to save PCA models (pickled PCAResult objects)
    '''
    return mkdir_and_return(get_processed_pose_cache_dir() / 'pca_models')

def get_cruise_embedding_dir(include_scoliotic_fish: bool) -> pathlib.Path:
    ''' Location of cruise embedding plots and models
    Returns
    -------
    pathlib.Path
    '''
    subdir = 'full_embedding' if include_scoliotic_fish else 'nonscoliotic_embedding'
    return mkdir_and_return(get_plots_dir() / 'cruise_embedding' / subdir)

def get_fish_cache() -> pathlib.Path:
    '''
    Returns
    -------
        cache_root / experiment_name / 'processed_pose_data' / 'fish'
    '''
    return mkdir_and_return(get_processed_pose_cache_dir() / 'fish')

def get_behaviors_model_dir() -> pathlib.Path:
    '''
    Returns
    -------
        cache_root / experiment_name / 'behaviors_model'
    '''
    return mkdir_and_return(get_experiment_cache_root() / 'behaviors_model')

def get_csv_output_dir() -> pathlib.Path:
    '''
    Returns
    -------
        cache_root / experiment_name / 'csv_results'
    '''
    return mkdir_and_return(get_experiment_cache_root() / 'csv_results')

def get_outcome_prediction_csv():
    ''' Location to store outcome predictions.
    '''
    return get_csv_output_dir() / 'outcome_predictions.csv'

def get_outcome_prediction_final_name_key_csv():
    ''' Location of csv that has the outcome name key.
    '''
    return get_experiment_cache_root() / 'outcome_prediction_final_name_key.csv'

def get_posture_novelty_models_dir() -> pathlib.Path:
    '''
    Returns
    -------
        cache_root / experiment_name / 'lof_models'
    '''
    return mkdir_and_return(get_experiment_cache_root() / 'lof_models')

def get_plots_dir() -> pathlib.Path:
    '''
    Returns
    -------
        cache_root / experiment_name / 'plots'
    '''
    return mkdir_and_return(get_experiment_cache_root() / 'plots')

### Files that will be created
def get_waveform_dir() -> pathlib.Path:
    ''' Path to save cruise waveform outputs (plots and )
    '''
    return mkdir_and_return(get_experiment_cache_root() / 'cruise_waveform')

def get_capacity_metrics_plot_dir() -> pathlib.Path:
    ''' Where to save the swim capacity output plots
    '''
    return mkdir_and_return(get_plots_dir() / 'swim_capacity')

def get_waveform_file(assay, annotation_type) -> pathlib.Path:
    ''' Path to specific cruise waveform statistics file
    (cruise annotation type can be specified)
    '''
    cruises = 'predicted_cruise' if annotation_type == AnnotationTypes.predicted else \
         'manual_cruises' if annotation_type == AnnotationTypes.human else 'any_cruises'
    return get_waveform_dir() / f'{cruises}_waveform_stats_assay{assay}.h5'

### Behavior Annotation File Location Helpers
def get_behavior_annotations_path(fish_name, assay_label, annotations_root=None) -> pathlib.Path:
    ''' Path to specific behavior annotation file (human annotations)
    '''
    # Note: this file name must not change, otherwise two other places
    # must be updated in this file.
    if annotations_root is None:
        annotations_root = get_behaviors_model_dir()
    return as_absolute_path(annotations_root) \
        / f'behaviors_{assay_label}wpi_{fish_name}.gz'

def get_all_behavior_annotations_files(annotations_root=None) -> List[str]:
    ''' Paths to all behavior annotation files (human annotations)
    '''
    searchpath = get_behavior_annotations_path('*', '*', annotations_root)
    return [p.as_posix() for p in searchpath.parent.glob(searchpath.name)]

##################  Helpers for production (these you can ignore)  ##################

def _get_universal_resources_dir() -> pathlib.Path:
    '''
    Returns
    -------
    pathlib.Path
        SWIMFUNCTION_ROOT / 'universal_fish_video_resources'
    '''
    return mkdir_and_return(SWIMFUNCTION_ROOT / 'universal_fish_video_resources')

class _GlobalFishResourcePaths:

    target_median_frame_with_divider = _get_universal_resources_dir()\
        / 'TARGET_MEDIAN_wide_divider.png'
    target_median_frame_without_divider = _get_universal_resources_dir()\
        / 'TARGET_MEDIAN_wide.png'
    behavior_umap_model = _get_universal_resources_dir()\
        / 'behaviors_umap.pickle'
    behavior_classifier_model = _get_universal_resources_dir()\
        / 'behaviors_classifying_model.pickle'
    novelty_detection_lof_model = _get_universal_resources_dir()\
        / 'novelty_detection_lof_model.pickle'
    novelty_null_counts_location = _get_universal_resources_dir()\
        / 'novelty_detection_null_counts.pickle'

    @property
    def global_pca(self):
        ''' Location of the global angle pose PCA model
        It was trained on all observed poses observed in the tracking experiment.
        (30 male, 30 female; assays including preinjury, 1 wpi, 2 wpi, ..., 8 wpi)
        '''
        return _get_universal_resources_dir() / 'tracking_experiment_pca.pickle'

    @staticmethod
    def train_test_filepath(feature_tag: str):
        ''' Where to save train and test DataSet objects

        Parameters
        ----------
        feature_tag: str
            For file naming. Saves as train_test_datasets_{feature_tag}.pickle
        '''
        return _get_universal_resources_dir() / f'train_test_datasets_{feature_tag}.pickle'

GlobalFishResourcePaths = _GlobalFishResourcePaths()

# Ignore SwimBehavior directory and
# a video that is not readable which will cause code to crash.
VIDEO_FNAME_IGNORES = ['SwimBehavior', '1-28-2020_17491_EKAB_4wpi_F16L_M16R.avi']

def video_is_readable(filepath) -> bool:
    ''' Check whether videos can be read.
    '''
    filepath = as_absolute_path(filepath)
    can_read = filepath.exists() and filepath not in VIDEO_FNAME_IGNORES
    if can_read:
        try: # Skip those we cannot read.
            fp_ffmpeg.VideoData(filepath.as_posix())
        except (KeyError, RuntimeError) as err:
            print(f'Skipping {filepath.name} due to {err}')
            can_read = False
    return can_read

def find_files_by_suffix(root_path, suffix: str, ignores=None):
    ''' Recursively searches through root_path and all subdirectories for all *.avi video files.
    Ignores directories called 'SwimBehavior'

    Parameters
    ----------
    root_path : pathlib.Path or str
        root directory of files.
    suffix : str
        File extension, e.g., '.avi'
    ignores : list
        list of file or directory names to ignore
    '''
    if ignores is None:
        ignores = []
    root_path = as_absolute_path(root_path)
    for item in root_path.glob('*'):
        item_path = as_absolute_path(item)
        if item_path.name in ignores:
            continue
        if item_path.is_dir():
            yield from find_files_by_suffix(item_path, suffix, ignores)
        elif as_absolute_path(item).suffix == suffix:
            yield item

def find_video_files(root_path=None, ignores: list=None):
    ''' Recursively searches through root_path and all subdirectories for all *.avi video files.
    Ignores directories called 'SwimBehavior'

    Parameters
    ----------
    root_path : pathlib.Path or str
        root directory of video files. Default: get_normalized_videos_dir()
    ignores : list
        list of file or directory names to ignore
    '''
    if ignores is None:
        ignores = []
    if isinstance(ignores, str):
        ignores = [ignores]
    if root_path is None:
        root_path = get_normalized_videos_dir()
    return find_files_by_suffix(root_path, '.avi', ignores=ignores + VIDEO_FNAME_IGNORES)

class _Video_Map_Singleton:
    ''' Stores the location of videos in a handy dict.
    '''
    map_lock = threading.Lock()
    __slots__ = ['video_maps']
    def __init__(self):
        self.video_maps = {}

    def _current_key(self):
        return f'{config.access_params.cache_root}:{config.experiment_name}'

    def add_video_to_map(self, vmap, vfile, check_readability):
        ''' Stores a filename and VideoData under
        the keys fish name and assay label.
        '''
        if not check_readability or video_is_readable(vfile):
            for details in parse_details_from_filename(vfile):
                vmap[details.name][details.assay_label] = (
                    vfile,
                    fp_ffmpeg.VideoData(vfile, force_grayscale=True))
        return vmap

    def load_video_map(self, videos_root, force_recalculate=False, check_readability=True):
        ''' Gets video map for videos, forces grayscale
        (assumes that the videos have been normalized and grayscaled)
        '''
        with _Video_Map_Singleton.map_lock:
            try:
                key = self._current_key()
                if key not in self.video_maps or force_recalculate:
                    self.video_maps[key] = self._assemble_video_map(
                        videos_root, force_recalculate, check_readability)
            except Exception as e:
                print('Could not get video map due to', e)
                print(traceback.format_exc())
        return self.video_maps[key]

    def _assemble_video_map(self, videos_root, force_recalculate=False, check_readability=True):
        video_map_cache = as_absolute_path(videos_root) / 'video_map.pickle'
        with CacheContext(video_map_cache) as vm_cache:
            vmap = vm_cache.getContents()
            if vmap is None or force_recalculate:
                print('Assembling video map for the first time...')
                vmap = defaultdict(dict)
                video_files = list(find_video_files(root_path=videos_root))
                with progress.Progress(len(video_files)) as p:
                    for i, vfile in enumerate(video_files):
                        p.progress(i)
                        vmap = self.add_video_to_map(vmap, vfile, check_readability)
                    vm_cache.saveContents(vmap)
                print('Video map assembled!')
        return vmap

video_mapper = _Video_Map_Singleton()
load_video_map = video_mapper.load_video_map

def find_video(fish_name, assay_label, videos_root=None, _first_try=True) -> pathlib.Path:
    ''' Given a fish name and assay_label,
    get the path to the associated (normalized) video file.
    Recursive. Do not use _first_try parameter, that is handled by default.
    '''
    if videos_root is None:
        videos_root = get_normalized_videos_dir()
    videos_root = pathlib.Path(videos_root)
    fname = None
    video_map = load_video_map(videos_root, force_recalculate=False)
    if fish_name in video_map and assay_label in video_map[fish_name]:
        fname = video_map[fish_name][assay_label][0]
        if not pathlib.Path(fname).exists():
            print('The file we found does not exist on your computer. \
                The video map is being reloaded. This will take a minute.')
            video_map = load_video_map(videos_root, force_recalculate=True)
            if fish_name in video_map and assay_label in video_map[fish_name]:
                fname = video_map[fish_name][assay_label][0]
    elif _first_try:
        load_video_map(videos_root, force_recalculate=True)
        fname = find_video(fish_name, assay_label, videos_root, _first_try=False)
    if fname is not None:
        fname = as_absolute_path(fname)
    if fname is None or not fname.exists():
        fname = None
        print(f'Could not find a video for {fish_name} {assay_label} located at {videos_root}')
    return fname

##################  Helpers for testing (these you can ignore)  ##################

def use_default_args(
        cache_root=None,
        experiment_name=None,
        config_path=None):
    ''' Unless the user provides commandline arguments that override these,
    this function sets the defaults.
    '''
    class ARGS:
        ''' Placeholder default arguments
        '''
        __slots__ = ['cache_root', 'experiment_name', 'config_path']
        def __init__(self, cache_root, experiment_name, config_path):
            self.cache_root = cache_root
            self.experiment_name = experiment_name
            self.config_path = config_path
    args = ARGS(
        config.access_params.cache_root,
        config.experiment_name,
        config.config_path.as_posix()
    )
    if cache_root is not None:
        args.cache_root = as_absolute_path(cache_root)
    if experiment_name is not None:
        args.experiment_name = experiment_name
    if config_path is not None:
        args.config_path = as_absolute_path(config_path).as_posix()
    config.clear()
    config.read(args.config_path)
    access_params = CacheAccessParams(
        cache_root=args.cache_root,
        experiment_name=args.experiment_name)
    config.set_access_params(access_params)
    return args

def parse_default_args(*add_arg_fns, unknown_args_throw_error=True):
    ''' Creates argument parser, parses for
        cache access parameters: cache_root, experiment_name by default.
    Saves these access parameters in global config singleton for use by other scripts.

    Note: additional arguments are ok, but '-r', '-e', '-c', and '-t' are reserved.

    Also takes any additional add_argument functions
        Example: parse_default_args(
            lambda parser: parser.add_argument('positional_arg'),
            lambda parser: parser.add_argument('-o', '--optional_arg'))

    Returns
    -------
        args
    '''
    parser = argparse.ArgumentParser()
    # First, add any possible positional arguments
    for arg_fn in add_arg_fns:
        arg_fn(parser)
    # Now, add optional cache access arguments
    parser.add_argument('-r', '--cache_root',
                        help='Location to save cache files, \
                            will create files that do not yet exist',
                        default=config.access_params.cache_root)
    parser.add_argument('-e', '--experiment_name',
                        help='Identifies internal cache directories.\
                            Experiment name, could be a date tag. ',
                        default=config.experiment_name)
    parser.add_argument('-c', '--config_path', default=config.config_path.as_posix())
    parser.add_argument(
        '-t', '--worker_threads',
        help='Number of worker threads to use.', default=1, type=int)

    args = parser.parse_args() if unknown_args_throw_error else parser.parse_known_args()[0]
    config.clear()
    config.read(args.config_path)
    access_params = CacheAccessParams(
        cache_root=args.cache_root,
        experiment_name=args.experiment_name
    )
    config.set_access_params(access_params)

    WorkerSwarm.num_allowed_workers = args.worker_threads
    return args
