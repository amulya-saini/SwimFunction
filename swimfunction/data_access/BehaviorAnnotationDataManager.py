
''' Handles all I/O for ground truth behavior dataset.
'''
from collections import defaultdict, namedtuple
from typing import Tuple
import gzip
import numpy
import pandas
import pathlib
import warnings
from tqdm import tqdm

from swimfunction.data_access import PoseAccess, data_utils
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.DataSet import DataSet
from swimfunction.data_models.Fish import Fish
from swimfunction.data_models.SwimAssay import SwimAssay
from swimfunction.data_models.WindowsDataSet import WindowsDataSet
from swimfunction.context_managers import CacheContext
from swimfunction.global_config.config import config
from swimfunction.pose_processing.pose_cleaners.CompleteAnglePositionNormalizer \
    import CompleteAnglePositionNormalizer

from swimfunction import FileLocations

ENCODING = 'utf-8'

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')

META_LABEL_NAMES = ['name', 'group', 'assay', 'appearance', 'flow', 'label', 'annotator', 'frame']

def time_to_flow(t):
    if t < 70*60*5:
        return 'L'
    elif t < 70*60*10:
        return 'M'
    else:
        return 'H'

DEBUG = print

CA_NORMALIZER = CompleteAnglePositionNormalizer()

UMAP_LOC = FileLocations.GlobalFishResourcePaths.behavior_umap_model
CLASSIFIER_LOC = FileLocations.GlobalFishResourcePaths.behavior_classifier_model

BehaviorsCache = namedtuple('BehaviorsCache', ['behaviors', 'annotators'])

#### Helpers
def _concat_arrays(arrs):
    if isinstance(arrs[0], (list, numpy.ndarray)):
        return numpy.concatenate(arrs)
    return pandas.concat(arrs, ignore_index=True)

def get_meta_labels(fish_name, assay_label, labels, annotator_names, frame_nums):
    ''' Creates pandas dataframe with META_LABEL_NAMES columns for all positions.
    '''
    assay_label = int(assay_label)
    group = data_utils.fish_name_to_group(fish_name)
    appearance = FDM.get_quality(fish_name, assay_label)
    return pandas.DataFrame([
        [
            fish_name, group, assay_label, appearance,
            time_to_flow(frame_num), l, annotator, frame_num
        ] for frame_num, l, annotator in zip(frame_nums, labels, annotator_names)
    ], columns=META_LABEL_NAMES)

#### Counting
def meta_labels_to_meta_episodes(df):
    '''
    Returns
    -------
    episodes: list of dict
        each element is a dict with keys [name, assay, label, episode_id, nframes]
    '''
    episodes = pandas.DataFrame(
        columns=pandas.Index(
            ['name', 'assay', 'label', 'episode_id', 'nframes']))
    name, assay, label = None, None, None
    episode_counter = 0
    for name in pandas.unique(df['name']):
        for assay in pandas.unique(df[(df['name'] == name)]['assay']):
            tmp = df.loc[(df['name'] == name) & (df['assay'] == assay)]
            assay = tmp['assay'].values[0]
            label = tmp['label'].values[0]
            e_len = 0
            for _assay, _label in zip(tmp['assay'], tmp['label']):
                if _assay != assay or _label != label:
                    episodes = episodes.append({
                        'name': name,
                        'assay': assay,
                        'label': label,
                        'episode_id': episode_counter,
                        'nframes': e_len}, ignore_index=True)
                    assay = _assay
                    label = _label
                    e_len = 0
                    episode_counter += 1
                e_len += 1
    # Ensure the final episode is added.
    if episodes['episode_id'].values[-1] != episode_counter:
        episodes = episodes.append(
            {'name': name,
             'assay': assay,
             'label': label,
             'episode_id': episode_counter,
             'nframes': e_len}, ignore_index=True)
    return episodes

def count_unannotated(list_of_movements):
    ''' Get number of unannotated frames ("unknown" behaviors).
    '''
    count = 0
    count_dict = dict(zip(*numpy.unique(list_of_movements, return_counts=True)))
    if BEHAVIORS.unknown in count_dict:
        count = count_dict[BEHAVIORS.unknown]
    return count

def count_annotated(list_of_movements):
    ''' Get number of annotated frames (not "unknown" behaviors).
    '''
    return len(list_of_movements) - count_unannotated(list_of_movements)

def report_meta_labels_summary_by_names(meta_labels):
    names = pandas.unique(meta_labels['name'])
    names = sorted(names, key=lambda a: (100 * ord(a[0])) + int(a[1:]))
    lines = [[]]
    entries_per_line = 5
    for fish_name in names:
        if len(lines[-1]) >= entries_per_line:
            lines.append([])
        lines[-1].append(f'{fish_name}:{meta_labels[meta_labels["name"] == fish_name].shape[0]}')
    print('\n'.join(['\t'.join(line) for line in lines]))

def report_meta_labels_summary(meta_labels, name='Annotated'):
    '''
    Parameters
    ----------
    meta_labels : pandas.DataFrame
    name : str, default='Annotated'
        name of the dataset to be printed (just for appearance)
    '''
    if meta_labels is None or not meta_labels.size:
        print('Nothing has been annotated yet.')
        return
    useful_columns = ['group', 'assay', 'appearance', 'flow', 'annotator']
    # Print dataset name
    print(name, f'({meta_labels.shape[0]} total annotations) :')
    report_meta_labels_summary_by_names(meta_labels)
    labels = pandas.unique(meta_labels['label'])
    for feature in useful_columns:
        print(feature)
        print('\t\t\t%s' %'\t\t'.join(labels))
        for val in sorted(list(pandas.unique(meta_labels[feature])), key=lambda e: (e is None, e)):
            count_of_val = meta_labels[meta_labels[feature] == val].shape[0]
            portion_of_val = int(100 * count_of_val / meta_labels.shape[0]) \
                if meta_labels.shape[0] != 0 else 0
            label_str = '\t'.join([
                f'''{
                    meta_labels[(meta_labels[feature] == val) &
                        (meta_labels["label"] == l)].shape[0]
                }({
                    int(
                        100 * meta_labels[(meta_labels[feature] == val) &
                            (meta_labels["label"] == l)].shape[0] / count_of_val
                    ) if count_of_val != 0 else 'NaN'
                }%)''' for l in labels
            ])
            print(f'\t{val}({count_of_val},{portion_of_val}%):\t{label_str}')

def report_train_test_summary(train: DataSet, test: DataSet=None):
    ''' Reports counts in one or each dataset stratified by label, assay_label, group, flow speed, and qualitative score

    Parameters
    ----------
    train : DataSet
        assumed to be training set.
    test : DataSet
        assumed to be test set.
    '''
    for name, DS in (('Train', train), ('Test', test)):
        if DS is None:
            continue
        report_meta_labels_summary(DS.meta_labels, name)

def count_labels(fish_name, assay_label, label):
    return numpy.where(load_behaviors(fish_name, assay_label)[0] == label)[0].shape[0]

#### Loading and Saving
def load_behavior_cache(annotations_path) -> tuple:
    '''
    NOTE: I tested speed and found that
    pandas.read_pickle is faster than pandas.read_hdf or joblib.load
    But in this case, a sparse array in gzip format is faster and smaller than either alternative.
    '''
    annotations_path = pathlib.Path(annotations_path)
    behaviors, annotators = None, None
    if not annotations_path.exists():
        return behaviors, annotators
    if annotations_path.suffix == '.gz':
        with gzip.open(annotations_path, 'rb') as fh:
            nframes = int(fh.readline().decode(ENCODING).strip())
            behaviors = numpy.full(nframes, BEHAVIORS.unknown)
            annotators = [None for _ in range(nframes)]
            for line in fh:
                i, b, a = line.decode(ENCODING).strip().split(',')
                behaviors[int(i)] = b
                annotators[int(i)] = a
    return behaviors, annotators

def save_behavior_cache(current_annotations, annotator_array, annotations_path):
    if current_annotations is None:
        print('Cannot save. Behaviors array is None.')
        return
    if annotator_array is None or len(annotator_array) == 0:
        annotator_array = [None for _ in range(current_annotations.shape[0])]
    annotations_path = pathlib.Path(annotations_path).with_suffix('.gz')
    DEBUG(f'Saving to {annotations_path}')
    counter = 0
    assert len(current_annotations) == len(annotator_array), 'Behaviors and annotator arrays are parallel arrays, must be the same length.'
    with gzip.open(annotations_path, 'wb') as fh:
        fh.write(f'{len(current_annotations)}\n'.encode(ENCODING))
        for i, (b, a) in enumerate(zip(current_annotations, annotator_array)):
            if b == BEHAVIORS.unknown:
                continue
            fh.write(f'{i},{b},{a}\n'.encode(ENCODING))
            counter += 1
    DEBUG(f'Saved {counter} annotations!')

def load_behaviors(fish_name, assay_label, nframes=None, annotations_root=None) -> BehaviorsCache:
    ''' Loads human-annotated list of BEHAVIORS from annotations_path.

    Parameters
    ----------
    fish_name : str
    assay_label : int
    nframes : int, optional
        if provided and there is no annotations_path file,
        a default list of BEHAVIORS.unknown will be returned
    annotations_root : str or pathlib.Path, optional
        path to annotations cache file

    Returns
    -------
    movements_and_annotators : BehaviorsCache
    '''
    annotations_path = FileLocations.get_behavior_annotations_path(fish_name, assay_label, annotations_root)
    movements = None
    movements, annotators = load_behavior_cache(annotations_path)
    if movements is None and nframes is not None:
        movements = numpy.full(nframes, BEHAVIORS.unknown, dtype='<U8')
    if annotators is None and movements is not None:
        annotators = [None for _ in movements]
    if movements is not None and nframes is not None and len(movements) < nframes:
        tmp = movements.copy()
        movements = numpy.full(nframes, BEHAVIORS.unknown, dtype='<U8')
        movements[:len(tmp)] = tmp
    return BehaviorsCache(movements, annotators)

def save_behaviors(
        fish_name: str,
        assay_label,
        current_annotations,
        annotator_array,
        annotations_root=None):
    ''' Save a list of BEHAVIORS

    Parameters
    ----------
    fish_name : str
    assay_label : int or str
        probably week post injury
    current_annotations : list or numpy.ndarray
        values in BEHAVIORS
    annotator_array: list
    annotations_root: str or pathlib.Path, optional
    '''
    # fish_name must not be None or empty
    # assay_label can be 0, cannot be None
    # current_annotations must not be None or empty
    if fish_name is None \
        or not fish_name \
        or assay_label is None \
        or current_annotations is None\
        or len(current_annotations) == 0:
        return
    annotations_path = FileLocations.get_behavior_annotations_path(fish_name, assay_label, annotations_root)
    save_behavior_cache(current_annotations, annotator_array, annotations_path)

def load_umap():
    ''' Load umap model
    '''
    mapper = None
    try:
        with CacheContext.CacheContext(UMAP_LOC) as cache:
            mapper = cache.getContents()
    except ModuleNotFoundError as e:
        print('UMAP model could not be loaded:', e)
    return mapper

def save_umap(umap_mapper):
    ''' Save umap model in UMAP_LOC and in behaviors model dir, just in case.
    '''
    if umap_mapper is not None:
        with CacheContext.CacheContext(UMAP_LOC) as cache:
            cache.saveContents(umap_mapper)
        with CacheContext.CacheContext(FileLocations.get_behaviors_model_dir() / f'backup_{UMAP_LOC.name}') as cache:
            cache.saveContents(umap_mapper)

def load_classifier():
    ''' Load classifying model
    '''
    model = None
    try:
        with CacheContext.CacheContext(CLASSIFIER_LOC) as cache:
            model = cache.getContents()
    except Exception as e:
        print('Classifying model could not be loaded:', e)
    return model

def save_classifier(classifying_model):
    ''' Save classifying model in CLASSIFIER_LOC and in behaviors model dir, just in case.
    '''
    if classifying_model is not None:
        with CacheContext.CacheContext(CLASSIFIER_LOC) as cache:
            cache.saveContents(classifying_model)
        with CacheContext.CacheContext(FileLocations.get_behaviors_model_dir() / f'backup_{CLASSIFIER_LOC.name}') as cache:
            cache.saveContents(classifying_model)

def save_training_data(train: DataSet, test: DataSet, feature_tag: str):
    ''' Save train and test DataSet objects

    Parameters
    ----------
    train: DataSet
        Must not be None
    test: DataSet
        Must not be None
    feature_tag: str
        For file naming. Saves as train_test_datasets_{feature}.pickle
    '''
    if train is None or test is None:
        return
    warnings.warn('WARNING! This will overwrite all global training data for the behavior classifier!')
    data_path = FileLocations.GlobalFishResourcePaths.train_test_filepath(feature_tag)
    with CacheContext.CacheContext(data_path) as cache:
        cache.saveContents({
            'train': train.as_dict(),
            'test': test.as_dict(),
        })

def load_training_data(feature_tag: str) -> Tuple[WindowsDataSet, WindowsDataSet]:
    ''' Load train and test DataSet objects

    Parameters
    ----------
    feature_tag: str
        The naming convention is train_test_datasets_{feature_tag}.pickle
    '''
    train = None
    test = None
    data_path = FileLocations.GlobalFishResourcePaths.train_test_filepath(feature_tag)
    with CacheContext.CacheContext(data_path) as cache:
        contents = cache.getContents()
        if contents is not None:
            train = DataSet.from_dict(contents['train'])
            test = DataSet.from_dict(contents['test'])
    if train is not None and train.data.shape[1] == WindowsDataSet.WINDOW_SIZE:
        train = WindowsDataSet.cast(train)
    if test is not None and test.data.shape[1] == WindowsDataSet.WINDOW_SIZE:
        test = WindowsDataSet.cast(test)
    return train, test

def get_all_training_meta_labels(annotations_root=None):
    meta_label_arrs = []
    annotation_files = FileLocations.get_all_behavior_annotations_files(annotations_root)
    for fname in annotation_files:
        fd = data_utils.parse_details_from_filename(fname)[0]
        name, assay_label = fd.name, fd.assay_label
        behaviors, annotator_arr = load_behavior_cache(fname)
        behaviors = numpy.asarray(behaviors)
        valid_loc = numpy.where(behaviors != BEHAVIORS.unknown)
        frame_nums = valid_loc[0]
        annotated_labels = behaviors[valid_loc]
        annotator_names = [annotator_arr[i] for i in valid_loc[0]]
        meta_label_arrs.append(get_meta_labels(name, assay_label, annotated_labels, annotator_names, frame_nums))
    meta_labels = _concat_arrays(meta_label_arrs)
    return meta_labels

def get_data_from_assay(
        fname: str,
        swim: SwimAssay,
        feature: str,
        make_windows_data_set: bool,
        normalize_head_coordinates: bool) -> tuple:
    behaviors, annotator_arr = load_behavior_cache(fname)
    if behaviors is None \
            or numpy.all(behaviors == BEHAVIORS.unknown) \
            or swim.fish_name not in FDM.get_available_fish_names():
        return (None, None, None)
    behaviors = numpy.asarray(behaviors)
    annotator_arr = list(annotator_arr)
    labels_loc = numpy.where(behaviors != BEHAVIORS.unknown)[0]
    poses_loc = labels_loc
    if labels_loc.size == 0:
        return (None, None, None)
    pose_data = PoseAccess.get_feature_from_assay(
        swim,
        feature=feature,
        filters=[],
        keep_shape=True)
    if normalize_head_coordinates:
        pose_data = CA_NORMALIZER.clean(pose_data)
    # Convert to windows, then to dataset. Concatenate the result with others.
    if make_windows_data_set:
        windows = WindowsDataSet().from_poses(pose_data, behaviors)
        pose_data = windows.data
        labels_loc = labels_loc[numpy.logical_and(
            labels_loc > windows.margin,
            labels_loc < pose_data.shape[0] - windows.margin)]
        poses_loc = labels_loc - windows.margin
    annotated_data = pose_data[poses_loc, ...]
    annotated_labels = behaviors[labels_loc]
    meta_labels = get_meta_labels(
        swim.fish_name,
        swim.assay_label,
        annotated_labels,
        annotator_names=[annotator_arr[i] for i in labels_loc],
        frame_nums=labels_loc)
    return annotated_data, annotated_labels, meta_labels

def organize_files_by_name(behavior_annotation_files: list) -> dict:
    files = defaultdict(list)
    for fname in behavior_annotation_files:
        fd = data_utils.parse_details_from_filename(fname)[0]
        files[fd.name].append((fname, fd.assay_label))
    return files

def get_all_annotated_data(
        feature: str,
        make_windows_data_set: bool = True,
        normalize_head_coordinates: bool = False) -> DataSet:
    training_data_arrs = []
    training_labels_arrs = []
    training_meta_labels_arrs = []
    annotation_files = sorted(FileLocations.get_all_behavior_annotations_files())
    files_dict = organize_files_by_name(annotation_files)
    print('Getting annotated data')
    for fish_name, fname_assay_label in tqdm(files_dict.items(), total=len(files_dict), colour='#9999ff', smoothing=0):
        fish = Fish(fish_name).load()
        for (fname, assay_label) in fname_assay_label:
            annotated_data, annotated_labels, meta_labels = get_data_from_assay(
                fname, fish[assay_label], feature,
                make_windows_data_set, normalize_head_coordinates)
            if annotated_data is not None:
                training_data_arrs.append(annotated_data)
                training_labels_arrs.append(annotated_labels)
                training_meta_labels_arrs.append(meta_labels)
    training_data = _concat_arrays(training_data_arrs)
    training_labels = _concat_arrays(training_labels_arrs)
    training_meta_labels = _concat_arrays(training_meta_labels_arrs)
    if len(training_data) == 0:
        raise RuntimeError('No training data could be found.')
    result = DataSet(
        data=training_data,
        labels=training_labels,
        meta_labels=training_meta_labels)
    if make_windows_data_set:
        result = WindowsDataSet().from_concatenated_poses(
            training_data,
            training_labels,
            training_meta_labels)
    result.drop_nan()
    return result

def get_data_for_training(
    percent_test: int,
    use_cached_training_data: bool,
    feature: str='smoothed_angles') -> Tuple[WindowsDataSet, WindowsDataSet]:
    '''
    Parameters
    ----------
    percent_test : int
        (0, 100), percent of data to reserve for test set
    enforce_equal_representations : bool
        whether to ensure equal representation for each label (could reduce total amount of training data)
    use_cached_training_data : bool
        whether to use cached train/test datasets instead of creating new train/test datasets
    feature : str, default='smoothed_angles'
        feature to use as data.
    '''
    train, test = load_training_data(feature) if use_cached_training_data else (None, None)
    if not use_cached_training_data or train is None or test is None:
        data_set = get_all_annotated_data(feature, make_windows_data_set=True)
        train, test = data_set.split_for_cross_validation(percent_test)
    return train, test

def get_percent_annotated(fish_name, assay_label):
    A = load_behaviors(fish_name, assay_label)[0]
    return count_annotated(A) / len(A) if A is not None else 0

if __name__ == '__main__':
    get_all_annotated_data('smoothed_angles')
    # report_meta_labels_summary(get_all_training_meta_labels())
