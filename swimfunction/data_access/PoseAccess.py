''' Helpers to access pose data
'''
from collections import namedtuple
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models import Fish
from swimfunction.data_models.SwimAssay import SwimAssay
from swimfunction.data_access.data_utils import AnnotationTypes
from swimfunction.pose_processing.pose_filters \
    import get_filter_mask, apply_filter_mask, BASIC_FILTERS
from swimfunction.global_config.config import config
from typing import List

from swimfunction import FileLocations
import numpy
from swimfunction import progress

WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')
BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')

def get_feature_from_assay(swim: SwimAssay, feature: str, filters, keep_shape: bool):
    ''' Returns complete angle poses.

    Parameters
    ----------
    swim : SwimAssay
    feature : str
        feature to take, must be an attribute of SwimAssay or smoothed_angles
        (e.g. raw_coordinates, likelihoods, smoothed_coordinates, smoothed_angles, smoothed_complete_angles, behaviors...)
    filters : list
        list of filters, can be empty
    keep_shape : bool
        whether to keep the shape of the original swim assay and fill irrelevant poses with numpy.nan

    Returns
    -------
    data : numpy.ndarray
        array of the requested feature.
    '''
    if not hasattr(swim, feature):
        raise ValueError(f'Feature {feature} not an attribute of {swim.__class__}')
    poses = getattr(swim, feature)
    data = apply_filter_mask(
        get_filter_mask(swim, filters),
        poses,
        keep_shape)
    return data

def get_feature(
        fish_names=None,
        assay_labels=None,
        feature: str='',
        filters: list=None,
        keep_shape: bool=False):
    ''' Returns FeaturesTuple.

    Parameters
    ----------
    fish_names : list, optional
        fish names, the default is all fish.
    assay_labels: list, optional
        weeks post injury, the default is all assays.
    feature : str
        feature to take, must be an attribute of SwimAssay or smoothed_angles
        (e.g. raw_coordinates, likelihoods, smoothed_coordinates,
        smoothed_angles, smoothed_complete_angles, behaviors...)
    filters : list, optional
        filters, default: empty list.
    keep_shape : bool
        whether to keep the shape of the original swim assay
        and fill irrelevant poses with numpy.nan

    Returns
    -------
    feature_data
        list of numpy.ndarray of feature data.
            [
                fish1 assay1 data,
                fish1 assay2 data,
                ...
                fish2 assay1 data,
                fish2 assay2 data,
                ...
                fish_N assay_M data
            ]
    fish_names
        list of fish names (same length as other two lists)
    assay_labels
        list of assay labels (same length as other two lists)

    Raises
    ------
    ValueError
        If feature string is None or empty.
    '''
    if fish_names is not None and isinstance(fish_names, str):
        fish_names = [fish_names]
    if assay_labels is not None and not isinstance(assay_labels, (list, numpy.ndarray)):
        assay_labels = [assay_labels]
    if feature is None or not feature:
        raise ValueError('Feature must be a valid attribute string.')
    fish_names = FDM.get_available_fish_names() if fish_names is None else fish_names
    assay_labels = WPI_OPTIONS if assay_labels is None else assay_labels
    filters = [] if filters is None else filters
    ts_data, names, labels = [], [], []
    total = len(fish_names) * len(assay_labels)
    if total > 10:
        progress.init(len(fish_names) * len(assay_labels))
    counter = 0
    for name in fish_names:
        fish = Fish.Fish(name=name).load()
        for assay_label in assay_labels:
            counter += 1
            assay_label = int(assay_label)
            if assay_label not in fish.swim_keys():
                continue
            swim = fish[assay_label]
            feature_data = get_feature_from_assay(swim, feature, filters, keep_shape)
            if len(feature_data) > 0:
                ts_data.append(feature_data)
                names.append(name)
                labels.append(assay_label)
        if total > 10:
            progress.progress(counter, f'{name}')
        fish.delete() # Explicitly free up the memory since these are large objects
    if total > 10:
        progress.finish()
    return namedtuple('FeaturesTuple', ['features','names','labels'])(ts_data, names, labels)

def split_into_consecutive_indices(list_of_indices):
    ''' Takes a list of indices, splits into blocks of consecutive indices.
    '''
    if (isinstance(list_of_indices, list) and not list_of_indices) \
        or (isinstance(list_of_indices, numpy.ndarray) and not list_of_indices.size):
        return []
    list_of_indices = sorted(list_of_indices)
    blocks = []
    episode = [list_of_indices[0]]
    for idx in list_of_indices[1:]:
        if episode[-1] + 1 != idx:
            # If this index is not adjacent to the previously observed one,
            # save the old episode and start a new one.
            blocks.append(episode)
            episode = []
        episode.append(idx)
    if episode:
        blocks.append(episode)
    return blocks

def get_episodes_of_unmasked_data(swim_assay: SwimAssay, filters: list, min_length=0) -> List[List[int]]:
    final_mask = get_filter_mask(
        swim_assay,
        filters)
    list_of_indices = sorted(numpy.where(final_mask)[0])
    episodes_idx = split_into_consecutive_indices(list_of_indices)
    if min_length:
        episodes_idx = list(filter(lambda ep: len(ep) >= min_length, episodes_idx))
    return episodes_idx

def _filter_behaviors(swim, behaviors, filters):
    filtered = numpy.asarray(behaviors).copy()
    if len(filtered) == 0:
        return filtered
    mask = numpy.ones_like(filtered, dtype=bool)
    if filters is not None:
        mask = mask & get_filter_mask(swim, filters)
    filtered[numpy.where(numpy.logical_not(mask))] = BEHAVIORS.unknown
    return filtered

def get_behaviors(swim_assay: SwimAssay, annotation_type: str):
    ''' Gets manually annotated behaviors or behaviors from SwimAssay.

    Priority:
        1. Behavior annotations file (manual annotations)
        2. swim_assay.predicted_behaviors
    '''
    if annotation_type == AnnotationTypes.predicted:
        if swim_assay.predicted_behaviors is not None:
            return swim_assay.predicted_behaviors
        else:
            return numpy.full(swim_assay.raw_coordinates.shape[0], BEHAVIORS.unknown)
    present_behaviors = []
    annotations_path = FileLocations.get_behavior_annotations_path(swim_assay.fish_name, swim_assay.assay_label, annotations_root=None)
    if annotations_path.exists():
        from swimfunction.data_access.BehaviorAnnotationDataManager import load_behavior_cache
        present_behaviors = load_behavior_cache(annotations_path)[0]
    if annotation_type == AnnotationTypes.human:
        return present_behaviors
    if len(present_behaviors) == 0 and swim_assay.predicted_behaviors is not None:
        present_behaviors = swim_assay.predicted_behaviors
    if len(present_behaviors) == 0:
        present_behaviors = numpy.full(swim_assay.raw_coordinates.shape[0], BEHAVIORS.unknown)
    if len(swim_assay.raw_coordinates) != len(present_behaviors):
        raise RuntimeError('Number of behaviors is not equal to the number of poses!')
    return present_behaviors

def episode_indices_to_feature(
        swim: SwimAssay, episode_indices: list, feature='smoothed_angles') -> list:
    ''' Takes a feature from a SwimAssay given a list of lists of indices.
    '''
    feature_data = get_feature_from_assay(swim, feature, [], True)
    if isinstance(episode_indices[0], (list, numpy.ndarray)):
        return [numpy.take(feature_data, e_idx, axis=0) for e_idx in episode_indices]
    else:
        return numpy.take(feature_data, episode_indices, axis=0)

def get_behavior_episodes_from_assay(swim: SwimAssay, behavior: str, annotation_type: str=AnnotationTypes.all, filters=None):
    ''' Generates lists of consecutive indices that all share the same assigned behavior.

    Parameters
    ----------
        swim: SwimAssay
            where to get the poses
        behavior: string, enum=BEHAVIORS
            target behavior
        filters: list, None
            pose_filters functions, can be empty.

    Returns
    ------
    list
        list of indexes for poses in an episode (example: [[2, 3, 4, 5], [2345, 2346, 2347, 2348]])
    '''
    masked_behaviors = _filter_behaviors(swim, get_behaviors(swim, annotation_type), filters)
    list_of_indices = sorted(numpy.where(masked_behaviors == behavior)[0])
    return split_into_consecutive_indices(list_of_indices)

def get_behavior_episodes(
        behavior: str,
        fish_names: list = None,
        assay_labels: list = None,
        filters: list = None,
        feature: str = 'smoothed_angles',
        annotation_type: str = AnnotationTypes.predicted):
    '''
    Generates lists of poses (behavior episodes) for an assigned behavior.

    Parameters
    ----------
    behavior: str
        target behavior
    fish_names : list, optional
        fish names, default: all fish.
    assay_labels : list, optional
        weeks post injury, default: all assays.
    filters : list, default=[].
    feature : str, default='smoothed_angles'
        feature to return. I don't yield the indices
        because I don't yield the fish name and assay label necessary to use the indices.
    annotation_type: str, default=AnnotationTypes.predicted

    Yields
    ------
        numpy.ndarray of the given feature in episodes.
    '''
    if fish_names is None:
        fish_names = FDM.get_available_fish_names()
    if assay_labels is None:
        assay_labels = WPI_OPTIONS
    if filters is None:
        filters = []
    for name in fish_names:
        fish = Fish.Fish(name=name).load()
        for assay_label in assay_labels:
            if assay_label not in fish.swim_keys():
                continue
            swim = fish[assay_label]
            masked_behaviors = _filter_behaviors(
                swim,
                get_behaviors(swim, annotation_type),
                filters)
            if len(masked_behaviors) == 0:
                continue
            list_of_indices = sorted(numpy.where(masked_behaviors == behavior)[0])
            feature_vals = get_feature_from_assay(swim, feature, filters=None, keep_shape=True)
            for episode_indices in split_into_consecutive_indices(list_of_indices):
                if len(episode_indices) == 0:
                    continue
                yield numpy.asarray([
                        feature_vals[i] for i in episode_indices
                    ], dtype=feature_vals.dtype)
        fish.delete() # Explicitly free up the memory since these are large objects

def get_decomposed_assay_cruise_episodes(fish: Fish, assay_label, pca_result, ndims=5) -> tuple:
    ''' Cruise episodes sorted from longest to shortest.
    Uses computer-predicted cruises.

    Returns
    -------
    (episodes_of_pca_decomposed_poses, frame_nums)
    '''
    episode_indices = get_behavior_episodes_from_assay(
        fish[assay_label],
        BEHAVIORS.cruise,
        AnnotationTypes.predicted,
        BASIC_FILTERS)

    episode_indices = list(filter(lambda a: len(a) >= 15, episode_indices))
    if not episode_indices:
        return [], []

    episode_indices = sorted(episode_indices, key=len, reverse=True)
    episodes = episode_indices_to_feature(
        fish[assay_label],
        episode_indices,
        'smoothed_angles')
    return [pca_result.decompose(e, ndims) for e in episodes], episode_indices
