#! /usr/bin/env python3

from collections import defaultdict
import numpy

from swimfunction.behavior_annotation.human_annotation import annotator_init
from swimfunction.behavior_annotation.human_annotation.BehaviorAnnotator import BehaviorAnnotator
from swimfunction.behavior_annotation.RestPredictor import RestPredictor
from swimfunction.behavior_annotation.UmapClassifier import UmapClassifier
from swimfunction.data_access import BehaviorAnnotationDataManager as BDM
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import PoseAccess
from swimfunction.data_models.Fish import Fish
from swimfunction.context_managers.CacheContext import CacheContext
from swimfunction.pose_processing import pose_filters
from swimfunction.global_config.config import config

from swimfunction import FileLocations
from swimfunction import progress

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')

NO_FLOW = config.getint('FLOW', 'none')
SLOW_FLOW = config.getint('FLOW', 'slow')
FAST_FLOW = config.getint('FLOW', 'fast')

NO_FLOW_START = config.getint('FLOW', 'none_start')
SLOW_FLOW_START = config.getint('FLOW', 'slow_start')
FAST_FLOW_START = config.getint('FLOW', 'fast_start')

PRIORITY_ASSAYS = [-1, 1, 6, 3, 8, 2, 4, 5, 7]
# PRIORITY_ASSAYS = [8]

# All assays this and higher are chosen for annotation
TARGET_MIN_RECOVERY = 1

TARGET_BEHAVIOR = BEHAVIORS.cruise
TARGET_BEHAVIOR_NAME = 'cruise'
MAX_BEHAVIOR_EPISODE_LENGTH = 30

def get_flow_speed(i):
    ''' Get the flow speed given a frame number.
    '''
    if i < SLOW_FLOW_START:
        return NO_FLOW
    elif i >= SLOW_FLOW_START and i < FAST_FLOW_START:
        return SLOW_FLOW
    elif i >= FAST_FLOW_START:
        return FAST_FLOW
    return None

def require_minimum_length(list_of_indices, target_behavior, min_len):
    ''' Mask out any behavior episodes shorter than min_len.
    '''
    tmp = numpy.asarray(list_of_indices)
    list_of_indices = numpy.where(tmp == target_behavior)[0]
    episodes = PoseAccess.split_into_consecutive_indices(list_of_indices)
    assert len(list_of_indices) == sum([len(e) for e in episodes]), 'Did not find all episodes.'
    for episode in episodes:
        if len(episode) < min_len:
            for i in episode:
                list_of_indices[i] = BEHAVIORS.unknown
    return list_of_indices

def enforce_maximum_length(list_of_indices):
    rv = []
    for ep_idx in list_of_indices:
        if len(ep_idx) > MAX_BEHAVIOR_EPISODE_LENGTH:
            half = MAX_BEHAVIOR_EPISODE_LENGTH // 2
            rv.append(ep_idx[:half])
            rv.append(ep_idx[-half:])
        else:
            rv.append(ep_idx)
    return rv

def get_only_longest_episodes(
        labels, target_behavior, num_to_return_per_flow_speed) -> numpy.ndarray:
    ''' Returns the longest episodes of target behavior for each flow speed.
    '''
    list_of_indices = numpy.where(numpy.asarray(labels) == target_behavior)[0]
    episodes_idx = PoseAccess.split_into_consecutive_indices(list_of_indices)
    assert len(list_of_indices) == sum([len(e) for e in episodes_idx]), 'Did not find all episodes.'
    def get_longest(_eidx):
        longest = _eidx
        if len(longest) > num_to_return_per_flow_speed:
            longest = sorted(longest, key=len)[-num_to_return_per_flow_speed:]
        if MAX_BEHAVIOR_EPISODE_LENGTH is not None:
            longest = enforce_maximum_length(longest)
        return longest
    def get_limit_by_flow_fn(flow_speed):
        return lambda x: get_flow_speed(x[0]) == flow_speed
    low = get_longest(list(filter(get_limit_by_flow_fn(NO_FLOW), episodes_idx)))
    med = get_longest(list(filter(get_limit_by_flow_fn(SLOW_FLOW), episodes_idx)))
    fast = get_longest(list(filter(get_limit_by_flow_fn(FAST_FLOW), episodes_idx)))
    final_episodes = low + med + fast
    final_labels = numpy.full_like(labels, BEHAVIORS.unknown)
    final_labels[numpy.concatenate(final_episodes)] = target_behavior
    return final_labels

def get_fish_with_at_least_target_recovered_quality(target_recovery_quality) -> list:
    fish_names = []
    for name in FDM.get_available_fish_names():
        recovered_quality = FDM.get_highest_recovered_quality(name)
        if recovered_quality is not None and recovered_quality >= target_recovery_quality:
            fish_names.append(name)
    return sorted(fish_names, key=FDM.get_highest_recovered_quality)

def generate_sorted_fish_names_and_assays(target_recovery_quality):
    ''' Get name-assay tuples for all available assays.
    Sorted by which assays are most useful for us.
    Only chooses the fish that have at least the target best recovery score.
    '''
    fish_names = FDM.get_available_fish_names()
    if target_recovery_quality is not None:
        fish_names = get_fish_with_at_least_target_recovered_quality(target_recovery_quality)
    assay_labels = PRIORITY_ASSAYS
    for assay_label in assay_labels:
        for name in fish_names:
            if assay_label in FDM.get_available_assay_labels(name):
                yield name, assay_label

def ask_whether_to_save():
    should_save = None
    while should_save not in ['Y', 'N']:
        should_save = input(
            f'Save all {TARGET_BEHAVIOR} annotations (ONLY NEW BEHAVIORS WILL BE SAVED)? (Y = yes) '
        ).strip().upper()
    return should_save == 'Y'

def ask_whether_to_continue():
    decision = None
    while decision not in ['Y', 'N']:
        decision = input('Ready to continue? (Y = yes, N = exit) ').strip().upper()
    return decision == 'Y'

class MLAssistedBehaviorAnnotator:
    ''' Presents predicted TARGET_BEHAVIOR episodes for a human to review.
    Repeatedly opens an annotator with
    the next assay that needs annotation.
    '''
    def __init__(
            self, person_who_annotates, videos_root, annotations_root,
            behavior_predictor=None):
        self.deregister_annotator = lambda: 1
        self.predictor = behavior_predictor
        self.person_who_annotates = person_who_annotates.strip().lower()
        self.videos_root = videos_root
        self.annotations_root = annotations_root
        self.cleaned_predictions = defaultdict(dict)
        self.checklist_cache_fname = FileLocations.get_behaviors_model_dir() \
            / f'ML_{TARGET_BEHAVIOR_NAME}_checklist.txt'
        with CacheContext(self.checklist_cache_fname) as cache:
            contents = cache.getContents()
            self.checklist = [] if contents is None else contents.strip().split('\n')

    def annotate_with_assistance(self, total_as_denominator=True):
        ''' Repeatedly opens an annotator with
        the next assay that needs annotation.
        '''
        print('Running refining process...')
        fish_assay_tuples = list(generate_sorted_fish_names_and_assays(TARGET_MIN_RECOVERY))
        total = len(fish_assay_tuples)
        filtered_tuples = list(filter(lambda t: not self._is_checked(*t), fish_assay_tuples))
        nfish_remaining = len(set([t[0] for t in filtered_tuples]))
        remaining = len(filtered_tuples)
        print(f'{total} assays total for these conditions.')
        print(f'{nfish_remaining} fish at this level still need annotations')
        print(f'{remaining} assays remaining')
        print('Assays remaining: ' + ','.join([f'{n}_{w}'  for n, w in filtered_tuples]))
        if not total_as_denominator:
            total = remaining
        progress.init(total, start_i=(total-remaining))
        for counter, (fish_name, assay_label) in enumerate(filtered_tuples):
            print(remaining - counter, 'remaining assays to check')
            # for each video
            if self._is_checked(fish_name, assay_label):
                continue
            progress.progress(counter + (total - remaining), f'{fish_name} {assay_label}wpi')
            fish = Fish(name=fish_name).load()
            print(f'fish {fish_name}, {assay_label}wpi')
            # Predict episodes
            predictions, annotator_arr = self._predict_target_behaviors(fish, assay_label)
            n_episodes = numpy.where(predictions == TARGET_BEHAVIOR)[0].shape[0]
            if n_episodes == 0:
                print(f'No {TARGET_BEHAVIOR_NAME}s found.')
                continue
            assert len(predictions) == len(annotator_arr)
            print(f'{n_episodes} {TARGET_BEHAVIOR_NAME} frames to check')
            # Run annotator to refine predictions.
            print('Running annotator...')
            self._run_annotator(fish_name, assay_label, predictions, annotator_arr)
            should_save = ask_whether_to_save()
            should_continue = ask_whether_to_continue()
            # Save new episode annotations.
            if should_save:
                self._save_target_behaviors(fish_name, assay_label)
                self._mark_checked(fish_name, assay_label)
            if not should_continue:
                return
            try:
                self.deregister_annotator()
            except:
                # Closing the annotator shouldn't ever be a problem.
                pass
        progress.finish()

    def _mark_checked(self, fish_name, assay_label):
        self.checklist.append(f'{fish_name},{assay_label}')
        with CacheContext(self.checklist_cache_fname) as cache:
            cache.saveContents('\n'.join(self.checklist))

    def _is_checked(self, fish_name, assay_label):
        return f'{fish_name},{assay_label}' in self.checklist

    def _predict_target_behaviors(self, fish: Fish, assay_label: int):
        ''' Uses ML to predict TARGET_BEHAVIOR frames
        '''
        # Predicting on-the-fly is extremely slow.
        # Try doing one big overnight prediction for all assays,
        # then refine those.
        original_annotations, annotator_array = BDM.load_behaviors(fish.name, assay_label)
        predictions = []
        if hasattr(fish[assay_label], 'predicted_behaviors'):
            predictions = fish[assay_label].predicted_behaviors
        if annotator_array is None and predictions is not None:
            annotator_array = ['ML' for _ in predictions]
        if (predictions is None or len(predictions) == 0) and self.predictor is not None:
            print(f'Predicting {TARGET_BEHAVIOR_NAME}s...')
            predictions = self.predictor.predict_behaviors(
                PoseAccess.get_feature_from_assay(
                    fish[assay_label],
                    feature='smoothed_angles',
                    filters=pose_filters.BASIC_FILTERS,
                    keep_shape=True))
            if original_annotations is not None and TARGET_BEHAVIOR in original_annotations:
                predictions[numpy.where(original_annotations == TARGET_BEHAVIOR)] = TARGET_BEHAVIOR
        if predictions is not None and TARGET_BEHAVIOR in predictions:
            if TARGET_BEHAVIOR in predictions:
                predictions = get_only_longest_episodes(predictions, TARGET_BEHAVIOR, 2)
            for i in numpy.where(numpy.logical_not(original_annotations == TARGET_BEHAVIOR))[0]:
                annotator_array[i] = 'ML'
        return numpy.asarray(predictions), annotator_array

    def _save_target_behaviors(self, fish_name: str, assay_label: int):
        ''' Gets original annotations,
        updates with all frames that were not previously marked as TARGET_BEHAVIOR
        that are now marked as TARGET_BEHAVIOR, and updates the annotator array as well.
        '''
        cleaned = self.cleaned_predictions[fish_name][assay_label]
        # get original behaviors
        original_annotations, annotator_array = BDM.load_behaviors(fish_name, assay_label, len(cleaned))
        # set TARGET_BEHAVIOR positions, updating annotations and annotator
        # only where it was not originally labeled as TARGET_BEHAVIOR.
        for i in numpy.where(cleaned == TARGET_BEHAVIOR)[0]:
            if original_annotations[i] != TARGET_BEHAVIOR:
                original_annotations[i] = TARGET_BEHAVIOR
                annotator_array[i] = self.person_who_annotates
        # save behaviors
        BDM.save_behaviors(fish_name, assay_label, original_annotations, annotator_array)

    def _run_annotator(
            self,
            fish_name: str, assay_label: int,
            custom_behaviors: numpy.ndarray, annotator_arr: list):
        ''' Runs annotator with special save function that stores
        the cleaned predictions in self.cleaned_predictions instead of a file.
        '''
        def load_b_fn(*_args, **_kwargs):
            return custom_behaviors, annotator_arr
        BehaviorAnnotator.limit_annotation_capability(TARGET_BEHAVIOR)
        state = BehaviorAnnotator.init_state_dict(
            fish_name, assay_label, 0)
        BehaviorAnnotator.load_behaviors_fn = load_b_fn
        BehaviorAnnotator.SORTED_FISH_NAMES = [fish_name]
        BehaviorAnnotator.FORCED_WPI_OPTIONS = [assay_label]
        annotator = BehaviorAnnotator(
            self.person_who_annotates,
            self.videos_root,
            self.annotations_root,
            state=state,
            target_behavior=TARGET_BEHAVIOR)
        annotator.save = lambda: self._update_cleaned_predictions(fish_name, assay_label, annotator)
        self.deregister_annotator = annotator.register_on_behavior_change(annotator.save)
        annotator.run_annotator(
            lock_assay=True,
            extra_fn=lambda: annotator.scan_to(
                annotator_init._get_find_annotation_fn(TARGET_BEHAVIOR), TARGET_BEHAVIOR))
        self.cleaned_predictions[fish_name][assay_label] = annotator.movements

    def _update_cleaned_predictions(
            self,
            fish_name: str,
            assay_label: int,
            annotator: BehaviorAnnotator):
        self.cleaned_predictions[fish_name][assay_label] = annotator.movements

if __name__ == '__main__':
    ARGS = FileLocations.parse_default_args(
        lambda parser: parser.add_argument(
            'annotator', help='Name of person assigning behaviors using this tool. \
            Basically, what\'s your name?'),
        lambda parser: parser.add_argument(
            '-v', '--videos_root', default=None, help='Location of videos. \
            Default: FileLocations.get_normalized_videos_dir()'),
        lambda parser: parser.add_argument(
            '-a', '--annotations_root', default=None, help='Location to save annotations. \
            Default: FileLocations.get_behaviors_model_dir()')
    )
    _VIDEOS_ROOT = ARGS.videos_root
    _ANNOTATIONS_ROOT = ARGS.annotations_root
    if _VIDEOS_ROOT is None:
        _VIDEOS_ROOT = FileLocations.get_normalized_videos_dir()
    if _ANNOTATIONS_ROOT is None:
        _ANNOTATIONS_ROOT = FileLocations.get_behaviors_model_dir()
    if TARGET_BEHAVIOR == BEHAVIORS.rest:
        PREDICTOR = RestPredictor('smoothed_angles')
    else:
        PREDICTOR = UmapClassifier()
        PREDICTOR.load_models()
    PERSON = ARGS.annotator
    ANNOTATOR = MLAssistedBehaviorAnnotator(
        person_who_annotates=PERSON,
        videos_root=_VIDEOS_ROOT,
        annotations_root=_ANNOTATIONS_ROOT,
        behavior_predictor=PREDICTOR)
    ANNOTATOR.annotate_with_assistance(total_as_denominator=False)
