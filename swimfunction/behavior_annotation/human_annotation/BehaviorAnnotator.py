#! /usr/bin/env python

from swimfunction.data_access import BehaviorAnnotationDataManager as behavior_data_manager
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.DotEnabledDict import DotEnabledDict
from swimfunction.data_models.SimpleStack import SimpleStack
from swimfunction.data_models.Fish import Fish
from swimfunction.context_managers import CacheContext
from PyQt5 import Qt
from ris_widget import ris_widget
from ris_widget.qwidgets import progress_thread_pool
from swimfunction.video_processing import extract_frames

from swimfunction.behavior_annotation.human_annotation.annotator_helpers import FRAMES_PER_SECOND, \
    BEHAVIORS, BEHAVIOR_NAMES, NAME_TO_COLOR, NAME_TO_SYMBOL, SYMBOL_TO_NAME, \
    WPI_OPTIONS, BEHAVIOR_KEY, Hotkey, HISTORY_SIZE, \
    FRAME_OVERLAP, FIXED_ANNOTATOR_HEIGHT, \
    rgb_to_int, to_behavior_string, to_behavior_symbol, get_read_page_task, \
    ReadPageTaskPage_FISH

from swimfunction.behavior_annotation.human_annotation.annotator_init import remodel_RisWidget_flipbook, \
    setup_assay_inputs, setup_behavior_choice_field, setup_hotkeys, setup_time_buttons, \
    update_assay_label_options

import atexit
from swimfunction import FileLocations
import pathlib

''' BEGIN NOTE 1
    This is the only location where we need to manually
    enumerate behaviors. Everywhere else is loops based on this.
'''
HOTKEY_MAP = {
    BEHAVIORS.rest: Hotkey('S', BEHAVIOR_NAMES.rest),
    BEHAVIORS.cruise: Hotkey('W', BEHAVIOR_NAMES.cruise),
    BEHAVIORS.burst: Hotkey('E', BEHAVIOR_NAMES.burst),
    BEHAVIORS.turn_cw: Hotkey('D', BEHAVIOR_NAMES.turn_cw),
    BEHAVIORS.turn_ccw: Hotkey('A', BEHAVIOR_NAMES.turn_ccw),
    BEHAVIORS.course_correction: Hotkey('C', BEHAVIOR_NAMES.course_correction),
    BEHAVIORS.unknown: Hotkey('X', BEHAVIOR_NAMES.unknown)
}
''' END NOTE 1
'''

class _DEFAULT_SAVER:
    def __init__(self, ba):
        self.ba = ba
    def save(self):
        behavior_data_manager.save_behaviors(
            self.ba.state.fish_name,
            self.ba.state.assay_label,
            current_annotations=self.ba.movements,
            annotator_array=self.ba.annotator_array,
            annotations_root=self.ba.annotations_root)

class BehaviorAnnotator(ris_widget.RisWidget):
    ''' Annotates behaviors in videos.
    This object holds behavior symbols in a list called self.movements
    In the GUI, however, we show behavior names. Convert between them
    with the helper functions in this script.
    '''
    __slots__ = [
        'annotations_root',
        'batch_size',
        'extractor',
        'nframes'
        'on_bahavior_callbacks',
        'person_who_annotates',
        'state_file',
        'undo_stack',
        'videos_root',
        'assay_box',
        'minute_box',
        'names_box',
        'second_box',
        'target_behavior'
    ]

    @staticmethod
    def limit_annotation_capability(target_behavior):
        global HOTKEY_MAP
        HOTKEY_MAP = {
            target_behavior: HOTKEY_MAP[target_behavior],
            BEHAVIORS.unknown: Hotkey('X', BEHAVIOR_NAMES.unknown) # always allow unknown.
        }

    @staticmethod
    def init_state_dict(fish_name=None, assay_label=None, position_in_seconds=None):
        return DotEnabledDict(
            fish_name = '' if fish_name is None else fish_name,
            assay_label = 0 if assay_label is None else assay_label,
            position_in_seconds = 0 if position_in_seconds is None else position_in_seconds
        )

    FORCED_WPI_OPTIONS = None # Lets us force a single assay if needed.

    # Below must be a function taking any number of arguments,
    # returning two list: list of behaviors, list of annotator names.
    # Yes, this is goofy, but now it can be dynamically changed.
    load_behaviors_fn = behavior_data_manager.load_behaviors

    def __init__(self, person_who_annotates, videos_root, annotations_root, state=None, target_behavior=BEHAVIORS.cruise):
        super().__init__()
        self.target_behavior = target_behavior
        self.extractor = None
        self.batch_size = FRAMES_PER_SECOND
        self.undo_stack = SimpleStack(maxsize=HISTORY_SIZE)
        self.person_who_annotates = person_who_annotates.strip().lower()
        self.videos_root = pathlib.Path(videos_root)
        self.annotations_root = pathlib.Path(annotations_root)
        self.state_file = self.annotations_root / 'annotator_state.pickle'
        self.state = state if state is not None else self.read_cached_state()
        # self.save is stored as a variable so we can dynamically change the save function.
        self.save = _DEFAULT_SAVER(self).save
        # To set later
        self.minute_box, self.second_box, self.names_box, self.assay_box = None, None, None, None
        self.movements, self.annotator_array, self.fish = None, None, None
        self.on_bahavior_callbacks = []
        self.nframes = 0
        def final():
            self.save()
            self.store_state()
        atexit.register(final)

    @property
    def SORTED_FISH_NAMES(self):
        ''' All fish names, sorted by name and number.
        '''
        return sorted(
            FDM.get_available_fish_names(),
            key=lambda a: (100 * ord(a[0])) + int(a[1:]))

    def register_on_behavior_change(self, callback):
        '''
        Parameters
        ----------
        callback

        Returns
        -------
        deregister_function
        '''
        self.on_bahavior_callbacks.append(callback)
        return lambda: self.on_bahavior_callbacks.remove(callback)

    def run_annotator(self, lock_assay=False, extra_fn=None):
        '''
        Parameters
        ----------
        lock_assay: bool
            Whether to disallow changing assays.
        extra_fn: function
            If not None, this function will be called just before `self.run()`
        '''
        # Load state, images, and annotations
        if not self.state.fish_name:
            self.state.fish_name = self.SORTED_FISH_NAMES[0]
        if not self.state.assay_label:
            self.state.assay_label = FDM.get_available_assay_labels(self.state.fish_name)[0]
        if self.fish is None:
            self.fish = Fish(name=self.state.fish_name).load()
        remodel_RisWidget_flipbook(self, HOTKEY_MAP)
        setup_behavior_choice_field(self)
        setup_time_buttons(self)
        setup_assay_inputs(self, lock_assay)
        setup_hotkeys(self, HOTKEY_MAP)
        # Set fixed height for annotator since we want flipbook area to expand as much as possible.
        self.annotator.setFixedHeight(FIXED_ANNOTATOR_HEIGHT)
        self.set_input_box_values(
            self.state.fish_name,
            self.state.assay_label,
            self.state.position_in_seconds)
        self.save_and_load()
        if extra_fn is not None:
            extra_fn()
        self.run()

    def store_state(self):
        with CacheContext.CacheContext(self.state_file) as cache:
            cache.saveContents(self.state.as_dict())

    def read_cached_state(self):
        s = BehaviorAnnotator.init_state_dict()
        if self.state_file.exists():
            with CacheContext.CacheContext(self.state_file) as cache:
                if cache.getContents() is not None:
                    s = DotEnabledDict().from_dict(cache.getContents())
        return s

    def setEnabled(self, isEnabled):
        self.flipbook.setEnabled(isEnabled)
        self.annotator.layout().setEnabled(isEnabled)

    ''' Changing time, assay, and fish
    '''

    def set_batch_size_as_behavior_size(self, location, behavior=None):
        self.batch_size = 0
        for i in range(location, self.nframes):
            if (behavior is not None and self.movements[i] != behavior) \
                or (behavior is None and self.movements[i] == BEHAVIORS.unknown):
                break
            self.batch_size += 1

    def scan_to(self, find_next_fn, behavior=None):
        ''' find_next_fn should be find_next_unknown or find_next_annotated
        '''
        fish_start = self.SORTED_FISH_NAMES.index(self.state.fish_name)
        page_idx = self.flipbook.current_page_idx
        frame_start = self.page_idx_to_annotation_idx(page_idx) + 1
        for fish_i in range(fish_start, len(self.SORTED_FISH_NAMES)):
            fish_name = self.SORTED_FISH_NAMES[fish_i]
            assay_labels = sorted(WPI_OPTIONS)
            assay_label_start = 0 if fish_name != self.state.fish_name else assay_labels.index(self.state.assay_label)
            for assay_label_i in range(assay_label_start, len(assay_labels)):
                assay_label = assay_labels[assay_label_i]
                # NOTE: because we cannot annotate the first second, 65 is considered the start.
                # Sorry for the stupid magic number, folks!
                frame_start = 65 if fish_name != self.state.fish_name else frame_start
                location = find_next_fn(fish_name, assay_label, frame_start)
                if location is not None:
                    seconds = location // 70 if location > 70 else 1
                    if self.state.fish_name != fish_name \
                        or self.state.assay_label != assay_label \
                            or self.state.position_in_seconds != seconds:
                        self.set_input_box_values(fish_name, assay_label, seconds)
                        self.set_batch_size_as_behavior_size(location, behavior)
                        self.save_and_load(location)
                    self.go_to_frame(location)
                    return
        print('Scan failed. No such frame found!')

    def set_input_box_values(self, fish_name=None, assay_label=None, seconds=None):
        ''' Sets values in the boxes.
        '''
        if fish_name is not None \
                and (self.extractor is None \
                    or (self.extractor.fish_name != fish_name and self.extractor.assay != assay_label)):
            # Need a new extractor and new frame count.
            self.extractor = extract_frames.Extractor(
                self.fish.name,
                self.state.assay_label,
                frame_nums_are_full_video=False,
                videos_root=self.videos_root)
            self.set_nframes(self.extractor.nframes)
        if self.names_box is not None and fish_name is not None:
            self.names_box.setCurrentIndex(self.names_box.findText(fish_name))
            opt = None if fish_name is None else update_assay_label_options(self, fish_name)
            self.set_assay_label_options(options=opt)
        if self.assay_box is not None and assay_label is not None:
            self.assay_box.setCurrentIndex(self.assay_box.findText(str(assay_label)))
        if self.minute_box is not None and seconds is not None:
            self.minute_box.setValue(seconds // 60)
            self.second_box.setValue(seconds % 60)

    def undo(self):
        ''' Call the undo function at position undo_i, set that position to None, then move undo_i backward.
        '''
        fn = self.undo_stack.get()
        if fn is not None:
            fn()
            self.annotator.update_fields() # This must be called to update the GUI

    def set_nframes(self, nframes):
        self.nframes = nframes
        num_mins = self.nframes // (FRAMES_PER_SECOND*60)
        num_secs = self.nframes // (FRAMES_PER_SECOND)
        self.minute_box.setRange(0, num_mins)
        self.second_box.setRange(0, 59 if num_mins > 0 else num_secs)

    def set_assay_label_options(self, options=None):
        self.assay_box.clear()
        if self.FORCED_WPI_OPTIONS is not None:
            options = self.FORCED_WPI_OPTIONS
        if options is None:
            options = WPI_OPTIONS
            if self.fish is not None:
                options = self.fish.swim_keys()
        for x in sorted(options):
            self.assay_box.insertItem(0, str(x))

    def save_and_load(self, start_frame=None):
        fish_name, assay_label = self.names_box.currentText(), self.assay_box.currentText()
        if not fish_name or assay_label is None or (isinstance(assay_label, str) and len(assay_label) == 0):
            return
        self.setEnabled(False)
        self.save() # Before anything else, save
        # Now we can set new stuff.
        print('Loading!')
        self.state.fish_name = fish_name
        if self.fish is None or self.state.fish_name != self.fish.name:
            self.fish = Fish(name=self.state.fish_name).load()
        self.state.assay_label = int(assay_label)
        self.extractor = extract_frames.Extractor(
            self.fish.name,
            self.state.assay_label,
            frame_nums_are_full_video=False,
            videos_root=self.videos_root)
        self.set_nframes(self.extractor.nframes)
        self.movements, self.annotator_array = BehaviorAnnotator.load_behaviors_fn(
            fish_name, self.state.assay_label, self.nframes, annotations_root=self.annotations_root)
        self.state.position_in_seconds = self.minute_box.value() * 60 \
            + self.second_box.value()
        batch_start = self.state.position_in_seconds * FRAMES_PER_SECOND
        if start_frame is not None:
            batch_start = start_frame
        batch_start -= FRAME_OVERLAP
        batch_end = batch_start + self.batch_size + 2 * FRAME_OVERLAP
        # Make sure position never goes negative or farther than the end of the video.
        if batch_start < 0:
            batch_start = 0
            self.state.position_in_seconds = 0
            batch_end = batch_start + self.batch_size + FRAME_OVERLAP
        if batch_start >= self.nframes:
            batch_start = self.nframes - FRAMES_PER_SECOND - FRAME_OVERLAP
            batch_end = self.nframes
            self.state.position_in_seconds = self.nframes // FRAMES_PER_SECOND
        self.minute_box.setValue(self.state.position_in_seconds // 60)
        self.second_box.setValue(self.state.position_in_seconds % 60)
        self.annotator.update_fields() # This must be called to update the GUI
        self.report_progress()
        fnums = range(max((batch_start, 0)), min((batch_end, self.nframes)))
        if len(fnums) > 0:
            self.queue_page_creation_tasks(fnums)
        self.qt_object.zoom_editor.clearFocus()
        self.flipbook.setFocus()
        self.store_state()
        self.setEnabled(True)
        self.batch_size = FRAMES_PER_SECOND
        return self.annotator.all_annotations

    def go_to_frame(self, frame_num):
        page_idx = None
        for page_idx, page in enumerate(self.flipbook_pages):
            if page.fnum == frame_num:
                break
        if page_idx is not None:
            self.flipbook.current_page_idx = page_idx
            self.annotator.update_fields() # This must be called to update the GUI

    def load_next_batch(self):
        self.set_input_box_values(seconds=self.state.position_in_seconds+1)
        return self.save_and_load()

    def load_previous_batch(self):
        self.set_input_box_values(seconds=self.state.position_in_seconds-1)
        return self.save_and_load()

    def queue_page_creation_tasks(self, fnums):
        # Modified from RisWidget flipbook.py
        fb = self.flipbook
        fb.pages.clear()
        self.annotator.all_annotations = []
        if not hasattr(fb, 'thread_pool'):
            fb.thread_pool = progress_thread_pool.ProgressThreadPool(
                fb.cancel_page_creation_tasks, fb.layout)

        page_futures = []
        for fnum in fnums:
            behavior_name = to_behavior_string(self.movements[fnum])
            task_page = ReadPageTaskPage_FISH(fnum, behavior_name, self.annotator_array[fnum])
            # NB: below sets up a cyclic reference:
            # the future holds a reference to the task page via its on_error_args param
            # and the task page holds a reference to the future via its cancel method
            future = fb.thread_pool.submit(
                get_read_page_task(self), task_page,
                on_error=fb._on_task_error, on_error_args=(task_page, ))
            task_page.page.on_removal = future.cancel
            fb.pages.append(task_page.page)
            page_futures.append(future)
        fb.ensure_page_focused()
        self.annotator.update_fields() # This must be called to update the GUI
        return page_futures

    def report_progress(self):
        if self.nframes > 0:
            annotated = behavior_data_manager.count_annotated(self.movements)
            progress = annotated / len(self.movements) if len(self.movements) > 0 else 0
            print(f'Labeled {annotated} frames ({100 * progress:.2f}%)')

    def on_behavior_chosen(self, behavior_name: str, waterfall: bool):
        ''' Sets movement for the current page. If waterfall, then also replaces
        all visible pages before the current page that match the former annotation
        of the current page.

        Parameters
        ----------
        behavior_name : str
            name of the behavior (example: 'Cruise (W)')
        waterfall : bool
            whether to also apply annotation to all
            previous unannotated frames until an annotation is found
        '''
        def run_all(fns):
            for fn in fns:
                fn()
            self.annotator.update_fields() # This must be called to update the GUI
        # print('Setting behavior:', behavior_name)
        new_symbol = NAME_TO_SYMBOL[behavior_name]
        page_idx = self.flipbook.current_page_idx
        old_symbol = self.movements[self.page_idx_to_annotation_idx(page_idx)]
        # print(old_symbol, new_symbol)
        actions = []
        undo_actions = []
        actions.append(lambda i=page_idx, s=new_symbol: self.update_annotation(i, SYMBOL_TO_NAME[s]))
        undo_actions.append(lambda i=page_idx, s=old_symbol: self.update_annotation(i, SYMBOL_TO_NAME[s]))
        if waterfall:
            previous_idx = page_idx - 1
            while previous_idx >= 0 and self.movements[self.page_idx_to_annotation_idx(previous_idx)] == old_symbol:
                actions.append(lambda i=previous_idx, s=new_symbol: self.update_annotation(i, SYMBOL_TO_NAME[s]))
                undo_actions.append(lambda i=previous_idx, s=old_symbol: self.update_annotation(i, SYMBOL_TO_NAME[s]))
                previous_idx -= 1
        run_all(actions)
        self.undo_stack.put(lambda fns=undo_actions: run_all(fns))

    def update_annotation(self, page_idx, behavior_name):
        person = self.person_who_annotates if to_behavior_symbol(behavior_name) != BEHAVIORS.unknown else None
        frame_num = self.page_idx_to_annotation_idx(page_idx)
        self.movements[frame_num] = to_behavior_symbol(behavior_name)
        self.annotator_array[frame_num] = person
        self.flipbook_pages[page_idx].annotations.update({BEHAVIOR_KEY: behavior_name})
        self.flipbook_pages[page_idx].color = Qt.QColor(rgb_to_int(NAME_TO_COLOR[behavior_name]))
        self.flipbook_pages[page_idx].name = f'{frame_num}    {person}' if person is not None else str(frame_num)
        self.annotator.update_fields() # This must be called to update the GUI
        for fn in self.on_bahavior_callbacks:
            fn()

    def page_idx_to_annotation_idx(self, page_idx):
        return self.flipbook_pages[page_idx].fnum

if __name__ == '__main__':
    ARGS = FileLocations.parse_default_args(
        lambda parser: parser.add_argument(
            'annotator', help='Name of person assigning behaviors. Basically, what\'s your name?'),
        lambda parser: parser.add_argument(
            '-v', '--videos_root', default=None,
            help='Location of videos. Default: FileLocations.get_normalized_videos_dir()'),
        lambda parser: parser.add_argument(
            '-a', '--annotations_root', default=None,
            help='Location to save annotations. Default: FileLocations.get_behaviors_model_dir()'),
        lambda parser: parser.add_argument(
            '-n', '--fish_name', help='Name of fish to start annotating', default=None),
        lambda parser: parser.add_argument(
            '-w', '--assay_label', help='Assay label to start annotating', default=None, type=int),
        lambda parser: parser.add_argument(
            '-f', '--frame_num', help='Frame number to begin annotating', default=0, type=int),
    )
    STATE = None
    if ARGS.fish_name is not None \
        or ARGS.assay_label is not None:
        STATE = BehaviorAnnotator.init_state_dict(
            fish_name=ARGS.fish_name,
            assay_label = ARGS.assay_label,
            position_in_seconds = ARGS.frame_num // FRAMES_PER_SECOND
        )
    PERSON = ARGS.annotator
    VIDEOS_ROOT = ARGS.videos_root
    ANNOTATIONS_ROOT = ARGS.annotations_root
    if VIDEOS_ROOT is None:
        VIDEOS_ROOT = FileLocations.get_normalized_videos_dir()
    if ANNOTATIONS_ROOT is None:
        ANNOTATIONS_ROOT = FileLocations.get_behaviors_model_dir()
    ALL_META_LABELS = behavior_data_manager.get_all_training_meta_labels(ANNOTATIONS_ROOT)
    behavior_data_manager.report_meta_labels_summary(ALL_META_LABELS, 'Manually Annotated')
    BEHAVIOR_ANNOTATOR = BehaviorAnnotator(
        person_who_annotates=PERSON,
        videos_root=VIDEOS_ROOT,
        annotations_root=ANNOTATIONS_ROOT,
        state=STATE)
    BEHAVIOR_ANNOTATOR.run_annotator()
