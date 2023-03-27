''' Functions to initialize the annotator.
Basically sets up the GUI appearance.

RWA = RisWidget Annotator
'''

from swimfunction.behavior_annotation.human_annotation.annotator_helpers import FRAMES_PER_SECOND, \
    BEHAVIORS, NAME_TO_COLOR, NAME_TO_SYMBOL, SYMBOL_TO_NAME, WPI_OPTIONS, \
    BEHAVIOR_KEY, \
    rgb_to_int
from PyQt5 import Qt
from swimfunction.data_access.fish_manager import DataManager as FDM
from ris_widget.qwidgets import annotator

def _add_button(layout, title, callback):
    ''' Qt boilerplate
    '''
    button = Qt.QPushButton(title)
    button.clicked.connect(callback)
    layout.addWidget(button)
    return button

def _get_spin_box(low, high):
    ''' Qt boilerplate
    '''
    sb = Qt.QSpinBox()
    sb.setRange(low, high)
    sb.setSingleStep(1)
    return sb

def _get_combo_box(options) -> Qt.QComboBox:
    ''' Qt boilerplate
    '''
    box = Qt.QComboBox()
    for x in options:
        box.insertItem(0, str(x))
    return box

def _get_find_annotation_fn(behavior='any_annotated', reverse=False):
    ''' Get a function that will take a fish, assay, and start position
    and find the next instance of a specific behavior.

    Parameters
    ----------
    behavior : str, default='any_annotated'
        could be a BEHAVIORS element, or 'any_annotated'

    Returns
    -------
    function
        Method to find a target behavior
        given a fish, assay and start frame number.
    '''
    def find_fn(fish: str, assay_label: int, start: int):
        from swimfunction.behavior_annotation.human_annotation.BehaviorAnnotator import BehaviorAnnotator
        behaviors, _annotator_array = BehaviorAnnotator.load_behaviors_fn(fish, assay_label)
        if behaviors is None:
            return None
        i = start # assert i exists (paranoid habit, I know python probably doesn't require this line).
        to_check = list(range(start, len(behaviors)))
        if reverse:
            to_check = list(reversed(range(0, start))) + list(reversed(range(start, len(behaviors))))
        for i in to_check:
            if behaviors[i] == behavior or (behavior == 'any_annotated' and behaviors[i] != BEHAVIORS.unknown):
                break
        else:
            # Did not find it.
            return None
        return i
    return find_fn

def update_assay_label_options(RWA, fish_name):
    ''' Sets options for assay_label dropdown box.
    If the annotator's member variable FORCED_WPI_OPTIONS is not None,
    then it will use these values instead.
    '''
    options = FDM.get_available_assay_labels(fish_name)
    if RWA.FORCED_WPI_OPTIONS is not None:
        options = RWA.FORCED_WPI_OPTIONS
    if RWA.assay_box is not None:
        RWA.set_assay_label_options(options)

def remodel_RisWidget_flipbook(RWA, HOTKEY_MAP):
    ''' Change RisWidget to be more fish-video-work friendly.
    '''
    RWA.qt_object.histogram_dock_widget.close() # Close the histogram viewer.
    RWA.flipbook.pages_model.EDITABLE = False # Don't allow pages to be renamed.
    RWA.flipbook.delete_button.disconnect() # Disconnect its delete page function
    RWA.flipbook.delete_button = Qt.QPushButton('Delete pages')
    RWA.flipbook.delete_button.disconnect()
    RWA.flipbook.delete_button.setEnabled(False)
    RWA.flipbook.merge_button = Qt.QPushButton('Merge pages')
    RWA.flipbook.merge_button.setEnabled(False)
    textGrid = Qt.QGridLayout()
    # Add helpful list of names
    for i, (s, hotkey) in enumerate(HOTKEY_MAP.items()):
        n = SYMBOL_TO_NAME[s]
        c = NAME_TO_COLOR[n]
        # Display n with color NAME_TO_COLOR[n]
        key = hotkey.key
        label_s = f'{n} : {key}'
        label = Qt.QLabel(label_s)
        p = Qt.QPalette()
        p.setColor(Qt.QPalette.Foreground, Qt.QColor(rgb_to_int(c)))
        label.setPalette(p)
        row = i // 3
        col = i % 3
        textGrid.addWidget(label, row, col)
    RWA.flipbook.layout().addLayout(textGrid)

def setup_assay_inputs(RWA, lock_assay):
    widget = Qt.QGroupBox('Assay Settings')
    assay_layout = Qt.QHBoxLayout(widget)
    RWA.names_box = _get_combo_box(
        reversed(RWA.SORTED_FISH_NAMES) if not lock_assay else [RWA.state.fish_name])
    RWA.assay_box = _get_combo_box(sorted(WPI_OPTIONS) if not lock_assay else [RWA.state.assay_label])
    RWA.names_box.activated.connect(
        lambda i: update_assay_label_options(RWA, RWA.names_box.itemText(i)))
    RWA.set_assay_label_options()
    assay_layout.addWidget(Qt.QLabel('Fish'))
    assay_layout.addWidget(RWA.names_box)
    assay_layout.addWidget(Qt.QLabel('Assay'))
    assay_layout.addWidget(RWA.assay_box)
    RWA.names_box.setMinimumSize(75, 10) # RWA.names_box.setFixedWidth(75)
    RWA.assay_box.setMinimumSize(60, 10) # RWA.assay_box.setFixedWidth(60)

    RWA.minute_box = _get_spin_box(0, RWA.nframes // (FRAMES_PER_SECOND*60))
    RWA.minute_box.setSuffix('m')
    RWA.second_box = _get_spin_box(0, 59)
    RWA.second_box.setSuffix('s')

    time_layout = Qt.QHBoxLayout(widget)
    time_layout.addWidget(Qt.QLabel('Position in Video'))
    time_layout.addWidget(RWA.minute_box)
    time_layout.addWidget(Qt.QLabel(':'))
    time_layout.addWidget(RWA.second_box)
    def load_fn():
        RWA.batch_size = FRAMES_PER_SECOND
        RWA.save_and_load()
    _add_button(time_layout, 'Load', load_fn)

    RWA.annotator.layout().insertRow(0, time_layout)
    RWA.annotator.layout().insertRow(0, widget)
    RWA.annotator.layout().insertRow(0, assay_layout)

def setup_behavior_choice_field(RWA):
    # fields must contain all possible values in the GUI fields box.
    b_names_list = list(NAME_TO_SYMBOL.keys())
    cf = annotator.ChoicesField(BEHAVIOR_KEY, b_names_list)
    cf.widget.activated.connect(lambda i: RWA.on_behavior_chosen(b_names_list[i], waterfall=False))
    fields = [cf]
    RWA.add_annotator(fields)

def setup_hotkeys(RWA, HOTKEY_MAP):
    RWA.add_action('Ctrl+S is Save', Qt.QKeySequence("Ctrl+S"), RWA.save)
    RWA.add_action('Ctrl+Z is Undo', Qt.QKeySequence("Ctrl+Z"), RWA.undo)
    RWA.add_action('Ctrl+Right is next batch', Qt.QKeySequence('Ctrl+Right'), RWA.load_next_batch)
    RWA.add_action('Ctrl+Left is next batch', Qt.QKeySequence('Ctrl+Left'), RWA.load_previous_batch)
    RWA.add_action(
        'Crtl+Up is next cruise',
        Qt.QKeySequence('Ctrl+Up'),
        lambda: RWA.scan_to(_get_find_annotation_fn(RWA.target_behavior), RWA.target_behavior))
    RWA.add_action(
        'Crtl+Up is next cruise',
        Qt.QKeySequence('Ctrl+Down'),
        lambda: RWA.scan_to(_get_find_annotation_fn(RWA.target_behavior, reverse=True), RWA.target_behavior))
    print('\nCtrl+S\tsave\nCtrl+Z\tundo\nCtrl+Right\tnext batch\nCtrl+Left\tprevious batch\n')
    for tup in HOTKEY_MAP.values():
        print(f'{tup.key} is {tup.name}')
        RWA.add_action(
            f'{tup.key} is {tup.name}',
            Qt.QKeySequence(tup.key),
            lambda _, n=tup.name: RWA.on_behavior_chosen(n, waterfall=False))
        RWA.add_action(
            f'{tup.key} is {tup.name}',
            Qt.QKeySequence(f'Shift+{tup.key}'),
            lambda _, n=tup.name: RWA.on_behavior_chosen(n, waterfall=True))
    print('Adding shift assigns behaviors to prior unannotated frames\n')

def setup_time_buttons(RWA):
    widget = Qt.QGroupBox('Movement Annotations')
    buttonsLayout = Qt.QVBoxLayout(widget)
    _add_button(buttonsLayout, 'Undo', lambda: RWA.undo())
    _add_button(buttonsLayout, 'Save', lambda: RWA.save())
    _add_button(buttonsLayout, 'Back one second', lambda: RWA.load_previous_batch())
    _add_button(buttonsLayout, 'Forward one second', lambda: RWA.load_next_batch())

    scanLayout = Qt.QHBoxLayout()
    _add_button(
        scanLayout,
        'Scan to next unknown',
        lambda: RWA.scan_to(_get_find_annotation_fn(BEHAVIORS.unknown), BEHAVIORS.unknown))
    _add_button(
        scanLayout,
        'Scan to next annotated',
        lambda: RWA.scan_to(_get_find_annotation_fn()))
    _add_button(
        scanLayout,
        'Scan to next cruise',
        lambda: RWA.scan_to(_get_find_annotation_fn(BEHAVIORS.cruise), BEHAVIORS.cruise))
    # _add_button(
    #     scanLayout,
    #     'Scan to next burst',
    #     lambda: RWA.scan_to(_get_find_annotation_fn(BEHAVIORS.burst), BEHAVIORS.burst))
    _add_button(
        scanLayout,
        'Scan to next rest',
        lambda: RWA.scan_to(_get_find_annotation_fn(BEHAVIORS.rest), BEHAVIORS.rest))

    RWA.annotator.layout().insertRow(0, widget)
    RWA.annotator.layout().insertRow(0, scanLayout)
