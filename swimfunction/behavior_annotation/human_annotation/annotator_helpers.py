''' The BehaviorAnnotator file was getting too huge.
This file contains helper functions to separate some of the code.
'''
from collections import namedtuple
from swimfunction.global_config.config import config
from PyQt5 import Qt
from ris_widget.qwidgets import flipbook
from swimfunction.video_processing import CropTracker
import numpy
from swimfunction.video_processing.fp_ffmpeg import transpose_img

########### If you want to know the exact frame number
# (from the original video) and y-position.
PRINT_FRAME_NUM_AND_AVG_Y = False
###########


FRAMES_PER_SECOND = config.getint('VIDEO', 'fps')
BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
BEHAVIOR_NAMES = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'names')
NAME_TO_COLOR = config.getfloatdict('BEHAVIORS', 'names', 'BEHAVIORS', 'colors')
NAME_TO_COLOR[BEHAVIOR_NAMES.unknown] = (0, 0, 0)
WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')

NAME_TO_SYMBOL = dict(zip(BEHAVIOR_NAMES, BEHAVIORS))
SYMBOL_TO_NAME = dict(zip(BEHAVIORS, BEHAVIOR_NAMES))

BEHAVIOR_KEY = 'Behavior'
Hotkey = namedtuple('Hotkey', ['key', 'name'])
HISTORY_SIZE = 15 # Number of actions to remember for undo function
FRAME_OVERLAP = 5 # Number of frames to show from previous and next batch.

FIXED_ANNOTATOR_HEIGHT = 360

TRANSPOSE_IMG = False

def add_pose_to_frame(
        img: numpy.ndarray,
        frame_num: int,
        pose: numpy.ndarray,
        transpose: bool=False):
    ''' Given identifying information, get a frame from the video.
        If pose is provided, will plot the pose on the frame.

    Parameters
    ----------
    img : numpy.ndarray
    frame_num : int
    pose : numpy.ndarray
        must be a coordinate pose, shape=(n_points, 2)
    transpose : bool
        whether to transpose into a conventional top-left-origin format.
    '''
    pose = pose.astype(int)
    if transpose:
        # Swap x and y since it's conventional.
        pose = numpy.vstack((pose[:, 1], pose[:, 0])).T
    pose[pose < 0] = 0
    pose[:, 0][pose[:, 0] >= img.shape[0]] = img.shape[0] - 1
    pose[:, 1][pose[:, 1] >= img.shape[1]] = img.shape[1] - 1
    img[pose[:, 0], pose[:, 1], 0] = img.max()
    img[pose[:, 0], pose[:, 1], 1] = 0
    img[pose[:, 0], pose[:, 1], 2] = 0
    if PRINT_FRAME_NUM_AND_AVG_Y:
        print(frame_num, pose[:, 1].mean())
    return img

def to_video_fnum(BA, original_frame_num):
    video_frame_num = original_frame_num
    if BA.extractor.ct_logger is not None:
        video_frame_num = BA.extractor.ct_logger.original_frame_num_to_crop_track_frame_num(
            original_frame_num)
    return video_frame_num

def process_img(BA, original_frame_num, video_frame_num, img):
    if video_frame_num is not None:
        corner = CropTracker.Point(0, 0)
        if BA.extractor.ct_logger is not None:
            corner = BA.extractor.ct_logger.corners[video_frame_num]
        pose = CropTracker.to_fish_frame(
            BA.fish[BA.state.assay_label].smoothed_coordinates[original_frame_num],
            corner,
            is_transposed=TRANSPOSE_IMG)
        if img is None:
            img = numpy.ones((400, 400, 3))
        img = add_pose_to_frame(
            img.copy(),
            original_frame_num,
            pose,
            transpose=TRANSPOSE_IMG)
    else:
        img = numpy.ones((100, 100, 3))
    return img

def get_read_page_task(BA):
    ''' For async page loading.
    '''
    def inner(task_page):
        original_frame_num = task_page.fnum
        video_frame_num = to_video_fnum(BA, original_frame_num)
        img = BA.extractor.extract_frame(
            video_frame_num,
            transpose_to_match_convention=TRANSPOSE_IMG)
        img = process_img(BA, original_frame_num, video_frame_num, img)
        # Modified from RisWidget flipbook.py
        task_page.ims = [img]
        Qt.QApplication.instance().postEvent(
            BA.flipbook, flipbook._ReadPageTaskDoneEvent(task_page))
    return inner

def rgb_to_int(rgb_arr):
    ''' Takes rgb or rgba lists, converts to hex integer.
    '''
    rgb = int(rgb_arr[0] * 255)
    rgb = (rgb << 8) + int(rgb_arr[1] * 255)
    rgb = (rgb << 8) + int(rgb_arr[2] * 255)
    return rgb

def to_behavior_string(symbol):
    ''' Handy conversion for string or list.

    Parameters
    ----------
    symbol : str or list
        must be a symbol in BEHAVIORS

    Returns
    -------
    name or names : str or list
        name for behavior symbol.
    '''
    if isinstance(symbol, (list, numpy.ndarray)):
        return numpy.asarray([SYMBOL_TO_NAME[b] for b in symbol])
    return SYMBOL_TO_NAME[symbol]

def to_behavior_symbol(behavior_string):
    ''' Handy conversion for string or list.

    Parameters
    ----------
    symbol: str or list
        name for behavior symbol.

    Returns
    -------
    symbol or symbols : str or list
        behavior symbol for name string.
    '''
    if isinstance(behavior_string, (list, numpy.ndarray)):
        return numpy.asarray([NAME_TO_SYMBOL[b] for b in behavior_string])
    return NAME_TO_SYMBOL[behavior_string]

class ReadPageTaskPage_FISH:
    ''' Necessary task to read a page... not entirely sure I get it, but it works.
    '''
    __slots__ = ["page", "fnum", "im_names", "ims"]
    def __init__(self, fnum, behavior_name, person):
        self.page = flipbook.ImageList()
        self.page.name = f'{fnum}    {person}' if person is not None else str(fnum)
        self.page.fnum = int(fnum)
        self.fnum = int(fnum)
        self.im_names = [str(fnum)]
        self.page.annotations = {BEHAVIOR_KEY: behavior_name}
        self.page.color = Qt.QColor(rgb_to_int(NAME_TO_COLOR[behavior_name]))
