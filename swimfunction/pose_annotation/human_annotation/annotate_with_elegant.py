#! /usr/bin/env python3
''' Annotate centerlines of fish
NOTE: this annotates in video coordinates, so if you
used CropTracker to get the video, coordinates will
be in fish frame, not lab frame.

Page annotations are a dict of this organization:
{
    'pose': pose tuple or (None, None),
    'poses': list of poses,
    'individual': int indexing into 'poses'
}
Saved annotations are a dict of this organization:
{
    'poses': list of poses
}
'''
import argparse
import atexit
from concurrent import futures
from elegant.gui import pose_annotation
from elegant.gui.spline_overlay import spline_outline
from ris_widget import ris_widget
from ris_widget.qwidgets import progress_thread_pool
from ris_widget.qwidgets import flipbook
import numpy
import pathlib
from PyQt5 import Qt


from swimfunction.context_managers.CacheContext import CacheContext
from swimfunction.global_config.config import config
from swimfunction.pose_annotation.human_annotation.annotations_to_csv_h5 import annotations_to_csv_h5
from swimfunction.pose_annotation.to_grayscale import to_grayscale
from swimfunction.video_processing import fp_ffmpeg


POINTS_PER_POSE = config.getint('POSES', 'points_per_pose')

def _add_button(layout, title, callback):
    button = Qt.QPushButton(title)
    button.clicked.connect(callback)
    layout.addWidget(button)
    return button

class _ReadPageTaskPage_FISH:
    __slots__ = ["page", "img_path", "im_names", "ims"]

    def __init__(self, img_path, poses):
        if poses is None or len(poses) == 0:
            poses = [(None, None)]
        img_path = pathlib.Path(img_path)
        self.page = flipbook.ImageList()
        self.page.name = img_path.expanduser().resolve().as_posix()
        self.im_names = [img_path.name]
        self.page.annotations = {
            'pose': poses[0], 'poses': poses, 'individual': 0}


class _PoseAnnotation(pose_annotation.PoseAnnotation):
    ''' I inherit here so that the undo-redo feature works properly.
    '''

    def register_update_callback(self, fn):
        self.update_callback = fn

    def update_annotation(self, new_state):
        super().update_annotation(new_state)
        self.update_callback()


class PoseAnnotator(ris_widget.RisWidget):
    ''' Annotate fish poses in images.
    '''

    def __init__(self, annotation_root, scorer):
        super().__init__()
        self.pose_annotator = _PoseAnnotation(self)
        # ka = keypoint_annotation.KeypointAnnotation(
        #     self, ['head', 'tail'], worm_frame=False, auto_advance=True)
        self.pose_annotator.register_update_callback(self.on_annotation_change)
        self.poses_counter_label = None
        self.scorer = scorer
        self.recent_width_tck = None
        self.annotation_root = annotation_root
        self.current_outlines = [None for _ in range(20)] # We'll never have more than 20 poses on-screen.
        self.add_annotator([self.pose_annotator])  # , ka])
        self._remodel_RisWidget_flipbook()
        self._setup_pose_buttons()
        self.add_action('Ctrl+S is Save', Qt.QKeySequence("Ctrl+S"), self.save)
        self.add_action('C is Change Active Pose', Qt.QKeySequence("C"), self.change_active_pose)
        self.load()
        self.update_pose_count()
        self.flipbook.current_page_changed.connect(self.refresh_onscreen_poses)
        atexit.register(self.save)

    def setEnabled(self, isEnabled):
        self.flipbook.setEnabled(isEnabled)
        self.annotator.layout().setEnabled(isEnabled)

    ''' Annotator Panel Setup
    '''

    def _remodel_RisWidget_flipbook(self):
        ''' Change RisWidget to be more fish-video-work friendly.
        '''
        self.qt_object.histogram_dock_widget.close()  # Close the histogram viewer.
        # Don't allow pages to be renamed.
        self.flipbook.pages_model.EDITABLE = False
        self.flipbook.delete_button.disconnect() # Disconnect its delete function
        self.flipbook.delete_button.setEnabled(False)
        self.flipbook.delete_selected = lambda: None
        self.flipbook.merge_button.disconnect() # No merging allowed no matter what.
        self.flipbook.merge_button.setEnabled(False)
        self.flipbook.current_page_changed.connect(self.update_pose_count)

    def _setup_pose_buttons(self):
        widget = Qt.QGroupBox('Poses')
        self.poses_counter_label = Qt.QLabel('0 Poses')
        buttonsLayout = Qt.QVBoxLayout(widget)
        buttonsLayout.addWidget(self.poses_counter_label)
        _add_button(buttonsLayout, 'Add Pose', self.add_pose)
        _add_button(buttonsLayout, 'Remove Active Pose',
                    self.remove_active_pose)
        _add_button(buttonsLayout, 'Change Active Pose',
                    self.change_active_pose)
        _add_button(buttonsLayout, 'Save', self.save)
        self.annotator.layout().insertRow(0, widget)

    def save(self):
        print('Saving!')
        annotation_file = (pathlib.Path(self.annotation_root) /
                           'annotations.pickle')
        annotations = {
            fp.name: fp.annotations['poses'] for fp in self.flipbook_pages
        }
        with CacheContext(annotation_file) as cache:
            cache.saveContents(annotations)
        annotations_to_csv_h5(annotations, scorer=self.scorer, npoints=POINTS_PER_POSE, output_dir=self.annotation_root)

    def load(self):
        print('Loading...')
        annotation_file = (pathlib.Path(self.annotation_root) /
                           'annotations.pickle')
        fpaths = list(pathlib.Path(self.annotation_root).expanduser().resolve().glob('*.png'))
        self.setEnabled(False)
        self.annotator.update_fields()  # This must be called to update the GUI
        with CacheContext(annotation_file) as cache:
            cached_map = cache.getContents()
            self.queue_page_creation_tasks(fpaths, cached_map)
        self.qt_object.zoom_editor.clearFocus()
        self.flipbook.setFocus()
        self.setEnabled(True)
        return self.annotator.all_annotations

    def go_to_frame(self, frame_num):
        page_idx = None
        for page_idx, page in enumerate(self.flipbook_pages):
            if int(page.objectName()) == frame_num:
                break
        if page_idx is not None:
            self.flipbook.current_page_idx = page_idx
            self.annotator.update_fields()  # This must be called to update the GUI

    def _read_page_task(self, task_page):
        # Modified from RisWidget flipbook.py
        task_page.ims = [
            to_grayscale(
                numpy.asarray(
                    fp_ffmpeg.freeimage_read(
                        task_page.page.name)))]
        Qt.QApplication.instance().postEvent(
            self.flipbook, flipbook._ReadPageTaskDoneEvent(task_page))

    def queue_page_creation_tasks(self, fpaths, annotations_map):
        # Modified from RisWidget flipbook.py
        fb = self.flipbook
        fb.pages.clear()
        self.annotator.all_annotations = []
        if not hasattr(fb, 'thread_pool'):
            fb.thread_pool = progress_thread_pool.ProgressThreadPool(
                fb.cancel_page_creation_tasks, fb.layout)
        page_futures = []
        for fpath in fpaths:
            if not pathlib.Path(fpath).exists():
                continue
            poses = annotations_map[fpath] if annotations_map is not None and fpath in annotations_map else None
            task_page = _ReadPageTaskPage_FISH(fpath, poses)
            # NB: below sets up a cyclic reference: the future holds a reference to the task page via its on_error_args param
            # and the task page holds a reference to the future via its cancel method
            future = fb.thread_pool.submit(
                self._read_page_task, task_page, on_error=fb._on_task_error, on_error_args=(task_page, ))
            task_page.page.on_removal = future.cancel
            fb.pages.append(task_page.page)
            page_futures.append(future)
        fb.ensure_page_focused()
        futures.wait(page_futures, return_when=futures.FIRST_COMPLETED)
        self.refresh_onscreen_poses()
        self.annotator.update_fields()  # This must be called to update the GUI
        return page_futures

    def add_pose(self):
        page_idx = self.flipbook.current_page_idx
        self.flipbook_pages[page_idx].annotations['poses'].append((None, None))
        self.update_pose_count()

    def change_active_pose(self):
        self.pose_annotator.redo_stack.clear()
        self.pose_annotator.undo_stack.clear()
        page_idx = self.flipbook.current_page_idx
        n_poses = len(self.flipbook_pages[page_idx].annotations['poses'])
        current_individual = self.flipbook_pages[page_idx].annotations['individual']
        current_width = self.flipbook_pages[page_idx].annotations['poses'][current_individual][1]
        if self.recent_width_tck is None and current_width is not None:
            self.recent_width_tck = current_width
        next_individual = (current_individual + 1) % n_poses
        next_pose = self.flipbook_pages[page_idx].annotations['poses'][next_individual]
        if next_pose[1] is None:
            next_pose = (next_pose[0], self.recent_width_tck)
        self.flipbook_pages[page_idx].annotations['individual'] = next_individual
        self.flipbook_pages[page_idx].annotations['pose'] = next_pose
        self.refresh_onscreen_poses()
        self.annotator.update_fields()  # This must be called to update the GUI

    def remove_active_pose(self):
        if input('Really want to remove it? Y=yes').strip().upper() != 'Y':
            return
        self.pose_annotator.redo_stack.clear()
        self.pose_annotator.undo_stack.clear()
        page_idx = self.flipbook.current_page_idx
        n_poses = len(self.flipbook_pages[page_idx].annotations['poses'])
        current_individual = self.flipbook_pages[page_idx].annotations['individual']
        if n_poses == 1:
            self.flipbook_pages[page_idx].annotations['poses'] = [(None, None)]
        else:
            del self.flipbook_pages[page_idx].annotations['poses'][current_individual]
        self.flipbook_pages[page_idx].annotations['pose'] = \
            self.flipbook_pages[page_idx].annotations['poses'][0]
        self.flipbook_pages[page_idx].annotations['individual'] = 0
        self.update_pose_count()
        self.annotator.update_fields()  # This must be called to update the GUI

    def update_pose_count(self):
        page_idx = self.flipbook.current_page_idx
        n_poses = len(self.flipbook_pages[page_idx].annotations['poses'])
        self.poses_counter_label.setText(
            f'{n_poses} Poses' if n_poses != 1 else '1 Pose')
        self.annotator.update_fields()  # This must be called to update the GUI

    def on_annotation_change(self):
        page_idx = self.flipbook.current_page_idx
        individual = self.flipbook_pages[page_idx].annotations['individual']
        # Update the active individual's annotation
        # This annotator handles individuals separately but on the same image.
        self.flipbook_pages[page_idx].annotations['poses'][individual] = \
            self.flipbook_pages[page_idx].annotations['pose']
    
    def remove_red_pose_overlays(self):
        for outline in self.current_outlines:
            if outline is None:
                continue
            outline.center_spline.setPen(Qt.QPen(Qt.Qt.transparent))
            outline.setVisible(False)
            # outline.parentItem().setVisible(False)
            # outline.parentObject().setVisible(False)

    def refresh_onscreen_poses(self):
        ''' Updates red, inactive pose annotations on-screen.
        '''
        self.remove_red_pose_overlays()
        page_idx = self.flipbook.current_page_idx
        individual = self.flipbook_pages[page_idx].annotations['individual']
        poses = self.flipbook_pages[page_idx].annotations['poses']
        red_pen = Qt.QPen(Qt.QColor(255, 0, 255, 255))
        red_pen.setStyle(Qt.Qt.SolidLine)
        red_pen.setWidth(2)
        for i, (center_tck, width_tck) in enumerate(poses):
            if i == individual:
                continue
            if self.current_outlines[i] is None:
                self.current_outlines[i] = spline_outline.SplineOutline(self, Qt.QColor(255, 0, 0, 255))
            outline = self.current_outlines[i]
            outline.center_spline.setPen(red_pen)
            outline.set_locked(False)
            outline.center_spline.geometry = center_tck
            outline.width_spline.geometry = width_tck
            outline.update_outline()
            outline.setVisible(True)
            outline.set_locked(True)
        if len(self.flipbook_pages[page_idx]) > 0:
            self.flipbook_pages[page_idx][0].refresh()
        self.annotator.update_fields()  # This must be called to update the GUI

TESTING = config.getboolean('TEST', 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    if not TESTING:
        parser.add_argument(
            'annotation_root', help='Location of images to annotate.')
    parser.add_argument(
        '-s', '--scorer', help='Name of person annotating the poses. Important: must be the same as the DeepLabCut scorer.')
    args = parser.parse_args()
    if TESTING:
        __annotation_root = '~/code/swimfunction/multianimal/train_images/variety/small'
    else:
        __annotation_root = args.annotation_root
    pa = PoseAnnotator(
        pathlib.Path(__annotation_root).expanduser().resolve(),
        args.scorer)
    pa.run()
