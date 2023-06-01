'''
Extracts specific frames from a fish's swim.
'''
from typing import List
import numpy

from swimfunction import FileLocations
from swimfunction.data_access.data_utils import parse_details_from_filename
from swimfunction.global_config.config import config
from swimfunction.video_processing import fp_ffmpeg

UNINJURED_WEEK = config.getint('EXPERIMENT DETAILS', 'control_assay_label')

class Extractor:
    ''' Extracts frames from videos in a smart way.
    Compatible with CropTracker
    '''
    # Local import to avoid circular references.
    from swimfunction.video_processing import CropTracker, SplitCropTracker
    __slots__ = [
        'video_fname',
        'side',
        'ct_logger',
        'frame_nums_are_full_video',
        'vheight',
        'vwidth'
    ]
    def __init__(self, fish_name, assay_label, frame_nums_are_full_video=True, videos_root=None):
        '''
        Parameters
        ----------
        fish_name : str
            Examples: M13 or F23
        assay_label : int
            Week post injury, -1 is uninjured.
        frame_nums_are_full_video : bool, default=True
            Whether the frame numbers are relative to the original video.
            If True, will use CropTracker conversion if available.
            If False, then will NOT convert.
            (If you never used CropTracker, then no problem either way).
        videos_root : optional
            Location of video files.
        '''
        self.video_fname = FileLocations.find_video(fish_name, assay_label, videos_root=videos_root)
        fish_details = parse_details_from_filename(self.video_fname)
        needs_crop = len(fish_details) == 2
        self.side = None
        if needs_crop:
            self.side = fish_details[0].side
            if fish_details[1].name == fish_name:
                self.side = fish_details[1].side
        self.frame_nums_are_full_video = frame_nums_are_full_video
        self.vheight, self.vwidth = (
            config.getint('VIDEO', 'default_video_height'),
            config.getint('VIDEO', 'default_video_width')
        )
        self.ct_logger = None
        logfile = Extractor.CropTracker.find_existing_logfile(self.video_fname)
        if logfile:
            self.ct_logger = Extractor.CropTracker.CropTrackerLog().read_from_file(logfile)
            ct = Extractor.CropTracker.CropTracker.from_logfile(logfile)
            self.vheight, self.vwidth = ct.full_height, ct.full_width

    @property
    def nframes(self):
        ''' Total number of frames in the original video.
        '''
        if self.ct_logger is not None:
            return self.ct_logger.total_num_frames
        return fp_ffmpeg.get_nframes(self.video_fname)

    @property
    def fish_name(self):
        ''' Fish name
        '''
        return parse_details_from_filename(self.video_fname)[0].name

    @property
    def assay(self):
        ''' Assay label
        '''
        return parse_details_from_filename(self.video_fname)[0].assay_label

    def _frame_as_full_tunnel_img(self, frame_img, frame_num):
        if self.ct_logger is None:
            return frame_img
        img = numpy.full((self.vheight, self.vwidth, 3), 255, dtype=numpy.uint8)
        corner = self.ct_logger.get_corner(frame_num)
        ylims = (corner.y, min(corner.y + frame_img.shape[0], img.shape[0]))
        xlims = (corner.x, min(corner.x + frame_img.shape[1], img.shape[1]))
        img[ylims[0]:ylims[1], xlims[0]:xlims[1], ...] = \
            frame_img[:ylims[1]-ylims[0], :xlims[1]-xlims[0], ...]
        # Draw border
        img[0, :, :] = 0
        img[-1, :, :] = 0
        img[:, 0, :] = 0
        img[:, -1, :] = 0
        return img

    def _ensure_side_is_selected(self, frame):
        '''
        '''
        if self.side == 'R':
            frame = Extractor.SplitCropTracker.get_right_left_crops(frame).right
        elif self.side == 'L':
            frame = Extractor.SplitCropTracker.get_right_left_crops(frame).left
        return frame

    def _get_final_frame_number(self, requested_frame):
        frame_num = requested_frame
        if self.ct_logger is not None and self.frame_nums_are_full_video:
            frame_num = self.ct_logger.original_frame_num_to_crop_track_frame_num(requested_frame)
        return frame_num

    def extract_frame(
            self,
            frame_num,
            as_full_tunnel_img_if_necessary=False,
            **kwargs):
        ''' Get a single frame. Corrects with CropTrackerLog
        if available and frame_nums_are_full_video.
        Parameters
        ----------
        frame_num : int
        as_full_tunnel_img_if_necessary : bool
        **kwargs
            Passed into fp_ffmpeg.read_frame

        Returns
        -------
        frame : numpy.ndarray, None
        '''
        frame_num = self._get_final_frame_number(frame_num)
        frame = fp_ffmpeg.read_frame(self.video_fname, frame_num=frame_num, **kwargs)
        if as_full_tunnel_img_if_necessary:
            frame = self._frame_as_full_tunnel_img(frame, frame_num)
        self._ensure_side_is_selected(frame)
        return frame

    def extract_frames(
            self,
            frame_nums,
            as_full_tunnel_img_if_necessary=False,
            **kwargs) -> List[numpy.ndarray]:
        '''
        Parameters
        ----------
        frame_nums : list
            Which frames to extract (e.g. frames [245, 246, 247])
        as_full_tunnel_img_if_necessary : bool
        **kwargs
            Passed into fp_ffmpeg.read_frame

        Returns
        -------
        frames
            list of numpy.ndarray or None if frame number does not exist as a valid frame.
        '''
        frame_nums = list(map(self._get_final_frame_number, frame_nums))
        frame_nums = numpy.asarray(frame_nums)
        frames = []
        # Handle consecutive frames.
        if None not in frame_nums \
                and numpy.all(numpy.isclose(frame_nums[1:] - frame_nums[:-1], 1)):
            frames = fp_ffmpeg.generate_consecutive_frames(
                self.video_fname, len(frame_nums), frame_nums[0], **kwargs)
        else:
            frames = [
                fp_ffmpeg.read_frame(
                    self.video_fname, frame_num=num, **kwargs)
                for num in frame_nums]
        if as_full_tunnel_img_if_necessary:
            full_tunnel_frames = []
            for frame, frame_num in zip(frames, frame_nums):
                full_tunnel_frames.append(self._frame_as_full_tunnel_img(
                    frame,
                    frame_num))
            frames = full_tunnel_frames
        return list(map(self._ensure_side_is_selected, frames))

def extract_frames(
        fish_name,
        assay_label,
        frame_nums,
        frame_nums_are_full_video=True,
        videos_root=None,
        **kwargs) -> list:
    '''
    Parameters
    ----------
    fish_name : str
        Examples: M13 or F23
    assay_label : int
        Week post injury, -1 is uninjured.
    frame_nums : list
        Which frames to extract (e.g. frames [245, 246, 247])
    frame_nums_are_full_video : bool, default=True
        Whether the frame numbers are relative to the original video.
        If True, will use CropTracker conversion if available.
        If False, then will NOT convert.
        (If you never used CropTracker, then no problem either way).
    videos_root : optional
        Location of video files.
    **kwargs
        Passed into extractor.extract_frames
        then to fp_ffmpeg.read_frame

    Returns
    -------
    frames : numpy.array or None
    '''
    extractor = Extractor(
        fish_name,
        assay_label,
        frame_nums_are_full_video,
        videos_root=videos_root)
    return extractor.extract_frames(frame_nums, **kwargs)

def extract_frame(
        fish_name,
        assay_label,
        frame_num,
        frame_nums_are_full_video=True,
        videos_root=None,
        **kwargs) -> list:
    '''
    Parameters
    ----------
    fish_name : str
        Examples: M13 or F23
    assay_label : int
        Week post injury, -1 is uninjured.
    frame_nums : list
        Which frames to extract (e.g. frames [245, 246, 247])
    frame_nums_are_full_video : bool, default=True
        Whether the frame numbers are relative to the original video.
        If True, will use CropTracker conversion if available.
        If False, then will NOT convert.
        (If you never used CropTracker, then no problem either way).
    videos_root : optional
        Location of video files.
    **kwargs
        Passed into fp_ffmpeg.read_frame

    Returns
    -------
    frames : numpy.array or None
    '''
    extractor = Extractor(
        fish_name,
        assay_label,
        frame_nums_are_full_video,
        videos_root=videos_root)
    return extractor.extract_frame(frame_num, **kwargs)
