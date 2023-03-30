''' Frame by frame crops to the most likely location of the fish.

IMPORTANT:
Images are assumed to be in [width, height] indexing, despite convention.
This is to retain ffmpeg and freeimage compatibility.
'''
from collections import namedtuple

import gzip
import pathlib
import numpy
import tqdm

from swimfunction.video_processing import get_median_frames, fp_ffmpeg, standardize_video
from swimfunction import loggers

Point = namedtuple('Point', ['x', 'y'])
CropResult = namedtuple('CropResult', ['img', 'corner'])
WritersLoggers = namedtuple('WritersLoggers', ['video_writers', 'loggers'])

def logpath_to_videopath(logfpath: pathlib.Path) -> pathlib.Path:
    ''' Get the location of the video that matches the logfile.
    They are assumed to be in the same folder as one another, just with different extensions.
    '''
    logfpath = pathlib.Path(logfpath)
    valid_extensions = ['.avi', '.mp4']
    p = None
    for extension in valid_extensions:
        p = logfpath.parent / logfpath.name.replace('.ctlog', extension).replace('.gz', '')
        if p.exists():
            break
    return p

def videopath_to_gz_logfile(videopath: pathlib.Path):
    ''' DO NOT CALL THIS TO SEARCH FOR AN EXISTING LOGFILE

    Gets name for writing a new logfile.

    CropTracker log files can either be plaintext with the extension .ctlog
    or gzipped with extension .ctlog.gz
    By default, we write gzipped files.
    '''
    return pathlib.Path(videopath).with_suffix('.ctlog.gz')

def find_existing_logfile(video_path: pathlib.Path, rename_outdated_logfiles: bool=False):
    ''' Finds a CropTracker log file in the same location as a video file.
    If it does not exist, it returns None.
    '''
    possible_log = None
    if video_path is None:
        return possible_log
    video_path = pathlib.Path(video_path)
    if video_path.with_suffix('.ctlog.gz').exists():
        possible_log = video_path.with_suffix('.ctlog.gz')
    elif video_path.with_suffix('.ctlog').exists():
        possible_log = video_path.with_suffix('.ctlog')
    if possible_log is None and rename_outdated_logfiles:
        if video_path.with_suffix('.log.gz').exists():
            to_rename = video_path.with_suffix('.log.gz')
            possible_log = video_path.with_suffix('.ctlog.gz')
            to_rename.rename(possible_log)
        elif video_path.with_suffix('.log').exists():
            to_rename = video_path.with_suffix('.log')
            possible_log = video_path.with_suffix('.ctlog')
            to_rename.rename(possible_log)
    return possible_log

def to_fish_frame(lab_coords, corner: Point, is_transposed: bool):
    ''' Goes from coordinates in full space to cropped space.
    '''
    change = list(reversed(corner))
    if not is_transposed:
        change = list(corner)
    return lab_coords - change

def to_lab_frame(fish_coords, corner: Point):
    ''' Goes from coordinates in cropped space to full space.
    '''
    return fish_coords + numpy.asarray([corner.x, corner.y])

class CropTracker:
    ''' Base class for all trackers that crop to the area of the fish.
    '''
    quiet = False

    __slots__ = [
        'crop_width', 'crop_height',
        'full_width', 'full_height',
        'subsample_factor', 'debug']

    def __init__(
            self, crop_width:int, crop_height:int,
            full_width:int, full_height:int,
            subsample_factor:int, debug: bool = False,
            *args, **kwargs):
        '''
        Parameters
        ----------
        crop_width: int
        crop_height: int
        full_width: int
            image width
        full_height: int
            image height
        subsample_factor: int
            stride for subsampling the image
            (larger number speeds up, but could risk losing small fish)
        '''
        self.crop_width = int(crop_width)
        self.crop_height = int(crop_height)
        self.full_width = int(full_width)
        self.full_height = int(full_height)
        self.subsample_factor = int(subsample_factor)
        self.debug = bool(debug)

    def __str__(self):
        return '\n'.join([
            f'crop_width={self.crop_width}',
            f'crop_height={self.crop_height}',
            f'full_width={self.full_width}',
            f'full_height={self.full_height}',
            f'subsample_factor={self.subsample_factor}'
        ])

    def crop_track(self, vfile, output_dir, medians_loc, max_frames: int):
        ''' Detects and tracks fish, cropping to fit them on the screen,
        and saves each fish to its own video.

        Parameters
        ----------
        vfile
        output_dir
        medians_loc
        max_frames: int
        '''
        if vfile is None:
            return
        vfile = pathlib.Path(vfile)
        if not vfile.exists():
            return
        tracker_type_file = pathlib.Path(output_dir) / 'tracker_type.txt'
        if not tracker_type_file.exists():
            with open(tracker_type_file, 'wt') as ofh:
                ofh.write(type(self).__name__)
        v_median = get_median_frames.get_video_median(vfile, medians_loc)
        standardizers = self.get_standardizers(v_median)
        nframes = fp_ffmpeg.get_nframes(vfile)
        iteration_tool = enumerate(fp_ffmpeg.read_video(vfile))
        video_writers, ct_loggers = self.get_video_writers_and_loggers(
            vfile, output_dir, nframes, standardizers)
        should_crop_track = numpy.any([
            True for l in ct_loggers \
                if l is not None \
                    and l.crop_tracked_vfname is not None \
                        and not pathlib.Path(l.crop_tracked_vfname).exists()])
        if not should_crop_track:
            loggers.get_video_processing_logger(__name__).info(f'Cannot crop track {vfile}')
            return
        if not video_writers or numpy.all([v is None for v in video_writers]):
            return
        if not self.quiet:
            iteration_tool = tqdm.tqdm(iteration_tool, total=min(nframes, max_frames), mininterval=1)
        for frame_i, frame in iteration_tool:
            if frame_i >= max_frames:
                break
            self.handle_frame(frame_i, frame, video_writers, ct_loggers, standardizers)
        for writer in video_writers:
            if writer is not None:
                writer.close()
        for ct_logger in ct_loggers:
            if ct_logger is not None:
                ct_logger.write_log()
        loggers.get_video_processing_logger(__name__).info(f'Finished CropTracking {vfile}')

    ##### Static Methods

    @staticmethod
    def from_logfile(logfname, class_type=None):
        ''' Constructs a CropTracker (or child class) from a log file.
        '''
        if class_type is None:
            class_type = CropTracker
        attrs = {'debug': False}
        reader = None
        logfname = pathlib.Path(logfname)
        if logfname.suffix == '.gz':
            reader = gzip.open(logfname, 'rt')
        else:
            reader = open(logfname, 'rt')
        for line in reader:
            if '=' in line:
                dim_str = line.strip().split('=')
                if dim_str[0] in class_type.__slots__:
                    attrs[dim_str[0]] = dim_str[1]
        reader.close()
        return class_type(**attrs)

    @staticmethod
    def full_video_coords_to_lab_frame(coords, logfname):
        ''' Transforms coordinates from crop-tracked space to video space.
        '''
        logfname = pathlib.Path(logfname)
        log = CropTrackerLog().read_from_file(logfname)
        video_fname = logpath_to_videopath(logfname)
        nframes_true = fp_ffmpeg.get_nframes(video_fname)
        assert nframes_true == len(log.frames), \
            f'{nframes_true} frames in video != {len(log.frames)} corners in log. {logfname.name}'
        assert len(log.frames) == coords.shape[0], \
            f'{nframes_true} frames != {coords.shape[0]} poses. {logfname.name}'
        total_num_frames = log.total_num_frames
        final_coords = numpy.full((total_num_frames, *coords.shape[1:]), numpy.nan, dtype=float)
        for i in range(coords.shape[0]):
            frame_num = log.frames[i]
            final_coords[frame_num, ...] = to_lab_frame(coords[i, ...], log.corners[i])
        return final_coords

    @staticmethod
    def subsample_image(img: numpy.ndarray, scale_factor):
        ''' This is a fast, rough downscaling with no antialiasing.
        '''
        return img[::scale_factor, ::scale_factor]

    @staticmethod
    def upsample_image(img: numpy.ndarray, scale_factor, final_shape):
        ''' This is a rough upscaling. Only use it for binary masks.
        I used a line from ZisIsNotZis solution located at the link below:
        https://stackoverflow.com/questions/53330908/python-quick-upscaling-of-array-with-numpy-no-image-libary-allowed
        '''
        return numpy.broadcast_to(
            img[:, None, :, None], (img.shape[0], scale_factor, img.shape[1], scale_factor)
        ).reshape((
            img.shape[0]*scale_factor,
            img.shape[1]*scale_factor
        ))[:final_shape[0], :final_shape[1]]

    @staticmethod
    def rescale_point(point: Point, scale_factor):
        ''' Converts a point from the subsampled location to the full-scale point.
        '''
        return Point(int(point.x * scale_factor), int(point.y * scale_factor))

    @staticmethod
    def safe_corner(val, max_val):
        ''' Uses val if possible, but forces between 0 and max_val.
        '''
        return min(max(0, val), max(0, max_val))

    @staticmethod
    def center_to_corner(center, crop_width, crop_height, img_shape):
        ''' Gets the top-left corner coordinate for a crop with the given center.
        '''
        return Point(
            x=CropTracker.safe_corner(center[0] - crop_width // 2, img_shape[0] - crop_width),
            y=CropTracker.safe_corner(center[1] - crop_height // 2, img_shape[1] - crop_height)
        )

    ##### Abstract Methods

    def get_video_writers_and_loggers(
            self,
            vfile,
            output_dir,
            nframes,
            standardizers=None) -> WritersLoggers:
        '''
        Returns
        -------
        writers_loggers : WritersLoggers
            namedtuple that contains two lists (or tuples, or namedtuples).
        '''
        raise NotImplementedError('Abstract method')

    def get_standardizers(self, v_median):
        '''
        Returns
        -------
        standardizers
        '''
        raise NotImplementedError('Abstract method.')

    def handle_frame(
            self,
            frame_i,
            frame,
            video_writers,
            ct_loggers,
            standardizers):
        ''' Crop to each fish in a single frame.
        Returns
        -------
        None
        '''
        raise NotImplementedError('Abstract method.')

    ##### Helper Methods

    def apply_crop(self, img, corner: Point, force_dimensions: bool):
        ''' Crops an image to size at the corner.
        '''
        crop_area = standardize_video.CropArea(
            corner.x, corner.y,
            self.crop_width, self.crop_height)
        cropped = standardize_video.crop_img(img, crop_area)
        if force_dimensions:
            cropped = standardize_video.buffer_to_size(cropped, crop_area, fill_high=True)
        return cropped

    def corner_to_crop_area(self, corner: Point):
        ''' Converts a point into a CropArea namedtuple.
        '''
        return standardize_video.CropArea(
            int(corner.x), int(corner.y),
            self.crop_width, self.crop_height)

class CropTrackerLog:
    ''' Writes essential meta data.
    CropTracker only saves frames that likely contain a fish,
    so the log must link the crop-track video's frame number to
    the original video's frame number.
    '''
    __slots__ = [
        'original_video_path', 'total_num_frames',
        'crop_tracked_vfname', 'crop_tracker', 'frames', 'corners',
        'original_to_ct_frame_dict']

    def __init__(self):
        self.original_video_path = None
        self.total_num_frames = None
        self.crop_tracked_vfname = None
        self.crop_tracker = None
        self.frames = []
        self.corners = []
        self.original_to_ct_frame_dict = {}

    def __str__(self):
        lines = [
            f'original_video={self.original_video_path}\n',
            f'nframes={self.total_num_frames}\n',
            self.crop_tracker.__str__(),
            '\nframe_num,x,y\n'
        ]
        for frame_i, corner in zip(self.frames, self.corners):
            lines.append(self.corner_to_line(frame_i, corner))
        return ''.join(lines)

    def setup_ct_logger(
            self, original_video_path, crop_tracked_vfname,
            crop_tracker, total_num_frames):
        ''' Constructs a blank logger.

        Parameters
        ----------
        original_video_path : str, pathlib.Path
            Name of original video file (the one that will be crop-tracked.)
        crop_tracked_vfname : str, pathlib.Path
            Name of crop tracker output video file
        crop_tracker : CropTracker, MultiCropTracker, SplitCropTracker
        total_num_frames : int

        Returns
        -------
        self
            For convenient function chaining
        '''
        self.original_video_path = pathlib.Path(
            original_video_path
        ).expanduser().resolve().as_posix()
        self.total_num_frames = total_num_frames
        self.crop_tracked_vfname = pathlib.Path(crop_tracked_vfname).expanduser().resolve()
        self.crop_tracker = crop_tracker
        self.frames = []
        self.corners = []
        return self

    def read_from_file(self, logfilename, crop_tracker=None):
        ''' Constructs a logger from the logfile.

        Returns
        -------
        self
        '''
        if logfilename is None:
            return None
        self.crop_tracker = crop_tracker
        logpath = pathlib.Path(logfilename)
        reader = None
        self.crop_tracked_vfname = logpath_to_videopath(logpath)
        if logpath.suffix == '.gz':
            reader = gzip.open(logpath, 'rt')
        else:
            reader = open(logpath, 'rt')
        for line in reader:
            line = line.strip()
            if not line:
                continue
            # Parse attributes
            if '=' in line:
                opts = line.split('=')
                if opts[0] == 'original_video':
                    self.original_video_path = opts[1]
                elif opts[0] == 'nframes':
                    self.total_num_frames = int(opts[1])
            # Parse corners
            elif ',' in line:
                line = line.split(',')
                if str.isnumeric(line[0]):
                    self.log_corner(int(line[0]), Point(int(line[1]), int(line[2])))
        reader.close()
        return self

    def log_corner(self, frame_i, corner: Point):
        ''' Keep track of a frame number and its corner.
        '''
        self.frames.append(frame_i)
        self.original_to_ct_frame_dict[frame_i] = len(self.frames) - 1
        self.corners.append(corner)

    def write_log(self):
        ''' Writes the log to a .gz file.
        '''
        logfpath = videopath_to_gz_logfile(self.crop_tracked_vfname)
        with gzip.open(logfpath, 'wt') as ofh:
            ofh.write(self.__str__())

    def get_corner(self, original_frame_num: int) -> Point:
        ''' Given frame num in unprocessed video, get the crop tracked corner.
        '''
        ct_frame_num = self.original_frame_num_to_crop_track_frame_num(original_frame_num)
        corner = Point(0, 0)
        if ct_frame_num is not None:
            corner = self.corners[ct_frame_num]
        return corner

    def original_frame_num_to_crop_track_frame_num(self, frame_num: int) -> int:
        ''' Convert a frame number from original video to CropTracker video.
        For example, frame 10 in the original video may be 3 in the CropTracker video
        if the fish was not visible for 7 frames.

        Parameters
        ----------
        frame_num : int
            frame number in the original video

        Returns
        -------
        crop_tracked_video_num
            frame_num if CropTrackerLog has not been initialized.
            None if frame_num not in the crop-tracked video.
            crop-tracked video's frame_num otherwise.
        '''
        if self.crop_tracked_vfname is None:
            return frame_num
        ct_frame_num = None
        if frame_num in self.original_to_ct_frame_dict:
            ct_frame_num = self.original_to_ct_frame_dict[frame_num]
        return ct_frame_num

    def generate_frame_sequences(
            self,
            sequence_length: int,
            require_all_exist: bool,
            use_crop_tracked_video_frame_nums: bool):
        ''' Get sequential frame numbers (in original video).
        Note: use_crop_tracked_video_frame_nums = True
        to get the crop-tracked video frame numbers.
        '''
        for i in range(0, self.total_num_frames-sequence_length+1, sequence_length):
            frame_nums = numpy.arange(i, i+sequence_length)
            should_return = True
            if require_all_exist:
                should_return = numpy.all([
                    frame_num in self.original_to_ct_frame_dict
                    for frame_num in frame_nums])
            if should_return:
                if use_crop_tracked_video_frame_nums:
                    frame_nums = numpy.asarray([
                        self.original_to_ct_frame_dict[n]
                        for n in frame_nums])
                yield frame_nums

    @staticmethod
    def corner_to_line(frame_i, corner: Point):
        ''' Converts a frame number and corner to a log string.
        '''
        return f'{frame_i},{corner.x},{corner.y}\n'
