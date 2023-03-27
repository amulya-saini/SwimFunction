''' Frame by frame crops to the most likely location of two fish, separated by a divider.

Developer notes (most users can ignore this):
Images are assumed to be in [width, height] indexing, despite convention.
This is to retain ffmpeg and freeimage compatibility.
'''
from collections import namedtuple
from typing import List
import traceback
import pathlib
from scipy.ndimage import center_of_mass
import numpy

from swimfunction.context_managers.FileLock import FileLock
from swimfunction.global_config.config import config
from swimfunction.video_processing.image_processing import threshold, isolate_object_in_mask
from swimfunction.video_processing import standardize_video, fp_ffmpeg
from swimfunction.video_processing.CropTracker import CropTracker, CropResult, \
    Point, CropTrackerLog, WritersLoggers
from swimfunction import loggers
from swimfunction import FileLocations

MAX_FRAMES = config.getint('VIDEO', 'fps') * 60 * 15 # 15 min swims

Dims = namedtuple('Dims', ['width', 'height'])

SCALE_DIVISOR = config.getint('VIDEO', 'scale_divisor')
VIDEO_FULL_SIZE_DIMENSIONS = Dims(
    width=config.getint('VIDEO', 'crop_width'),
    height=config.getint('VIDEO', 'crop_height'))
VIDEO_RESIZED_DIMENSIONS = Dims(
    width=VIDEO_FULL_SIZE_DIMENSIONS.width//SCALE_DIVISOR,
    height=VIDEO_FULL_SIZE_DIMENSIONS.height//SCALE_DIVISOR)

CROP_WIDTH = config.getint('CROP_TRACKER', 'crop_width')
CROP_HEIGHT = config.getint('CROP_TRACKER', 'crop_height')
SUBSAMPLE_FACTOR = config.getint('CROP_TRACKER', 'subsample_factor')

MAX_FRAMES = config.getint('VIDEO', 'fps') * 60 * 15 # 15 min swims

LeftRight = namedtuple('LeftRight', ['left', 'right'])
Standardizers = namedtuple('Standardizers', ['left', 'right', 'subsampled'])

def get_sides(img):
    ''' Because freeimage and ffmpeg produce images that are rotated and mirrored,
    the order is a bit backwards here. The left side is in fact the second half of the image,
    and the right side is the first half of the image. Yes, you read that correctly.
    '''
    return LeftRight(
        img[:img.shape[0] // 2, :],
        img[img.shape[0] // 2:, :]
    )

def get_right_left_paths(input_path, output_dir) -> LeftRight:
    ''' Assumes file names in this format:
            .+_[^_]+L_[^_]+R\\..+

    Returns
    -------
    paths_result: namedtuple
        right: pathlib.Path
        left: pathlib.Path
    '''
    input_path = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_dir)
    name = input_path.name
    parts = name.split('.')
    assert len(parts) == 2, \
        f'{name} : Trouble parsing filename since it does not have exactly one period.'
    base_parts = parts[0].split('_')
    right_bit = ''
    left_bit = ''
    if base_parts[-1][-1] == 'R':
        right_bit = base_parts[-1]
    elif base_parts[-1][-1] == 'L':
        left_bit = base_parts[-1]
    if base_parts[-2][-1] == 'R':
        right_bit = base_parts[-2]
    elif base_parts[-2][-1] == 'L':
        left_bit = base_parts[-2]
    if not (right_bit or left_bit):
        raise RuntimeError(' '.join([
            name,
            ': Could not find the right and left',
            'components in the file name.']))
    right, left = None, None
    if right_bit:
        right = output_dir / (name.replace(''.join(('_', left_bit)), '') if left_bit else name)
    if left_bit:
        left = output_dir / (name.replace(''.join(('_', right_bit)), '') if right_bit else name)
    return LeftRight(left=left, right=right)

def get_right_left_crops(frame: numpy.ndarray) -> LeftRight:
    ''' DEPRECATED: Pull two crops from frame (right and left)
    each with dimensions equal to the closer of VIDEO_FULL_SIZE_DIMENSIONS
    or VIDEO_RESIZED_DIMENSIONS.
    '''
    if frame is None:
        return LeftRight(None, None)
    dims = VIDEO_FULL_SIZE_DIMENSIONS
    closeness_to_resized = abs(VIDEO_RESIZED_DIMENSIONS.height - frame.shape[1])
    closeness_to_full_sized = abs(VIDEO_FULL_SIZE_DIMENSIONS.height - frame.shape[1])
    if closeness_to_resized < closeness_to_full_sized:
        dims = VIDEO_RESIZED_DIMENSIONS
    width = dims.width
    height = dims.height
    return LeftRight(left=frame[:width, :height], right=frame[-width:, :height])

class SplitCropTracker(CropTracker):
    ''' Splits a video into right and left sides.
    Tracks each side's fish and crops to the area of the fish.
    '''

    @staticmethod
    def from_logfile(logfname, class_type=None):
        return super().from_logfile(logfname, class_type=SplitCropTracker)

    ##### Implement Abstract Methods

    def get_video_writers_and_loggers(
            self,
            vfile,
            output_dir,
            nframes,
            standardizers=None) -> WritersLoggers:
        output_dir = pathlib.Path(output_dir)
        framerate = fp_ffmpeg.VideoData(vfile).fps
        vfile = pathlib.Path(vfile).expanduser().resolve()
        out_vfnames = get_right_left_paths(vfile, output_dir)
        make_right = out_vfnames.right is not None and not out_vfnames.right.exists()
        make_left = out_vfnames.left is not None and not out_vfnames.left.exists()
        video_writers = LeftRight(
            left=fp_ffmpeg.VideoWriter(
                framerate, out_vfnames.left, preset=None, lossless=False, verbose=False
            ) if make_left else None,
            right=fp_ffmpeg.VideoWriter(
                framerate, out_vfnames.right, preset=None, lossless=False, verbose=False
            ) if make_right else None
        )
        ct_loggers = LeftRight(
            left=CropTrackerLog().setup_ct_logger(
                vfile,
                out_vfnames.left,
                self,
                nframes) if make_left else None,
            right=CropTrackerLog().setup_ct_logger(
                vfile,
                out_vfnames.right,
                self,
                nframes) if make_right else None
        )
        return WritersLoggers(video_writers, ct_loggers)

    def get_standardizers(self, v_median):
        has_divider = True
        standardizer = standardize_video.Standardizer(v_median, has_divider)
        subsample_inner_content = standardize_video.StandardizerInnerContent(
            self.subsample_image(standardizer.target_median, self.subsample_factor),
            standardizer.normalizing_lut,
            self.subsample_image(standardizer.v_median_normalized, self.subsample_factor)
        )
        subsample_standardizer = standardize_video.Standardizer(
            self.subsample_image(v_median, self.subsample_factor),
            has_divider,
            subsample_inner_content)
        median_sides = get_sides(v_median)
        median_normalized_sides = get_sides(standardizer.v_median_normalized)
        left_inner_content = standardize_video.StandardizerInnerContent(
            median_sides.left,
            standardizer.normalizing_lut,
            median_normalized_sides.left
        )
        right_inner_content = standardize_video.StandardizerInnerContent(
            median_sides.right,
            standardizer.normalizing_lut,
            median_normalized_sides.right
        )
        left_standardizer = standardize_video.Standardizer(
            v_median, has_divider, left_inner_content)
        right_standardizer = standardize_video.Standardizer(
            v_median, has_divider, right_inner_content)
        return Standardizers(left_standardizer, right_standardizer, subsample_standardizer)

    def handle_frame(
            self, frame_i,
            frame,
            video_writers: LeftRight,
            ct_loggers: LeftRight,
            standardizers: Standardizers):
        ''' Crop to the fish in the frame,
        encode the frame, store its corner in the log file.
        '''
        crops = self.get_divided_fish_crops(
            frame,
            standardizers,
            do_track_LeftRight=LeftRight(
                left=video_writers.left is not None,
                right=video_writers.right is not None))
        if video_writers.right is not None and crops.right.corner is not None:
            video_writers.right.encode_frame(crops.right.img)
            ct_loggers.right.log_corner(frame_i, crops.right.corner)
        if video_writers.left is not None and crops.left.corner is not None:
            video_writers.left.encode_frame(crops.left.img)
            ct_loggers.left.log_corner(frame_i, crops.left.corner)

    ##### Helper Methods

    def get_divided_fish_crops(
            self,
            frame,
            standardizers: Standardizers,
            do_track_LeftRight: LeftRight) -> LeftRight:
        ''' Divides a frame into two sides, cropped to center each fish.
        '''
        ssframe = standardizers.subsampled.get_standardized_frame(
            self.subsample_image(frame, self.subsample_factor)
        )
        frame_sides = get_sides(frame)
        ssframe_sides = get_sides(ssframe)

        right_frame = None
        right_corner = None
        if do_track_LeftRight.right:
            right_corner = self.get_corner_of_fish_crop(ssframe_sides.right, is_subsampled=True)
            if right_corner is not None:
                right_corner = self.rescale_point(right_corner, self.subsample_factor)
                right_frame = standardizers.right.get_standardized_frame(
                    frame_sides.right, self.corner_to_crop_area(right_corner))

        left_frame = None
        left_corner = None
        if do_track_LeftRight.left:
            left_corner = self.get_corner_of_fish_crop(ssframe_sides.left, is_subsampled=True)
            if left_corner is not None:
                left_corner = self.rescale_point(left_corner, self.subsample_factor)
                left_frame = standardizers.left.get_standardized_frame(
                    frame_sides.left, self.corner_to_crop_area(left_corner))

        return LeftRight(
            CropResult(left_frame, left_corner),
            CropResult(right_frame, self.right_corner_to_global(right_corner)))

    def get_corner_of_fish_crop(self, img: numpy.ndarray, is_subsampled: bool):
        ''' Crops to the fish.
        Parameters
        ----------
        img: numpy.ndarray
            Image, assumed to be normalized and background-subtracted.
        is_subsampled: bool
            Whether the image is already subsampled.
            If so, the crop_width and crop_height are reduced accordingly.
        '''
        assert len(img.shape) == 2
        width = self.crop_width
        height = self.crop_height
        if is_subsampled:
            width = self.crop_width // self.subsample_factor
            height = self.crop_height // self.subsample_factor
        mask = threshold(img)
        if mask is None:
            return None
        mask = isolate_object_in_mask(mask)
        if mask is None:
            return None
        center = [int(x) for x in center_of_mass(mask)]
        return self.center_to_corner(center, width, height, img.shape)

    def right_corner_to_global(self, corner) -> Point:
        ''' Converts a point relative to the right side
        of the video to the global point space.
        '''
        if corner is None:
            return corner
        return Point(corner.x + self.full_width // 2, corner.y)

def split_crop_track_video(vfile, medians_loc, output_dir, max_frames: int=MAX_FRAMES) -> LeftRight:
    ''' Gets crop track videos for the left and right sides of the video.
    If crop-tracked video already exists, it will return None.

    Parameters
    ----------
    vfile
    medians_loc
    outdir
    max_frames: int
        Maximum allowed frames.
        Normally, 63000 is maximum (15 min * 60 sec/min * 70 frames/sec)

    Returns
    -------
    pathlib.Path
        Path to files that were written by this function.
        Note: files that were already written will not be returned.
    '''
    vfile = pathlib.Path(vfile)
    if not vfile.exists():
        return LeftRight(None, None)
    try:
        # Check whether right and left both exist in the filename.
        rl = get_right_left_paths(vfile, output_dir)
        if rl.left is None and rl.right is None:
            return rl
    except RuntimeError as e:
        logger = loggers.get_video_processing_logger(__name__)
        logger.error(vfile.as_posix())
        logger.error(traceback.format_exc())
        logger.error('Make sure file name is formatted correctly for SplitCropTracker\n%s', str(e))
        return None
    with FileLock(
            lock_dir=output_dir,
            lock_name=vfile.name) as lock_acquired:
        if lock_acquired:
            frame = fp_ffmpeg.read_frame(vfile, frame_num=0)
            full_width, full_height = frame.shape[:2]
            tracker = SplitCropTracker(
                CROP_WIDTH, CROP_HEIGHT,
                full_width, full_height, SUBSAMPLE_FACTOR)
            tracker.crop_track(vfile, output_dir, medians_loc, max_frames)
    return rl

def split_crop_track(
        video_root=None,
        outdir=None,
        max_frames: int=MAX_FRAMES,
        logger=None) -> List[pathlib.Path]:
    ''' Gets crop track videos for the left and right sides of the video.
    If crop-tracked video already exists, it will be skipped.

    Parameters
    ----------
    video_root
    outdir
    max_frames: int
        Maximum allowed frames.
        Normally, 63000 is maximum (15 min * 60 sec/min * 70 frames/sec)

    Returns
    -------
    List[pathlib.Path]
        Path to files that were written by this function.
        Note: files that were already written will not be yielded.
    '''
    if logger is None:
        logger = loggers.get_vanilla_logger(
            __name__,
            pathlib.Path('./crop_track.log').expanduser().resolve(),
            False)
    videos_root = FileLocations.get_original_videos_dir() \
        if video_root is None else pathlib.Path(video_root)
    output_dir = FileLocations.get_normalized_videos_dir()\
        if outdir is None else pathlib.Path(outdir)
    videos = list(FileLocations.find_video_files(videos_root, ignores=['crop_tracked']))
    logger.debug('Tracking...')
    logger.info('Saving cropped videos to %s', output_dir.as_posix())
    tracked = []
    for i, vfile in enumerate(videos):
        vfile = pathlib.Path(vfile)
        logger.info('Checking video %d of %d: %s', i, len(videos), vfile.name)
        lr = split_crop_track_video(
            vfile,
            FileLocations.get_median_frames_dir(videos_root),
            output_dir,
            max_frames)
        if lr.left is not None:
            tracked.append(lr.left)
        if lr.right is not None:
            tracked.append(lr.right)
    logger.info('Done tracking all videos!')
    return tracked

if __name__ == '__main__':
    ARGS = FileLocations.parse_default_args(
        lambda parser: parser.add_argument(
            '-q',
            '--quiet',
            help='Whether to supress progress output.',
            action='store_true',
            default=False),
        lambda parser: parser.add_argument(
            '-f',
            '--video_root',
            help='(Optional) Path to a folder containing videos to analyze \
                (will search recursively).',
            default=None),
        lambda parser: parser.add_argument(
            '-o',
            '--outdir',
            help='Output directory, where to save the new videos.',
            default=None)
    )
    CropTracker.quiet = ARGS.quiet
    split_crop_track(ARGS.video_root, ARGS.outdir, max_frames=MAX_FRAMES)
