''' Frame by frame crops to the most likely location of each fish in the video.

Crop-tracking turns a large video into a cropped video
    that centers the fish in-frame.

Multi-animal crop-tracking turns one video into several
    crop-tracked videos, one for each fish.

Warning! Multi-animal crop-tracking is experimental, unpublished, and not guaranteed to work well.

IMPORTANT DEVELOPMENT POINT (regular users can ignore this):
Internal functions modify images using [width, height] indexing, despite convention.
This is to retain freeimage compatibility (see fp_ffmpeg).
'''

from collections import namedtuple
import pathlib
from scipy import ndimage, optimize
from scipy.ndimage.measurements import center_of_mass
import numpy
from sklearn.cluster import KMeans
from swimfunction.context_managers.FileLock import FileLock
from swimfunction.global_config.config import config
from swimfunction.video_processing.image_processing import threshold
from swimfunction.video_processing import standardize_video, fp_ffmpeg
from swimfunction.video_processing.CropTracker import CropTracker, CropResult, \
    Point, CropTrackerLog, WritersLoggers, find_existing_logfile
from swimfunction import FileLocations

# In full-sized image, I observed four fish with
# 8640, 6720, 8704, 8512 pixels.
# So, this is a conservative minimum and maxiumum, I think.
MIN_FISH_SIZE = 3000
MAX_FISH_SIZE = 13000
BACKGROUND_LABEL = 0

BINARIZING_THRESHOLD = 140

MAX_FRAMES = config.getint('VIDEO', 'fps') * 60 * 15 # 15 min swims

Standardizers = namedtuple('Standardizers', ['full', 'subsampled'])

FindFishResult = namedtuple('FindFishResult', ['center', 'corner', 'crop'])

class MultiCropTracker(CropTracker):
    ''' Splits video into smaller videos,
    each of which follows a different fish and masks the others.
    '''

    __slots__ = ['nfish', 'prior_centers']

    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------
        nfish: int
            Number of fish in the video.
        '''
        super().__init__(*args, **kwargs)
        self.nfish = None if 'nfish' not in kwargs else kwargs['nfish']
        self.prior_centers = []

    @staticmethod
    def from_logfile(logfname, class_type=None):
        return super().from_logfile(logfname, class_type=MultiCropTracker)

    ##### Implement Abstract Methods

    def confirm_number_of_fish(self, vfile, standardizers):
        ''' Detect number of fish, make sure the user agrees the number is correct.
        '''
        first_frame = fp_ffmpeg.read_frame(vfile, frame_num=0)
        if self.nfish is None or self.nfish == 0:
            found_fish = self.find_fish_in_frame(first_frame, standardizers)
            self.nfish = len(found_fish)
            check_str = f'Found {self.nfish} fish. Is this correct?'
            if not self.debug and input(check_str).strip().upper() != 'Y':
                self.nfish = int(input('How many fish?').strip())
        else:
            # img_ss is image subsampled
            img_ss = standardizers.subsampled.get_standardized_frame(
                self.subsample_image(first_frame, self.subsample_factor)
            )
            mask_ss = threshold(img_ss, th=BINARIZING_THRESHOLD)
            points = numpy.asarray(numpy.where(mask_ss == 1)).T
            centers = KMeans(n_clusters=self.nfish).fit(points).cluster_centers_.tolist()
            self.prior_centers = [
                self.rescale_point(Point(*c), self.subsample_factor)
                for c in centers
            ]

    def get_video_writers_and_loggers(
            self, vfile, output_dir, nframes, standardizers=None) -> WritersLoggers:
        output_dir = pathlib.Path(output_dir)
        framerate = fp_ffmpeg.VideoData(vfile).fps
        self.confirm_number_of_fish(vfile, standardizers)
        vfile = pathlib.Path(vfile).expanduser().resolve()
        output_vfiles = [
            output_dir / vfile.with_name(
                f'{vfile.with_suffix("").name}_TF{i}'
                ).with_suffix(vfile.suffix).name
            for i in range(self.nfish)
        ]
        video_writers = []
        ct_loggers = []
        for output_vfile in output_vfiles:
            # The log file is the most important bit.
            # If the log file was not produced, the script didn't finish.
            existing_logfile = find_existing_logfile(output_vfile)
            if existing_logfile is not None:
                return WritersLoggers([], []) # Exists, so just return.
            video_writers.append(fp_ffmpeg.VideoWriter(
                framerate,
                output_vfile,
                preset=None,
                lossless=False,
                verbose=False,
                is_conventional=False))
            ct_loggers.append(CropTrackerLog().setup_ct_logger(
                vfile, output_vfile, self, nframes))
        return WritersLoggers(video_writers, ct_loggers)

    def get_standardizers(self, v_median):
        standardizer = standardize_video.Standardizer(v_median, has_divider=False)
        subsample_inner_content = standardize_video.StandardizerInnerContent(
            self.subsample_image(standardizer.target_median, self.subsample_factor),
            standardizer.normalizing_lut,
            self.subsample_image(standardizer.v_median_normalized, self.subsample_factor)
        )
        subsample_standardizer = standardize_video.Standardizer(
            self.subsample_image(v_median, self.subsample_factor),
            has_divider=False,
            inner_content=subsample_inner_content)
        return Standardizers(standardizer, subsample_standardizer)

    def handle_frame(
            self,
            frame_i,
            frame,
            video_writers,
            ct_loggers,
            standardizers):
        for fish_i, crop_result in self.generate_labeled_crop_results(
                frame, standardizers):
            video_writers[fish_i].encode_frame(crop_result.img)
            ct_loggers[fish_i].log_corner(frame_i, crop_result.corner)

    ##### Helper Methods

    def generate_labeled_crop_results(self, frame, standardizers: Standardizers):
        ''' For each fish in frame, get its CropResult.
        '''
        found_fish = self.find_fish_in_frame(frame, standardizers)
        centers = [find_result.center for find_result in found_fish]
        if self.prior_centers is None or not self.prior_centers:
            self.prior_centers = centers
        pairwise_distance_matrix = [
            [(((pc.x - c.x)**2) + ((pc.y - c.y)**2))**.5 for pc in self.prior_centers]
            for c in centers
        ]
        for center_i, fish_i in MultiCropTracker.bipartite_matching(pairwise_distance_matrix):
            if center_i >= len(centers):
                # Only keep HA output for centers that exist.
                continue
            self.prior_centers[fish_i] = centers[center_i]
            yield fish_i, CropResult(found_fish[center_i].crop, found_fish[center_i].corner)

    def find_fish_in_frame(self, img: numpy.ndarray, standardizers: Standardizers) -> list:
        ''' Identifies all fish in frame and creates a FindFishResult for each one.
        Each FindFishResult contains a center, corner,
        and normalized crop with other fish masked out.

        Returns
        -------
        identified_fish : list of FindFishResult
        '''
        objs, fish_labels, centers_of_fish = self.get_labeled_fish_mask(img, standardizers)
        size_of_crop = self.crop_width * self.crop_height
        standardize_first_costs_less = size_of_crop * len(fish_labels) > img.shape[0] * img.shape[1]
        if standardize_first_costs_less:
            img = standardizers.full.get_standardized_frame(img)
        identified_fish = []
        for center, label in zip(centers_of_fish, fish_labels):
            if numpy.isnan(center[0]):
                continue
            center = Point(int(center[0]), int(center[1]))
            corner = self.center_to_corner(center, self.crop_width, self.crop_height, img.shape)
            crop_area = self.corner_to_crop_area(corner)
            crop = None
            if standardize_first_costs_less:
                crop = standardize_video.crop_img(img, crop_area, copy=True)
            else:
                crop = standardizers.full.get_standardized_frame(img, crop_area)
            crop = self.mask_other_fish(
                crop,
                standardize_video.crop_img(objs, crop_area),
                label)
            identified_fish.append(FindFishResult(center, corner, crop))
        return identified_fish

    @staticmethod
    def mask_other_fish(img: numpy.ndarray, objs: numpy.ndarray, fish_label: int):
        ''' Masks every object with fish_label that is not current_fish.
        '''
        where_not_fish_mask = numpy.logical_and(objs != fish_label, objs)
        buffer_val = standardize_video.get_buffer_val(img.dtype, fill_with_max=True)
        img[where_not_fish_mask] = buffer_val
        return img

    def get_labeled_fish_mask(self, img, standardizers):
        '''
        Returns
        -------
        labeled_mask_result : tuple
            objects
                image where each pixel has an object label
            fish_labels
                labels that correspond to probable fish
            centers_of_fish
                center for each fish label
        '''
        # img_ss is image subsampled
        img_ss = standardizers.subsampled.get_standardized_frame(
            self.subsample_image(img, self.subsample_factor)
        )
        mask_ss = threshold(img_ss, th=BINARIZING_THRESHOLD)
        # Connect any pixels joined at a corner.
        mask_ss = ndimage.grey_dilation(mask_ss, size=(3, 3))
        objs_ss, labels = ndimage.label(mask_ss, structure=numpy.ones(9).reshape((3, 3)))
        fish_labels, objs_ss = self.isolate_fish_labels(objs_ss, labels, is_subsampled=True)
        centers_of_fish = list(map(
            lambda c, tracker=self: tracker.rescale_point(Point(*c), tracker.subsample_factor),
            center_of_mass(mask_ss, objs_ss, fish_labels)))
        objs_upsampled = self.upsample_image(objs_ss, self.subsample_factor, img.shape)
        return objs_upsampled, fish_labels, centers_of_fish

    def isolate_fish_labels(self, objs: numpy.ndarray, max_label: int, is_subsampled: bool):
        ''' Identify which labels in objs likely belong to real fish.

        Returns
        -------
        tuple
            fish_labels
            objs
        '''
        # Start at 1 to ignore the background (0)
        obj_labels = numpy.arange(1, max_label+1)
        fish_labels = []
        sizes = []
        for label in obj_labels:
            match = objs == label
            size = objs[match].size
            if is_subsampled:
                size *= (self.subsample_factor**2)
            if MIN_FISH_SIZE <= size <= MAX_FISH_SIZE:
                fish_labels.append(label)
                sizes.append(size)
        fish_labels = numpy.asarray(fish_labels)
        if self.nfish > 0 and len(fish_labels) > self.nfish:
            # HANDLE EXTRA OBJECTS
            # Take the largest nfish objects.
            fish_labels = fish_labels[numpy.argsort(sizes)[-int(self.nfish):]]
        return fish_labels, objs

    @staticmethod
    def bipartite_matching(cost_matrix: list) -> numpy.ndarray:
        ''' Finds pairing between bipartite graph that minimizes sum cost.
        '''
        if cost_matrix is None or not cost_matrix:
            return numpy.array([])
        return numpy.asarray(optimize.linear_sum_assignment(cost_matrix)).T

def run_multi_crop_tracker(video_root=None, outdir=None, num_fish=None, max_frames=MAX_FRAMES):
    ''' Crop track all videos according to args.
    If crop-tracked video already exists, it will be skipped.

    Parameters
    ----------
    video_root
    outdir
    '''
    print('Crop-tracking turns a large video into a cropped video that centers the fish in-frame.')
    print('Multi-animal crop-tracking turns one video into crop-tracked videos, one for each fish.')
    print('Warning! Multi-animal crop-tracking is experimental, unpublished, and not guaranteed.')
    videos_root = FileLocations.get_original_videos_dir() \
        if video_root is None else pathlib.Path(video_root)
    output_dir = FileLocations.get_normalized_videos_dir() \
        if outdir is None else pathlib.Path(outdir)
    output_dir = output_dir.expanduser().resolve()
    crop_width = config.getint('CROP_TRACKER', 'crop_width')
    crop_height = config.getint('CROP_TRACKER', 'crop_height')
    subsample_factor = config.getint('CROP_TRACKER', 'subsample_factor')
    # Do not crop track videos in the output directory
    videos = list(filter(
        lambda path: pathlib.Path(path).expanduser().resolve().parent != output_dir,
        FileLocations.filter_out_unparseable_files(
            FileLocations.find_video_files(videos_root))))
    print('Tracking...')
    print(f'Saving cropped videos to {output_dir.as_posix()}')
    for i, vfile in enumerate(videos):
        vfile = pathlib.Path(vfile)
        with FileLock(
                lock_dir=output_dir,
                lock_name=vfile.name) as lock_acquired:
            if lock_acquired:
                print(f'WORKING ON VIDEO {i} OF {len(videos)}: {vfile.name}')
                frame = fp_ffmpeg.read_frame(vfile, frame_num=0)
                full_width, full_height = frame.shape[:2]
                tracker = MultiCropTracker(
                    crop_width, crop_height,
                    full_width, full_height,
                    subsample_factor, nfish=num_fish)
                tracker.crop_track(
                    vfile,
                    output_dir,
                    FileLocations.get_median_frames_dir(videos_root),
                    max_frames)
    print('Done!')

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
            help='(Optional) Output directory, where to save the new videos.',
            default=None),
        lambda parser: parser.add_argument(
            '-n',
            '--num_fish',
            help='Number of fish in the video.',
            type=int,
            default=None),
    )
    CropTracker.quiet = ARGS.quiet
    run_multi_crop_tracker(ARGS.video_root, ARGS.outdir, ARGS.num_fish)
