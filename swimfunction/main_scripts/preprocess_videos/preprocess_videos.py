''' This run necessary preprocessing.
It will take an extremely long time to run as a standalone.
Remember, you can run multiple instances concurrently.
   Those are instance-safe. I couldn't use multithreading as it's traditionally done because
   ffmpeg must be run in the main thread of a program, or something like that. Anyway, it
   threw an error when I tried.
Well, have fun!
'''

from swimfunction.global_config.config import config
from swimfunction.video_processing import SplitCropTracker
from swimfunction.video_processing import MultiCropTracker

from swimfunction import FileLocations
from swimfunction import loggers

MAX_FRAMES = config.getint('VIDEO', 'fps') * 60 * 15 # 15 min swims

def preprocess_videos(is_multianimal, max_frames):
    ''' Runs the relevant child class of CropTracker.
    '''
    original_location = FileLocations.get_original_videos_dir()
    normalized_location = FileLocations.get_normalized_videos_dir()
    if is_multianimal:
        MultiCropTracker.run_multi_crop_tracker(original_location, normalized_location, max_frames=MAX_FRAMES)
    else:
        SplitCropTracker.split_crop_track(
            original_location,
            normalized_location,
            max_frames=MAX_FRAMES,
            logger=loggers.get_video_processing_logger(__name__))

if __name__ == '__main__':
    ARGS = FileLocations.parse_default_args(
        lambda parser: parser.add_argument(
            '-m', '--multianimal', action='store_true', default=False,
            help='By default, it will assume you want to SplitCropTrack \
            (two fish separated by a divider). This flag lets you do \
            multi animal crop tracking (multiple fish, no divider).'),
    )
    preprocess_videos(ARGS.multianimal)
