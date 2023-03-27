''' Test some basic CropTracker stuff.
Basically, just make sure it doesn't crash.
'''
import pytest
import tqdm

from swimfunction.video_processing import standardize_video, MultiCropTracker, SplitCropTracker
from swimfunction.pytest_utils import CACHE_DUMP
from swimfunction.pytest_utils import set_test_cache_access
from swimfunction.video_processing import fp_ffmpeg
from swimfunction import FileLocations

N_ITERS = 10

def setup():
    ''' Allows fixture to be called directly if absolutely necessary.
    '''
    set_test_cache_access()

@pytest.fixture(autouse=True)
def __setup():
    setup()
    yield
    # Cleanup
    for fpath in list(CACHE_DUMP.glob('*')):
        fpath.unlink()

def test_standardize_img():
    ''' Can we get a standardized image?
    '''
    print(FileLocations.get_original_videos_dir())
    print(list(FileLocations.get_original_videos_dir().glob('*.avi')))
    vfile = FileLocations.get_original_videos_dir().glob('*.avi').__next__()
    frame = fp_ffmpeg.read_frame(vfile, 1)
    median_fname = FileLocations.get_median_frames_dir(
        FileLocations.get_original_videos_dir()).glob('*.png').__next__()
    median = fp_ffmpeg.freeimage_read(median_fname)
    tracker = SplitCropTracker.SplitCropTracker(
        500, 500, frame.shape[0], frame.shape[1], subsample_factor=1, debug=True)
    standardizer = standardize_video.Standardizer(median, has_divider=True)
    for _ in tqdm.tqdm(range(N_ITERS)):
        standardized = standardizer.get_standardized_frame(frame)
        standardized = standardized[:standardized.shape[0]//2, :]
        corner = tracker.get_corner_of_fish_crop(standardized, is_subsampled=False)
        cropped = tracker.apply_crop(standardized, corner, force_dimensions=True)
    fp_ffmpeg.freeimage_write(standardized, CACHE_DUMP / 'standard.png')
    fp_ffmpeg.freeimage_write(cropped, CACHE_DUMP / 'cropped.png')

def test_split_crop_tracker():
    ''' Can split crop tracker work on a video?
    '''
    vfile = FileLocations.get_original_videos_dir().glob('*.avi').__next__()
    height, width = fp_ffmpeg.VideoData(vfile).shape[:2]
    tracker = SplitCropTracker.SplitCropTracker(
        400, 400, height, width, subsample_factor=8, debug=True)
    tracker.crop_track(
        vfile=vfile,
        output_dir=CACHE_DUMP,
        medians_loc=FileLocations.get_median_frames_dir(FileLocations.get_original_videos_dir()),
        max_frames=63000
    )

def test_multi_crop_tracker():
    ''' Can multi crop tracker work on a video?
    '''
    vfile = FileLocations.get_original_videos_dir().glob('*.avi').__next__()
    height, width = fp_ffmpeg.VideoData(vfile).shape[:2]
    tracker = MultiCropTracker.MultiCropTracker(
        400, 400, height, width, subsample_factor=8, debug=True, nfish=2)
    tracker.crop_track(
        vfile=vfile,
        output_dir=CACHE_DUMP,
        medians_loc=FileLocations.get_median_frames_dir(FileLocations.get_original_videos_dir()),
        max_frames=63000
    )
    # Cleanup
    for fpath in list(CACHE_DUMP.glob(f'{vfile.with_suffix("").name}*')):
        fpath.unlink()

if __name__ == '__main__':
    setup()
    test_standardize_img()
    test_split_crop_tracker()
    test_multi_crop_tracker()
