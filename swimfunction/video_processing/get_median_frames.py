''' Calculate and store median (background) frames for videos.
'''
import argparse
import pathlib
from typing import List
import numpy
import tqdm

from swimfunction.context_managers.FileLock import FileLock
from swimfunction.video_processing import fp_ffmpeg
from swimfunction import loggers
from swimfunction import FileLocations

NUM_FRAMES_TO_TAKE_FOR_MEDIAN = 1000

def get_video_median_frame(video_loc: pathlib.Path) -> numpy.ndarray:
    ''' Selects frames evenly throughout a video
    and calculates a pixel-wise median frame
    that represents the background image.
    '''
    video_loc = pathlib.Path(video_loc)
    if not video_loc.exists():
        return None
    video_nframes = fp_ffmpeg.get_nframes(video_loc.as_posix())
    video_data = fp_ffmpeg.VideoData(video_loc.as_posix())
    width = video_data.shape[0]
    height = video_data.shape[1]
    num_frames = min(NUM_FRAMES_TO_TAKE_FOR_MEDIAN, video_nframes)
    frames = numpy.empty((num_frames, width, height, 3), dtype=video_data.dtype)
    target_frames = numpy.linspace(0, video_nframes-3, num=num_frames, dtype=int)
    counter = 0
    for frame_num, frame in tqdm.tqdm(
            enumerate(fp_ffmpeg.read_video(video_loc)),
            total=video_nframes):
        if frame_num in target_frames:
            frames[counter, :, :, :] = frame
            counter += 1
    f_median = numpy.median(frames, axis=0)
    numpy.clip(f_median, 0, 255, out=f_median)
    if f_median.dtype != numpy.uint8:
        f_median = f_median.astype(numpy.uint8)
    return f_median

def save_median_frame(
        video_loc: pathlib.Path,
        output_fpath: pathlib.Path):
    ''' Save the median frame for the video in the given location
    '''
    logger = loggers.get_video_processing_logger(__name__)
    video_loc = pathlib.Path(video_loc)
    output_fpath = pathlib.Path(output_fpath)
    logger.debug('Saving median: %s', video_loc.name)
    f_median = get_video_median_frame(video_loc)
    fp_ffmpeg.freeimage_write(f_median, output_fpath.as_posix())

def get_median_path(
        medians_loc: pathlib.Path,
        video_fname: pathlib.Path) -> pathlib.Path:
    ''' Get the path and name of the median file
    that would be produced from the video.
    '''
    return (
        pathlib.Path(medians_loc) \
            / f'median_{pathlib.Path(video_fname).name}'
    ).with_suffix('.png')

def get_video_median(
        video_fname: pathlib.Path,
        medians_loc: pathlib.Path) -> numpy.ndarray:
    ''' Finds video median matching the fish in video_fname.
    Crops the video median if two fish exist in the median but only one exists in the video.
    '''
    logger = loggers.get_video_processing_logger(__name__)
    video_fname = pathlib.Path(video_fname)
    medians_loc = pathlib.Path(medians_loc)
    median_fpath = get_median_path(medians_loc, video_fname)
    if not median_fpath.exists():
        logger.info(
            'Could not find median for %s in %s, so we\'re calculating it...',
            video_fname.name, medians_loc.as_posix())
        save_median_frame(video_fname, median_fpath)
        logger.info('Median calculated!')
    median = fp_ffmpeg.freeimage_read(median_fpath.as_posix())
    return median

def write_all_median_frames_for_videos(
        video_paths: List[pathlib.Path],
        output_dir: pathlib.Path):
    ''' For each video, calculate and save its median frame.
    '''
    output_dir = pathlib.Path(output_dir)
    for video_loc in video_paths:
        output_loc = get_median_path(output_dir, video_loc)
        # For some reason, numpy.median does not like
        # to be called inside a separate thread.
        # It broke when I tried concurrent.futures.
        with FileLock(
                lock_dir=output_dir,
                lock_name=output_loc.name,
                output_path=output_loc) as fl:
            if fl:
                save_median_frame(video_loc, output_loc)

def write_all_median_frames(video_root: pathlib.Path):
    ''' Write a median (background) frame for every video
    recursively found in the root directory.
    Does not check video name formatting.
    '''
    logger = loggers.get_video_processing_logger(__name__)
    output_dir = FileLocations.get_median_frames_dir(video_root)
    logger.info('Saving median frames to %s', output_dir.as_posix())
    video_locs = list(FileLocations.find_video_files(root_path=video_root))
    write_all_median_frames_for_videos(video_locs, output_dir)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '-d',
        '--video_dir',
        default=FileLocations.get_original_videos_dir(),
        help='Location of videos. A subfolder "median_frames" will be created.')
    write_all_median_frames(PARSER.parse_args().video_dir)
