'''
Until the desired number of frames have been extracted,
    Randomly selects a video
        Randomly saves frame in the video
All frame numbers are relative to the video from which it was extracted.
(No correction is done for CropTracker videos.)
'''

from swimfunction.video_processing import fp_ffmpeg, SplitCropTracker
import argparse
from swimfunction import FileLocations
import json
import numpy
import pathlib
from swimfunction import progress
import subprocess

VERBOSE = False

def print_verbose(x):
    if VERBOSE:
        print(x)

def get_framecount(fname):
        ffprobe_command = [fp_ffmpeg.FFPROBE_BIN,
                '-loglevel', 'fatal',
                '-select_streams', 'V:0',
                '-show_entries', 'stream=duration',
                '-print_format', 'json',
                fname]
        probe = subprocess.run(ffprobe_command, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if probe.returncode != 0:
                raise RuntimeError('Could not read video metadata:\n'+probe.stderr.decode())
        metadata = json.loads(probe.stdout.decode())['streams'][0]
        duration_in_seconds = float(metadata['duration'])
        return fp_ffmpeg.VideoData(fname).fps * duration_in_seconds

def get_right_left_frame_paths(input_path, output_dir, num_sec) -> SplitCropTracker.LeftRight:
    ''' Returns new paths for the left and right side of the frame.
    Adds the .png extension.
    '''
    paths = SplitCropTracker.get_right_left_paths(input_path, output_dir)
    right, left = None, None
    if paths.right is not None:
        right = output_dir / f'{num_sec}_{paths.right.name.replace(".avi", ".png")}'
    if paths.left is not None:
        left = output_dir / f'{num_sec}_{paths.left.name.replace(".avi", ".png")}'
    return SplitCropTracker.LeftRight(left=left, right=right)

def save_random_frame(input_path, output_dir, split_videos):
    input_path = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_dir)
    nframes = get_framecount(input_path.as_posix())
    if nframes is None or nframes == 0:
        print('Failed to calculate number of seconds in the video.')
        print(input_path.as_posix())
        return None
    print_verbose(f'Number of frames: {nframes}')
    _video_padding = 0 # _video_padding seconds at start and end are ignored
    frame_num = numpy.random.randint(low=0, high=nframes)
    frame = fp_ffmpeg.read_frame(input_path.as_posix(), frame_num=frame_num)
    if split_videos:
        paths = get_right_left_frame_paths(input_path, output_dir, frame_num)
        crops = SplitCropTracker.get_right_left_crops(frame)
        fp_ffmpeg.freeimage_write(crops.right, paths.right)
        fp_ffmpeg.freeimage_write(crops.left, paths.left)
    else:
        ofname = output_dir / f'{frame_num}_{input_path.name.replace(".avi",".png")}'
        fp_ffmpeg.freeimage_write(frame, ofname)
    return None

def extract_training_frames_from_videos(video_paths: list, output_dir: str, total_frame_output: int, split_videos: bool):
    '''
    Parameters
    ----------
    video_paths : list
        list of paths to videos
    output_dir : str
        location to save the output frames
    total_frame_output : int
        number of randomly selected frames to extract
    split_videos : bool
        Whether to split
    '''
    video_paths = list(filter(lambda x: pathlib.Path(x).exists(), video_paths))
    print(f'Getting {total_frame_output} frames from {len(video_paths)} videos.')
    i = 0
    progress.init(total_frame_output)
    for input_path in numpy.random.choice(video_paths, total_frame_output):
        i += 1
        progress.progress(i, status=f'Working on frame {i}')
        save_random_frame(input_path, output_dir, split_videos)

def extract_training_frames(video_root_dir, output_dir, total_frame_output, split_videos):
    video_root_dir = pathlib.Path(video_root_dir)
    output_dir = pathlib.Path(output_dir)
    video_paths = list(FileLocations.find_video_files(video_root_dir))
    extract_training_frames_from_videos(video_paths, output_dir, total_frame_output, split_videos)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--video_root_dir', default=FileLocations.get_normalized_videos_dir())
    parser.add_argument('-o', '--output_dir', default=FileLocations.get_training_root_dir(), help='Location to save training data')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-t', '--total_frame_output', type=int, default=400)
    parser.add_argument('-s', '--split_videos', action='store_true', default=False,
        help='Whether the videos should be split into right and left sides \
            (use this if the input videos have two fish, one on each side of the video).')
    args = parser.parse_args()
    if args.verbose:
        VERBOSE = True
    extract_training_frames(args.video_root_dir, args.output_dir, args.total_frame_output, args.split_videos)
