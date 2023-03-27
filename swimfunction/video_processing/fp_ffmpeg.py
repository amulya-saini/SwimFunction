''' freeimage and ffmpeg have some quirky features.
This wrapper does some tricks to deal with those
in ways that I found convenient.
'''

import json
import pathlib
import subprocess
from zplib.image import ffmpeg as _ffmpeg
import freeimage as _freeimage
import numpy

FFPROBE_BIN = _ffmpeg.FFPROBE_BIN
FFMPEG_BIN = _ffmpeg.FFMPEG_BIN

VideoData = _ffmpeg.VideoData

MAX_FRAME_NUM_TO_READ_SLOWLY = 10

class _Memory:
    def __init__(self):
        self.memory = {}
    def memory_decorator(self, handle_filename_function):
        ''' Decorator that uses cached value if possible.
        '''
        def inner(fname):
            if fname not in self.memory:
                self.memory[fname] = handle_filename_function(fname)
            return self.memory[fname]
        return inner

_OFFSET_MEMORY = _Memory()
@_OFFSET_MEMORY.memory_decorator
def _calculate_frame_offset(fname):
    # The "_ffmpeg.read_frame" function unfortunately does not always get the exact frame.
    # It often gets the num - 2 frame. However, we need to calculate the offest to be sure.
    n_frames = MAX_FRAME_NUM_TO_READ_SLOWLY * 2
    first_n_frames = []
    for i, frame in enumerate(_ffmpeg.read_video(fname)):
        first_n_frames.append(frame)
        if i == n_frames:
            break
    # Find out which frame we actually get when we ask for a specific frame number.
    target_frame_num = int(MAX_FRAME_NUM_TO_READ_SLOWLY * 1.5)
    test_frame = _ffmpeg.read_frame(fname, frame_num=target_frame_num)
    matches = [numpy.all(numpy.isclose(frame, test_frame)) for frame in first_n_frames]
    actual_frame_num = numpy.where(matches)[0][0]
    if numpy.where(matches)[0].shape[0] != 1:
        raise RuntimeError(f'Failed to calculate frame offset for video {fname}')
    # The difference between the frame number we asked for and the actual frame number returned
    # is the desired offset.
    return target_frame_num - actual_frame_num

_VIDEO_DATA_MEMORY = _Memory()
@_VIDEO_DATA_MEMORY.memory_decorator
def _get_video_data(fname):
    return VideoData(fname, force_grayscale=False)

_VIDEO_DATA_GRAYSCALE_MEMORY = _Memory()
@_VIDEO_DATA_GRAYSCALE_MEMORY.memory_decorator
def _get_video_data_grayscale(fname):
    return VideoData(fname, force_grayscale=True)

def get_video_data(vfile: str, force_grayscale: bool):
    '''Get the video data for a video file.

    Parameters
    ----------
    vfile : str
    force_grayscale : bool

    Returns
    -------
    _ffmpeg.VideoData
    '''
    if force_grayscale:
        return _get_video_data_grayscale(vfile)
    return _get_video_data(vfile)

class VideoWriter(_ffmpeg.VideoWriter):
    ''' Wrapper for ffmpeg.VideoWriter
    '''
    def __init__(
            self, framerate, outpath, preset=None,
            lossless=False, verbose=True, is_conventional=False, **h264_options):
        '''
        Parameters
        ----------
        frames
        framerate
        outpath
        threads : default=None
        verbose : default=True
        is_conventional : bool, default=False
            Whether the input follows the top-left origin convention.
            zplib.image.ffmpeg default returns not-conventional images.
            If True, will transpose all frames.
        '''
        super().__init__(
            framerate, outpath, preset=preset,
            lossless=lossless, verbose=verbose, **h264_options)
        self.is_conventional = is_conventional

    def encode_frame(self, frame):
        if self.is_conventional:
            super().encode_frame(transpose_img(frame))
        else:
            super().encode_frame(frame)

class LosslessVideoWriter(_ffmpeg.LosslessVideoWriter):
    ''' Wrapper for ffmpeg.LosslessVideoWriter
    '''
    def __init__(self, framerate, outpath, threads=None, verbose=True, is_conventional=False):
        super().__init__(framerate, outpath, threads=threads, verbose=verbose)
        self.is_conventional = is_conventional

    def encode_frame(self, frame):
        if self.is_conventional:
            super().encode_frame(transpose_img(frame))
        else:
            super().encode_frame(frame)

def transpose_img(img):
    ''' Transposes an ffmpeg or freeimage image into standard
    computer vision (pyplot compatible) image, where the origin
    is in the top-left corner, first dimension is Y (going down),
    second dimension is X (going to the right).
    '''
    if img is None:
        return img
    if len(img.shape) == 3:
        return numpy.transpose(img, axes=(1, 0, 2))
    return numpy.transpose(img)

def _read_frame_slowly(infile, frame_num, force_grayscale):
    frame = None
    for i, frame in enumerate(_ffmpeg.read_video(infile, force_grayscale)):
        if i == frame_num:
            break
    return frame

def _yields_transposed(generates_conventional_frames):
    for frame in generates_conventional_frames:
        yield transpose_img(frame)

def read_frame(
        vfile,
        frame_num=None,
        frame_time=None,
        video_data=None,
        force_grayscale=False,
        transpose_to_match_convention=False):
    ''' Fixes two major issues with ffmpeg.read_frame:
    failure to get early frames, and frame_num offset.
    ffmpeg.read_frame only works for frame_num 9 and above.
    Basically, ffmpeg.read_frame will always return frame_num - offset.
    Ask ffmpeg for frame 40, and you'll get frame 38. So, instead, this adds the offset back.
    Ask this function for 40, and internally it will ask for 42.
    '''
    if vfile is None:
        return None
    if video_data is None:
        video_data = get_video_data(vfile, force_grayscale)
    frame = None
    if frame_num is not None:
        if frame_num <= MAX_FRAME_NUM_TO_READ_SLOWLY:
            frame = _read_frame_slowly(vfile, frame_num, force_grayscale)
        else:
            frame = _ffmpeg.read_frame(
                vfile,
                frame_num=(frame_num + _calculate_frame_offset(vfile)),
                video_data=video_data,
                force_grayscale=force_grayscale)
    elif frame_time is not None:
        frame = _ffmpeg.read_frame(
            vfile,
            frame_time=frame_time,
            video_data=video_data,
            force_grayscale=force_grayscale)
    return frame if not transpose_to_match_convention else transpose_img(frame)

def generate_consecutive_frames(
        vfile, nframes, start_frame,
        video_data=None, force_grayscale=False, transpose_to_match_convention=False):
    """ Code taken and adapted from zplib.image.ffmpeg
    Efficiently locates the desired start, then returns a certain number of frames.
    Parameters
    ----------
    vfile
        filename to open
    nframes : int
    start_frame : int
        frame number to start retrieving (0-indexed)
    video_data: VideoData
        Contains results of ffprobe.
        Best to precalculate this if you will be getting lots of frames.
        If None, then a VideoData instance will be constructed.
    force_grayscale: default=False
        If True, return uint8 grayscale frames, otherwise
        returns rgb frames. Note: force_grayscale is only effective if video_data is None,
        otherwise you must set this flag when constructing the VideoData instance.
    transpose_to_match_convention : bool, optional
    """
    if video_data is None:
        video_data = get_video_data(vfile, force_grayscale)
    start_frame += _calculate_frame_offset(vfile)
    frame_time = start_frame / video_data.fps

    command = [
        FFMPEG_BIN,
        '-loglevel', 'fatal',
        '-nostdin', # do not expect interaction, do not fuss with tty settings
        '-accurate_seek',
        '-ss', str(frame_time), # seek to a specific time as a fraction of seconds
        '-i', vfile,
        '-map', '0:V:0', # grab video channel 0, just like ffprobe above
        '-frames:v', f'{nframes}', # grab only this many frames
        '-f', 'rawvideo', # write raw image data
        '-pix_fmt', video_data.pixel_format_out,
        '-' # pipe output to stdout
    ]

    ff = subprocess.Popen(
        command, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while True:
        frame = video_data.get_frame_array(ff.stdout)
        if frame is None:
            break
        if transpose_to_match_convention:
            frame = transpose_img(frame)
        yield frame
    ff.stdout.close()
    ff.wait()
    if ff.returncode != 0:
        raise RuntimeError('Could not read video data:\n'+ff.stderr.read().decode())

def read_frames(
        vfile,
        frame_nums=None,
        video_data=None,
        force_grayscale=False,
        transpose_to_match_convention=False):
    '''
    Parameters
    ----------
    vfile : str or pathlib.Path
        location of video file
    frame_nums : list, optional
        frame numbers (use frame_nums or frame_seconds, not both)
    frame_seconds : list, optional
        frame positions in seconds (use frame_nums or frame_seconds, not both)
    video_data : _ffmpeg.VideoData, optional
    force_grayscale : bool, optional
    transpose_to_match_convention : bool, optional
    '''
    if video_data is None:
        video_data = get_video_data(vfile, force_grayscale)
    if frame_nums is None:
        frame_nums = []
    can_use_consecutive = True
    for i, fnum in enumerate(frame_nums[:-1]):
        if fnum != frame_nums[i + 1] - 1:
            can_use_consecutive = False
    if can_use_consecutive:
        yield from generate_consecutive_frames(
            vfile, len(frame_nums), frame_nums[0],
            video_data, force_grayscale, transpose_to_match_convention)
    else:
        for fnum in frame_nums:
            yield read_frame(
                vfile,
                frame_num=fnum,
                video_data=video_data,
                force_grayscale=force_grayscale,
                transpose_to_match_convention=transpose_to_match_convention)

def read_video(
        infile,
        force_grayscale=False,
        transpose_to_match_convention=False):
    ''' Generate frames from a video.
    '''
    for frame in _ffmpeg.read_video(infile, force_grayscale):
        if transpose_to_match_convention:
            yield transpose_img(frame)
        else:
            yield frame

def write_video(
        frames,
        framerate: float,
        outpath: pathlib.Path,
        preset=None,
        lossless: bool=False,
        verbose: bool=True,
        is_conventional: bool=False,
        **h264_opts):
    '''
    Parameters
    ----------
    frames : list or generator
    framerate : float
    outpath : pathlib.Path
    preset : default=None
    lossless : bool, default=False
    verbose : bool, default=True
    is_conventional : bool, default=False
        Whether the input follows the top-left origin convention.
        zplib.image.ffmpeg default returns not-conventional images.
        If True, will transpose all frames.
    '''
    if is_conventional:
        frames = _yields_transposed(frames)
    _ffmpeg.write_video(
        frames, framerate, outpath,
        preset=preset, lossless=lossless,
        verbose=verbose, **h264_opts)

def write_lossless_video(
        frames,
        framerate,
        outpath,
        threads=None,
        verbose=True,
        is_conventional=False):
    '''
    Parameters
    ----------
    frames
    framerate
    outpath
    threads : default=None
    verbose : default=True
    is_conventional : bool, default=False
        Whether the input follows the top-left origin convention.
        zplib.image.ffmpeg default returns not-conventional images.
        If True, will transpose all frames.
    '''
    if is_conventional:
        frames = _yields_transposed(frames)
    _ffmpeg.write_lossless_video(
        frames, framerate, outpath,
        threads=threads, verbose=verbose)

def freeimage_read(filename, flags=0, transpose_to_match_convention=False):
    ''' Wrapper for freeimage.read

    Parameters
    ----------
    filename
    flags
    transpose_to_match_convention : bool, default=False
        Whether the output should follow the top-left origin convention.
        zplib.image.ffmpeg default returns not-conventional images.
    '''
    img = _freeimage.read(filename, flags)
    if transpose_to_match_convention:
        img = transpose_img(img)
    return img

def freeimage_write(array, filename, flags=0, is_conventional=False):
    ''' Wrapper for freeimage.write

    Parameters
    ----------
    array
    filename
    flags
    is_conventional : bool, default=False
        Whether the input follows the top-left origin convention.
        zplib.image.ffmpeg default returns not-conventional images.
    '''
    if is_conventional:
        _freeimage.write(transpose_img(array), filename, flags)
    else:
        _freeimage.write(array, filename, flags)

_MEMORY = _Memory()
@_MEMORY.memory_decorator
def get_nframes(fname):
    ''' Get number of frames in the video
    '''
    if fname is None:
        return 0
    fname = pathlib.Path(fname).expanduser().as_posix()
    ffprobe_command = [
        FFPROBE_BIN,
        '-loglevel', 'fatal',
        '-select_streams', 'V:0',
        '-show_entries', 'stream=nb_frames',
        '-print_format', 'json',
        fname]
    probe = subprocess.run(
        ffprobe_command, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if probe.returncode != 0:
        raise RuntimeError('Could not read video metadata:\n'+probe.stderr.decode())
    metadata = json.loads(probe.stdout.decode())['streams'][0]
    nframes = int(metadata['nb_frames'])
    return nframes
