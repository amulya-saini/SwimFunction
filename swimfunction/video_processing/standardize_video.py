''' Performs background subtraction and normalization.
IMPORTANT:
Images are assumed to be in [width, height] indexing, despite convention.
This is to retain ffmpeg and freeimage compatibility.
'''

from collections import namedtuple
import numpy

from swimfunction.video_processing import fp_ffmpeg
from swimfunction import FileLocations

# To grayscale using weights from
# https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_rgb_to_gray.html
RGB_WEIGHTS = numpy.asarray([0.2125, 0.7154, 0.0721])

CropArea = namedtuple('CropArea', ['top_left_x', 'top_left_y', 'width', 'height'])

StandardizerInnerContent = namedtuple(
    'StandardizerInnerContent',
    ['target_median', 'normalizing_lut', 'v_median_normalized'])

class Standardizer:
    ''' Subtracts median and normalizes a full-sized (left and right sides not split apart) video.
    '''
    def __init__(
            self, v_median,
            has_divider,
            inner_content: StandardizerInnerContent=None):
        self.v_median = v_median
        if inner_content is not None:
            self.target_median = inner_content.target_median
            self.normalizing_lut = inner_content.normalizing_lut
            self.v_median_normalized = inner_content.v_median_normalized
        else:
            if has_divider:
                median_path = FileLocations.GlobalFishResourcePaths.target_median_frame_with_divider
            else:
                median_path = FileLocations.GlobalFishResourcePaths\
                    .target_median_frame_without_divider
            self.target_median = fp_ffmpeg.freeimage_read(median_path)
            self.normalizing_lut = get_normalizing_lut(self.v_median, self.target_median)
            vmn = self.apply_normalizing_function(self.v_median)
            self.v_median_normalized = vmn - vmn.max()

    def apply_normalizing_function(self, frame):
        ''' Converts values from each channel of a frame
                into a normalization like the target_median_frame,
                based on median_frame of the video.
                Also grayscales that frame.
        Returns
        -------
        normalized_frame: numpy.ndarray
            dtype is float
        '''
        return self.normalizing_lut[0, frame[:, :, 0]] \
            + self.normalizing_lut[1, frame[:, :, 1]] \
            + self.normalizing_lut[2, frame[:, :, 2]]

    def get_standardized_frame(self, rgb_frame: numpy.ndarray, crop_area: CropArea=None):
        ''' Subtracts the background and uses a normalizing function.

        Parameters
        ----------
        rgb_frame : numpy.ndarray
        crop_area : CropArea, optional
        '''
        normalized = self.apply_normalizing_function(
            crop_img(rgb_frame, crop_area)
        )
        bg_subtracted = subtract_median(
            normalized,
            crop_img(self.v_median_normalized, crop_area)
        )
        return buffer_to_size(bg_subtracted, crop_area, fill_high=True)

def _to_uint8_by_clip(frame):
    # Note: this is faster than numpy.clip
    # because numpy.clip creates an intermediate array of dtype float.
    clipped = frame.astype(numpy.uint8)
    clipped[frame > 255] = 255
    clipped[frame < 0] = 0
    return clipped

def get_buffer_val(dtype, fill_with_max: bool):
    ''' Get a filler value based on the dtype.
    Either the largest or smallest possible value (depending on fill_with_max)
    '''
    return numpy.iinfo(dtype).max if fill_with_max else numpy.iinfo(dtype).min

def buffer_to_size(crop: numpy.ndarray, crop_area: CropArea, fill_high: bool):
    ''' Forces an exact size for the image.
    Overflow is filled either as dtype max (fill_high=True) or min.
    '''
    if crop_area is None \
        or (crop.shape[0] == crop_area.width \
            and crop.shape[1] == crop_area.height):
        return crop
    buffer_val = get_buffer_val(crop.dtype, fill_high)
    buffered = None
    if len(crop.shape) == 3:
        buffered = numpy.full((crop_area.width, crop_area.height, 3), buffer_val, dtype=crop.dtype)
    else:
        buffered = numpy.full((crop_area.width, crop_area.height), buffer_val, dtype=crop.dtype)
    buffered[:crop.shape[0], :crop.shape[1]] = crop
    return buffered

def crop_img(img, crop: CropArea, copy=False) -> numpy.ndarray:
    ''' Crops an image using slice indexing. If copy, will return a new array.
    '''
    if crop is None:
        return img
    bottom_right_x = min(crop.top_left_x + crop.width, img.shape[0])
    bottom_right_y = min(crop.top_left_y + crop.height, img.shape[1])
    cropped_img = img[
        crop.top_left_x:bottom_right_x,
        crop.top_left_y:bottom_right_y]
    return cropped_img.copy() if copy else cropped_img

def subtract_median(frame, v_median):
    ''' Subtracts the median frame and converts to unit8 by clipping values.
    '''
    if not numpy.issubdtype(frame.dtype, numpy.floating):
        frame = frame.astype(float)
    return _to_uint8_by_clip(frame - v_median)

def get_normalizing_lut(v_median, target_median, apply_grayscale_weights=True):
    ''' NOTE: normalizing look up table also applies RGB weights for grayscale.
    '''
    scale_factors = numpy.median(target_median, axis=(0, 1)) / numpy.median(v_median, axis=(0, 1))
    lut = numpy.asarray([
        (numpy.arange(256) * sf).clip(0, 255).astype(numpy.uint8)
        for sf in scale_factors
    ])
    if not apply_grayscale_weights:
        return lut
    return lut * RGB_WEIGHTS[:, numpy.newaxis]
