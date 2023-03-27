''' Helpers for processing images.
No swimfunction imports allowed.
(Avoid circular imports.)
'''
from scipy import ndimage
import numpy

def threshold(frame: numpy.ndarray, th: float=100) -> numpy.ndarray:
    ''' Binarize a frame by thresholding
    Parameters
    ----------
    frame: numpy.ndarray
    th: float
        Threshold for binarizing the frame.
    '''
    if frame is None:
        return frame
    img = frame
    if len(frame.shape) == 3:
        img = frame.max(axis=2)
    mask = numpy.ones(frame.shape[:2], dtype=bool)
    mask[img > th] = 0
    return mask

def isolate_object_in_mask(mask):
    ''' Labels masked objects. Largest is background, next largest is fish.
    '''
    if mask is None or mask.max() == 0:
        return None
    objs, _labels = ndimage.label(mask, structure=numpy.ones(9).reshape((3, 3)))
    if _labels < 1:
        # Nothing found. Return None.
        return None
    mask[objs != numpy.argsort(numpy.bincount(objs.ravel()))[-2]] = 0
    return mask
