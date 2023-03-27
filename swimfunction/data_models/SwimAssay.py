import traceback
import numpy

from swimfunction.data_access.data_utils import AnnotationTypes
from swimfunction.global_config.config import config

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
MAX_CAUDAL_FIN_LENGTH = config.getfloat('POSE FILTERING', 'max_caudal_fin_length')
MIN_LIKELIHOOD_THRESHOLD = config.getfloat('POSE FILTERING', 'min_likelihood_threshold')

def prebaked_caudal_fin_filters(coords: numpy.ndarray, caudal_fin_likelihoods: numpy.ndarray):
    ''' Filter by max fin length and DLC likelihood

    Parameters
    ----------
    coords: numpy.ndarray
        smoothed_coordinates_with_caudal
    caudal_fin_likelihoods: numpy.ndarray
    '''
    coords[
        numpy.logical_or(
            numpy.linalg.norm(coords[:, -2, :] - coords[:, -1, :], axis=1) > MAX_CAUDAL_FIN_LENGTH,
            (caudal_fin_likelihoods < MIN_LIKELIHOOD_THRESHOLD).flatten()
        ), ...
    ] = numpy.nan
    return coords

class SwimAssayMemory:
    def __init__(self, size=10):
        self.memory_size = size
        self.memory = {}
    def smoothed_coordinates_with_caudal(self, swim_assay):
        key = f'{swim_assay.fish_name}{swim_assay.assay_label}'
        if key in self.memory:
            return self.memory[key]
        if len(self.memory) > self.memory_size:
            self.memory.popitem()
        coords = numpy.asarray([])
        try:
            if swim_assay.caudal_fin_coordinates is not None and swim_assay.caudal_fin_coordinates.size:
                coords = prebaked_caudal_fin_filters(numpy.concatenate((
                    swim_assay.smoothed_coordinates,
                    swim_assay.caudal_fin_coordinates.reshape((swim_assay.smoothed_coordinates.shape[0], 1, swim_assay.smoothed_coordinates.shape[2]))
                ), axis=1), swim_assay.caudal_fin_likelihoods)
        except ValueError as e:
            print(traceback.format_exc())
            print(f'{swim_assay.fish_name} {swim_assay.assay_label} should be re-cached. Smoothed coordinates do not match with caudal fin coordinates.')
        self.memory[key] = coords
        return self.memory[key]

SWIM_ASSAY_MEMORY = SwimAssayMemory()

class SwimAssay:
    ''' SwimAssay holds swim related content.
    (A Fish has SwimAssays, a SwimAssay does not have a Fish, only the fish's name.)

    Note that smoothed_complete_angles has format [headX, headY, headAngle, a1, a2, ..., aN ]
    '''
    __slots__ = [
        'raw_coordinates',
        'likelihoods',
        'smoothed_coordinates',
        'was_cleaned_mask',
        'smoothed_complete_angles',
        'predicted_behaviors',
        'caudal_fin_coordinates',
        'caudal_fin_likelihoods',
        'fish_name',
        'assay_label']

    def __init__(self, vals_dict: dict=None, fish_name: str=None, assay_label: int=None):
        self.raw_coordinates = []
        self.likelihoods = []
        self.smoothed_coordinates = []
        self.was_cleaned_mask = []
        self.smoothed_complete_angles = []
        self.predicted_behaviors = []
        self.caudal_fin_coordinates = []
        self.caudal_fin_likelihoods = []
        self.fish_name = fish_name
        self.assay_label = assay_label
        if vals_dict is not None:
            self.from_dict(vals_dict)

    def __str__(self):
        return f'{self.fish_name} {self.assay_label}'

    @property
    def smoothed_angles(self):
        from swimfunction.pose_processing import pose_conversion
        return pose_conversion.complete_angles_poses_to_angles_pose(self.smoothed_complete_angles)

    @property
    def smoothed_coordinates_with_caudal(self):
        ''' Coordinate pose including tip of caudal fin
        '''
        return SWIM_ASSAY_MEMORY.smoothed_coordinates_with_caudal(self)

    ''' Load and Save functions
    '''
    def as_dict(self):
        ''' Returns the standard python dictionary version of self.
        '''
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def from_dict(self, d: dict):
        ''' Clears then sets its keys/values from a dictionary.
        Normally you'd want to do this at initialization, but you can
        call this function if you want to load after creating the object.
        '''
        for k, v in d.items():
            if k in self.__slots__:
                self.__setattr__(k, v)

    def get_numeric_behaviors(self, predicted=False):
        ''' Returns behavior array as integers, indices into BEHAVIORS [1, 0, ...]
        '''
        from swimfunction.data_access.PoseAccess import get_behaviors
        annotation_type = AnnotationTypes.predicted if predicted else AnnotationTypes.human
        swim_behaviors = get_behaviors(self, annotation_type)
        rv = numpy.full_like(swim_behaviors, BEHAVIORS.index(BEHAVIORS.unknown), dtype=numpy.int8)
        for i, b in enumerate(BEHAVIORS):
            if b == BEHAVIORS.unknown:
                continue
            if b in swim_behaviors:
                rv[numpy.where(swim_behaviors == b)] = i
        return rv
