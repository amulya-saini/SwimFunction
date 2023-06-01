'''
Utility functions that should be shared with many scripts.
NOTE: lightweight internal imports occur within functions to avoid circular import errors.
'''

from collections import namedtuple
from typing import List

import numpy
import pathlib
import re

from swimfunction.context_managers.CacheContext import CacheContext
from swimfunction.global_config.config import config
from swimfunction.data_models.PCAResult import PCAResult

# name identifies the fish.
# side informs how to crop the video (can be None if not found).
# assay_label identifies the assay.
_FishDetails = namedtuple('FishDetails', ['name', 'side', 'assay_label'])

GroupNumSide = namedtuple('GroupNumSide', ['group', 'num', 'side'])

SLOW_FLOW = config.getint('FLOW', 'slow')
FAST_FLOW = config.getint('FLOW', 'fast')
HAS_FLOW = config.getint('FLOW', 'has_flow')

class AnnotationTypes:
    predicted = 'P'
    human = 'H'
    all = 'A'

class FishDetails(_FishDetails):
    def __eq__(self, fd2):
        return self.name == fd2.name and self.assay_label == fd2.assay_label

    def matches(self, name, assay_label):
        return name == self.name and assay_label == self.assay_label

def fish_name_to_group_num_side(piece: str) -> GroupNumSide:
    ''' Parses string into group, number, and (optional) side.
    {group}{number}{side(optional)}
    Note: will be returned in upper-case.
    '''
    piece = piece.upper()
    gns = GroupNumSide(None, None, None)
    m = re.match(r'([A-Z]+)(\d+)([LR]?)', piece)
    if m:
        gns = GroupNumSide(
            piece[m.regs[1][0]:m.regs[1][1]],
            int(piece[m.regs[2][0]:m.regs[2][1]]),
            piece[m.regs[3][0]:m.regs[3][1]])
    return gns

def fish_name_to_group(fish_name: str) -> str:
    ''' Parses the group from the fish name.
    Note: fish names are expected to be {group}{number}
    Note: will be returned in upper-case.
    '''
    gns = fish_name_to_group_num_side(fish_name)
    if gns is gns.group is None or gns.num is None:
        return None
    return gns.group

def _match_assay_label(piece: str) -> int:
    ''' Get the assay label from the string,
    or return None if the string is not an assay label.
    '''
    preinjury_assay = config.getint('EXPERIMENT DETAILS', 'control_assay_label')
    assay_label = None
    if 'DPI' in piece or 'WPI' in piece or 'PREINJ' in piece and assay_label is None:
        assay_label = preinjury_assay if 'PREINJ' in piece else int(piece.replace('WPI', '').replace('DPI', ''))
    if 'TREATMENT' in piece and assay_label is None:
        assay_label = -1 if 'PRE' in piece else 1 if 'POST' in piece else None
    return assay_label

def parse_details_from_filename(fname) -> List[FishDetails]:
    ''' Returns array of FishDetails namedtuples.
    Every fish detected in the filename gets a FishDetails tuple
    Every returned FishDetails has a non-empty fish name.

    Expectations:
        Basically it's like this:
            [anything]_[assay label]_[fish names separated by underscore].[everything here ignored]
        Everything is converted to uppercase.
        Parts are separated by underscores '_'
        Anything can go before the assay label as long as it does not match an assay label
            assay label can be '[number]wpi' or ['preinj', 'pre-inj'] or ['pretreatment', 'posttreatment'] exactly once
        Every part after the assay label must include a fish name as {GroupString}{Number}{Side} where {side} is optional.
    Valid filename examples:
        3wpi_M23.avi
        preinj_M23R_F43L.avi
        Pre-inj_M23R_F43L.avi
        1-7-2020_17491_EKAB_1wpi_F3L_M3R.avi
        2022_0223_22261_ntrl1_pretreatment_RPFW7.avi
    Invalid filename examples:
        M23_3wpi.avi # Assay label must come before fish name.
    '''
    list_of_details = []
    fname = pathlib.Path(fname).name.upper().replace(' ', '_')
    assay_label = None
    basename = fname
    if '.' in fname: # remove extensions
        basename = fname.split('.')[0]
    if 'DLC_RESNET' in basename: # remove DeepLabCut extra stuff if it exists
        basename = basename.split('DLC_RESNET')[0]
    if 'PRE-INJ' in basename:
        # remove hyphen between pre-inj to make every preinjury say preinj
        basename = basename.replace('-', '')
    for piece in basename.split('_'):
        if assay_label is None:
            assay_label = _match_assay_label(piece)
        elif assay_label is not None:
            gns = fish_name_to_group_num_side(piece)
            if gns:
                if gns.group is None or gns.num is None:
                    # An invalid name was encountered.
                    # Don't trust the parsing.
                    # The user likely did not check their filenames.
                    list_of_details = []
                    break
                fish_name = f'{gns.group}{gns.num}'
                list_of_details.append(FishDetails(name=fish_name, side=gns.side, assay_label=assay_label))
    return list_of_details

def calculate_pca(
        group: str=None,
        feature: str='smoothed_angles',
        assay_label: int=None,
        force_recalculate: bool=False,
        verbose: bool=False) -> PCAResult:
    ''' By default, will return PCA of all poses (filtered using BASIC_FILTERS)

    Parameters
    ----------
    group : str
        Fish group for which to calculate PCA result. Default is all fish (group=None)
    feature : str
    assay_label : int, default=None
        Which assay label (if not None) to use for the PCA.
    force_recalculate : bool
        Whether to perform full calculation regardless of cache existance.
        The newly calculated PCA will overwrite the existing cache.
    verbose : bool
    '''
    # To avoid circular import errors, we import within this function.
    from swimfunction.data_access.fish_manager import DataManager as FDM
    from swimfunction.data_access.PoseAccess import get_feature
    from swimfunction.pose_processing.pose_filters import BASIC_FILTERS
    from swimfunction.FileLocations import get_local_pca_cache
    groupstr = group if group is not None else 'all_groups'
    assay_str = f'_{assay_label}' if assay_label is not None else ''
    cache_path = get_local_pca_cache() / f'{groupstr}{assay_str}_{feature}_pca.pickle'
    names = None
    if group is None:
        names = FDM.get_available_fish_names()
    else:
        names = FDM.get_available_fish_names_by_group()[group]
    with CacheContext(cache_path) as cache:
        if cache.getContents() is not None and not force_recalculate:
            return PCAResult(cache.getContents())
        poses = numpy.concatenate(
            get_feature(
                fish_names=names,
                assay_labels=[assay_label] if assay_label is not None else None,
                feature=feature,
                filters=BASIC_FILTERS,
                keep_shape=False).features)
        pca_result = PCAResult().pca(poses, verbose=verbose)
        cache.saveContents(pca_result.as_dict())
    return pca_result

def get_tracking_experiment_pca() -> PCAResult:
    from swimfunction.FileLocations import GlobalFishResourcePaths
    with CacheContext(GlobalFishResourcePaths.global_pca) as cache:
        tracking_experiment_pca_result = PCAResult(cache.getContents())
    return tracking_experiment_pca_result
