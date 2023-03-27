from swimfunction.data_models.SwimAssay import SwimAssay
from swimfunction.pose_processing import pose_filters
from swimfunction.pytest_utils import set_test_cache_access
from swimfunction.global_config.config import config

import numpy

set_test_cache_access()
FRAME_SIDE_BUFFER_WIDTH = config.getint('POSE FILTERING', 'frame_side_buffer_width')
FRAME_REAR_BUFFER_WIDTH = config.getint('POSE FILTERING', 'frame_rear_buffer_width')

COORDINATES = numpy.asarray([[2,3],[4,5],[numpy.nan,7],[8,9]])
LIKELIHOODS = numpy.asarray([[1,1],[0,0],[1,1],[0,1]])
SWIM_ASSAY = SwimAssay(vals_dict={
    'raw_coordinates': COORDINATES,
    'likelihoods': LIKELIHOODS,
    'smoothed_coordinates': COORDINATES,
    'was_cleaned_mask': [],
    'smoothed_complete_angles': [],
    'predicted_behaviors': [],
    'fish_name': 'F1',
    'assay_label': 1
})

NAN_FILTER_MASK = numpy.asarray([True, True, False, True], dtype=bool)
LIKELIHOOD_FILTER_MASK = numpy.asarray([True, False, True, False], dtype=bool)

def test_nan_filter():
    mask = pose_filters.filter_raw_coordinates_nan(SWIM_ASSAY)
    assert numpy.all(mask == NAN_FILTER_MASK)

def test_likelihood_filter():
    mask = pose_filters.filter_by_likelihood(SWIM_ASSAY)
    assert numpy.all(mask == LIKELIHOOD_FILTER_MASK)

def test_nan_and_likelihood_filter():
    mask = pose_filters.filter_by_likelihood(SWIM_ASSAY) & pose_filters.filter_raw_coordinates_nan(SWIM_ASSAY)
    target = NAN_FILTER_MASK & LIKELIHOOD_FILTER_MASK
    assert numpy.all(mask == target)

def test_filter_by_distance_from_edges_left_fish():
    ''' Fish cannot touch the sides or the 
    '''
    # Coordinates are [x, y] or similarly [along_the_width, top_to_bottom]
    ok_x = FRAME_SIDE_BUFFER_WIDTH + 1
    ok_y = 100
    good = [ok_x, ok_y]
    bad_left = [0, ok_y]
    bad_right = [100000, ok_y]
    bad_top = [ok_x, FRAME_REAR_BUFFER_WIDTH - 1]
    coordinates = numpy.asarray([
        # Good
        [good, good, good],
        [good, good, good],
        [good, good, good],
        [good, good, good],
        # Bad
        [bad_left, good, good],
        [good, bad_right, bad_right],
        [bad_top, good, bad_top]
    ])
    target = numpy.asarray([True, True, True, True, False, False, False], dtype=bool)
    swim_assay = SwimAssay(vals_dict={
        'raw_coordinates': coordinates,
        'likelihoods': [],
        'smoothed_coordinates': coordinates,
        'was_cleaned_mask': [],
        'smoothed_complete_angles': [],
        'predicted_behaviors': [],
        'fish_name': 'F1',
        'assay_label': 1
    })
    mask = pose_filters.filter_by_distance_from_edges(swim_assay)
    assert numpy.all(mask == target), 'Failed filter by distance from edges.'

def test_filter_by_distance_from_edges_right_fish():
    ok_x = 500
    ok_y = 100
    good = [ok_x, ok_y]
    good = [ok_x, ok_y]
    bad_left = [0, ok_y]
    bad_right = [100000, ok_y]
    bad_top = [ok_x, FRAME_REAR_BUFFER_WIDTH - 1]
    coordinates = numpy.asarray([
        # Good
        [good, good, good],
        [good, good, good],
        [good, good, good],
        [good, good, good],
        # Bad
        [bad_left, good, good],
        [good, bad_right, bad_right],
        [bad_top, good, bad_top]
    ])
    target = numpy.asarray([True, True, True, True, False, False, False], dtype=bool)
    swim_assay = SwimAssay(vals_dict={
        'raw_coordinates': coordinates,
        'likelihoods': [],
        'smoothed_coordinates': coordinates,
        'was_cleaned_mask': [],
        'smoothed_complete_angles': [],
        'predicted_behaviors': [],
        'fish_name': 'M1',
        'assay_label': 1
    })
    mask = pose_filters.filter_by_distance_from_edges(swim_assay)
    assert numpy.all(mask == target), 'Failed filter by distance from edges.'

if __name__ == '__main__':
    test_filter_by_distance_from_edges_right_fish()
    test_nan_and_likelihood_filter()
    test_nan_filter()
    test_likelihood_filter()
