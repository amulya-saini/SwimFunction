from matplotlib import pyplot as plt

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import PoseAccess
from swimfunction.data_models import Fish
from swimfunction.recovery_metrics.metric_analyzers.SwimCapacityProfiler\
    import SwimCapacityProfiler
from swimfunction.recovery_metrics.metric_analyzers \
      import swim_capacity_calculations as SCC
from swimfunction.global_config.config import config
from swimfunction.pytest_utils import set_test_cache_access, assert_equal_lists

import numpy

NO_FLOW = config.getint('FLOW', 'none')

CROP_WIDTH = config.getint('VIDEO', 'crop_width')
CROP_HEIGHT = config.getint('VIDEO', 'crop_height')
SCALE_DIVISOR = config.getint('VIDEO', 'scale_divisor')
WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')

_WIDTH = CROP_WIDTH//SCALE_DIVISOR
_HEIGHT = CROP_HEIGHT//SCALE_DIVISOR

set_test_cache_access()

FISH_NAMES = FDM.get_available_fish_names()
ENFORCE_SAME_LENGTH = False

IS_MAIN = __name__ == '__main__'

DUMMY_PROPS = SCC.AssayProperties(height=_HEIGHT, width=_WIDTH, is_resized=True)

def get_valid_assays(fish_name):
    return Fish.Fish(name=fish_name).load().swim_keys()

def test_get_distance():
    ''' Makes sure the distance from _get_distance matches the
    slower but correct implementation in this function.
    '''
    xs = numpy.asarray([7, 36, 59, 60, 15])
    ys = numpy.asarray([87, 8, 56, 44, 0])
    ps = numpy.array([xs, ys]).T
    distances = []
    for i in range(1, len(ps)):
        distances.append(numpy.linalg.norm(ps[i-1] - ps[i]))
    pred = SCC.distance_swum(
        numpy.asarray([xs, ys]).T,
        NO_FLOW,
        adjust_for_flow=True,
        assay_properties=DUMMY_PROPS,
        use_exhaustion_boundary=True
    )
    assert_equal_lists(distances, pred)

def test_distance_swum():
    pose_arrays, names, labels = PoseAccess.get_feature(
        FISH_NAMES, WPI_OPTIONS, feature='smoothed_coordinates')
    metric_profiler = SwimCapacityProfiler(
        pose_arrays, ENFORCE_SAME_LENGTH, names, labels, assay_properties=DUMMY_PROPS)
    distances = metric_profiler.distance_swum(SCC.MetricOptions.Centroid)
    if IS_MAIN:
        _fig, axs = plt.subplots(1, 2)
        axs[0].set_title('Distances swum (raw)')
        for dist, name, label in zip(distances, names, labels):
            axs[0].scatter(label, dist, label=f'{name}_{label}wpi')
        plt.legend()
        plt.tight_layout()
        plt.show()

def test_percent_active():
    pose_arrays, names, labels = PoseAccess.get_feature(
        FISH_NAMES, WPI_OPTIONS, feature='smoothed_coordinates')
    metric_profiler = SwimCapacityProfiler(
        pose_arrays, ENFORCE_SAME_LENGTH, names, labels, assay_properties=DUMMY_PROPS)
    percents = metric_profiler.percent_active(SCC.MetricOptions.Centroid)
    print(percents)
    if IS_MAIN:
        _fig, axs = plt.subplots(1, 2)
        axs[0].set_title('Percent of frames active (raw)')
        for p, name, label in zip(percents, names, labels):
            axs[0].scatter(label, p*100, label=f'{name}_{label}wpi')
        plt.legend()
        plt.tight_layout()
        plt.show()

def test_location_heatmap():
    for name in FISH_NAMES:
        for label in get_valid_assays(name):
            pose_arrays, names, labels = PoseAccess.get_feature(
                [name], [label], feature='smoothed_coordinates')
            metric_profiler = SwimCapacityProfiler(
                pose_arrays, ENFORCE_SAME_LENGTH, names, labels, assay_properties=DUMMY_PROPS)
            hms = metric_profiler.location_heatmap()
            assert len(hms) == 1
            assert names[0] == name and len(names) == 1
            assert labels[0] == label and len(labels) == 1

def test_x_position_all_data():
    pose_arrays, names, labels = PoseAccess.get_feature(
        FISH_NAMES, WPI_OPTIONS, feature='smoothed_coordinates')
    metric_profiler = SwimCapacityProfiler(
        pose_arrays, ENFORCE_SAME_LENGTH, names, labels, assay_properties=DUMMY_PROPS)
    pos_matrix = metric_profiler.x_position()
    if IS_MAIN:
        plt.figure()
        plt.suptitle('X position')
        for fish_dat, name, label in zip(pos_matrix, names, labels):
            plt.plot(
                fish_dat, range(len(fish_dat)),
                label=f'{name}_{label}wpi', linewidth=1, alpha=0.5)
        plt.legend()
        plt.show()

def test_y_position_all_data():
    pose_arrays, names, labels = PoseAccess.get_feature(
        FISH_NAMES, WPI_OPTIONS, feature='smoothed_coordinates')
    metric_profiler = SwimCapacityProfiler(
        pose_arrays, ENFORCE_SAME_LENGTH, names, labels, assay_properties=DUMMY_PROPS)
    pos_matrix = metric_profiler.y_position()
    if IS_MAIN:
        plt.figure()
        plt.suptitle('Y position')
        for fish_dat, name, label in zip(pos_matrix, names, labels):
            plt.plot(
                range(len(fish_dat)), fish_dat,
                label=f'{name}_{label}wpi', linewidth=1, alpha=0.5)
        plt.legend()
        plt.show()

def test_y_position():
    for name in FISH_NAMES:
        for label in get_valid_assays(name):
            pose_arrays, names, labels = PoseAccess.get_feature(
                [name], [label], feature='smoothed_coordinates')
            metric_profiler = SwimCapacityProfiler(
                pose_arrays, ENFORCE_SAME_LENGTH, names, labels, assay_properties=DUMMY_PROPS)
            pos_matrix = metric_profiler.y_position()
            assert names[0] == name and len(names) == 1
            assert labels[0] == label and len(labels) == 1
            if IS_MAIN:
                plt.figure()
                plt.suptitle('Y position')
                for i, fish_dat in enumerate(pos_matrix):
                    plt.plot(range(len(fish_dat)), fish_dat, label=i, linewidth=1, alpha=0.5)
                raw = PoseAccess.get_feature([name], [label], feature='raw_coordinates').features[0]
                raw_ys = SwimCapacityProfiler(
                    [raw], enforce_same_length=ENFORCE_SAME_LENGTH,
                    fish_names=[name], labels=[label], assay_properties=DUMMY_PROPS).y_position()[0]
                cleaned_mask = PoseAccess.get_feature(
                    [name], [label], feature='was_cleaned_mask').features[0]
                plt.scatter(
                    numpy.where(cleaned_mask)[0],
                    pos_matrix[0][numpy.where(cleaned_mask)[0]], c='green', s=2)
                plt.scatter(
                    numpy.where(cleaned_mask)[0],
                    raw_ys[numpy.where(cleaned_mask)[0]], c='red', s=2)
                plt.show()

def test_buffer_to_length_bools():
    full_length = SCC.EXPECTED_FINAL_FRAME
    arrs = [
        numpy.ones(10, dtype=bool),
        numpy.ones(full_length-100, dtype=bool),
        numpy.ones(full_length-1, dtype=bool),
        numpy.ones(full_length, dtype=bool),
        numpy.ones(full_length+1, dtype=bool),
        numpy.ones(full_length+100, dtype=bool)
    ]
    res = SCC.buffer_lists_to_full_length(arrs, buffer_val=False)
    for a1, a2 in zip(arrs, res):
        assert len(a2) == full_length, 'length is incorrect'
        if a1.size < full_length:
            assert a1.sum() == a2.sum(), 'Did not copy the correct information'
        else:
            assert a2.sum() == full_length, 'Did not copy the full content'

def test_buffer_to_length_floats():
    full_length = SCC.EXPECTED_FINAL_FRAME
    arrs = [
        numpy.arange(10, dtype=float),
        numpy.arange(full_length-100, dtype=float),
        numpy.arange(full_length-1, dtype=float),
        numpy.arange(full_length, dtype=float),
        numpy.arange(full_length+1, dtype=float),
        numpy.arange(full_length+100, dtype=float)
    ]
    res = SCC.buffer_lists_to_full_length(arrs, buffer_val=numpy.nan)
    for a1, a2 in zip(arrs, res):
        assert len(a2) == full_length, 'length is incorrect'
        if a1.size < full_length:
            assert numpy.all(numpy.isclose(a1, a2[:a1.size])), 'Did not buffer correctly'
        else:
            assert numpy.all(numpy.isclose(a1[:full_length], a2)), 'Did not trim correctly'

def test_pose_orientations():
    o_against = SCC.ORIENTED_AGAINST
    o_with = SCC.ORIENTED_WITH
    o_nagainst = SCC.NOT_QUITE_AGAINST
    o_nwith = SCC.NOT_QUITE_WITH
    def _test_o(orientation, angle_low, angle_high):
        angles = numpy.linspace(numpy.deg2rad(angle_low), numpy.deg2rad(angle_high), 100)
        noses = numpy.asarray([numpy.cos(angles), numpy.sin(angles)]).T.reshape((100, 1, 2))
        pa = numpy.concatenate((noses, numpy.zeros_like(noses)), axis=1)
        true_o = numpy.full(angles.shape[0], orientation)
        orientations = SCC.pose_orientations(pa)
        if not numpy.all(numpy.isclose(true_o, orientations)):
            if IS_MAIN:
                # Plot issues
                from matplotlib import pyplot as plt
                o_to_color = {
                    o_against: 'blue',
                    o_with: 'fuchsia',
                    o_nagainst: 'lightskyblue',
                    o_nwith: 'thistle'
                }
                fig, ax = plt.subplots()
                for p, o in zip(pa, orientations):
                    ax.plot(p[:, 0], p[:, 1], color=o_to_color[o])
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                fig.tight_layout()
                plt.show()
            msg = f'Failed for orientation {orientation} angles {angle_low} to {angle_high}'
            print(msg)
            assert False, msg
    _test_o(o_against, 90 - 44, 90 + 44)
    _test_o(o_with, 270 - 44, 270 + 44)
    _test_o(o_nagainst, 1, 44)
    _test_o(o_nagainst, 90 + 46, 179)
    _test_o(o_nwith, 181, 270 - 46)
    _test_o(o_nwith, 270 + 46, 359)

if IS_MAIN:
    test_pose_orientations()
    # test_get_distance()
    # test_distance_swum()
    # test_percent_active()
    # test_x_position_all_data()
    # test_y_position_all_data()
    # test_y_position()
    # test_location_heatmap()
