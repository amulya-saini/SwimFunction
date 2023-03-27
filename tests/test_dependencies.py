''' Make sure dependencies can be imported
without raising errors.
'''
import traceback
import warnings
import pytest

def test_library_dependencies():
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    failures = []
    try:
        import argparse
        import atexit
        from collections import namedtuple
        from functools import reduce
        import gc
        import glob
        import json
        import joblib
        import matplotlib
        import numba
        import numpy
        import pandas
        import pathlib
        import pickle
        import re
        import sklearn
        import shutil
        import subprocess
        import sys
        import time
        import umap
        import zplib
        print('Success importing library dependencies\n')
    except ModuleNotFoundError as e:
        print(traceback.format_exc())
        print(e)
        print('Please install all library dependencies.\n')
        failures.append(e)
    if len(failures) > 0:
        pytest.fail(f'{len(failures)} modules could not be imported: {failures}')

def test_optional_imports():
    ''' Test importing optional dependencies
    '''
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    failures = []
    try:
        from PyQt5 import Qt
        import elegant
        import ris_widget
        from swimfunction.pose_annotation.human_annotation import annotate_with_elegant
        from swimfunction.behavior_annotation.human_annotation import BehaviorAnnotator
        from swimfunction.pose_annotation.human_annotation import annotations_to_csv_h5
    except ModuleNotFoundError as e:
        print(traceback.format_exc())
        print(e)
        print('Please install all library dependencies.\n')
        failures.append(e)
    if len(failures) > 0:
        pytest.fail(f'{len(failures)} modules could not be imported: {failures}')

def test_internal_imports():
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    failures = []
    try:
        from swimfunction import FileLocations
        from swimfunction import progress
        from swimfunction.data_access import data_utils, fish_manager, \
            BehaviorAnnotationDataManager, PoseAccess
        from swimfunction.data_models import DotEnabledDict, Fish, PCAResult, SwimAssay
        from swimfunction.pose_annotation import extract_training_frames
        from swimfunction.pose_processing import pose_filters
        from swimfunction.recovery_metrics.metric_analyzers \
            import AbstractMetricAnalyzer, CruiseWaveformAnalyzer, \
                CruiseWaveformCalculator, ScoliosisAnalyzer, \
            MetricCalculator, PostureNoveltyAnalyzer, SwimCapacityProfiler
        from swimfunction.global_config.config import config
        from swimfunction.qc_scripts import behavior_qc, pose_qc
        from swimfunction.video_processing import extract_frames, get_median_frames, \
            standardize_video, CropTracker, fp_ffmpeg
        print('Success importing swimfunction specific scripts\n')
    except ModuleNotFoundError as e:
        print(traceback.format_exc())
        print(e)
        print('Please install swimfunction to your python path.')
        print('pip install /path/to/swimfunction')
        failures.append(e)
    if len(failures) > 0:
        pytest.fail(f'{len(failures)} modules could not be imported: {failures}')

if __name__ == '__main__':
    # test_dlc()
    test_library_dependencies()
    test_internal_imports()
