'''
 Loads and saves fish swim assay data, csv to cache or directly on the cache.
If __main__, then converts csv files to cache and calculates pose representations.
'''

from typing import List
from collections import defaultdict
import functools
import pathlib
import threading
import time
import traceback
import warnings
import numpy
import pandas
from tqdm import tqdm

from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
from swimfunction.data_access import data_utils
from swimfunction.context_managers.CacheContext import CacheContext
from swimfunction.context_managers.AccessContext import AccessContext
from swimfunction.pose_processing import pose_conversion
from swimfunction.pose_processing.pose_cleaners.SplineSmoother \
    import SplineSmootherVideoAvailable, SplineSmootherNoVideoAvailable
from swimfunction.pose_processing.pose_cleaners.TeleportationCleaner \
    import TeleportationCleaner
from swimfunction.global_config.config import config
from swimfunction.video_processing import CropTracker, fp_ffmpeg
from swimfunction.video_processing.extract_frames import Extractor
from swimfunction.recovery_metrics.metric_analyzers\
    .PrecalculatedMetricsAnalyzer import PrecalculatedMetricsAnalyzer

from swimfunction import loggers

from swimfunction import FileLocations
from swimfunction import progress

POINTS_PER_POSE = config.getint('POSES', 'points_per_pose')
NO_FLOW = config.getint('FLOW', 'none')

Y_CLEANER = TeleportationCleaner(40)
X_CLEANER = TeleportationCleaner(20)
XY_CLEANER = TeleportationCleaner(45)

class FishWarning(UserWarning):
    ''' Named warning specific for fish.
    '''

class _FishManager:
    ''' Manages essential fish swim assay data,
    from DeepLabCut (DLC) output to human-assigned scores.
    '''
    DATA_ACCESS_LOCK = threading.Lock()

    #    ------ Writing DLC Annotations To Cache ------
    def create_cache(self, force_reload: bool=False):
        ''' Load all DLC output files into cached Fish objects.
        '''
        logger = loggers.get_dlc_import_logger(__name__)
        logger.info('Loading fish')
        start = time.time()
        self.cache_all_annotation_files(force_reload)
        self.create_available_assays_csv()
        logger.info('Done importing DLC files in %d seconds', int(time.time()-start))

    def cache_all_annotation_files(self, force_reload=False):
        ''' Prefer to load combined annotation files (body and caudal fin)
        otherwise load them separately.
        '''
        fish_assay_pairs = self.cache_all_combined_annotation_files(force_reload)
        self.cache_all_centerline_pose_files(fish_assay_pairs, force_reload)
        self.cache_all_caudal_fin_pose_files(fish_assay_pairs, force_reload)

    def cache_all_caudal_fin_pose_files(self, fish_assay_pairs_to_ignore, force_reload=False):
        ''' Get the caudal fin pose file for all fish
        that already have cached centerlines.

        This, unfortunately, cannot be run in parallel (via WorkerSwarm).
        '''
        logger = loggers.get_dlc_import_logger(__name__)
        logger.info('Loading any caudal fin pose files...')

        for name in tqdm(self.get_available_fish_names()):
            fish_data = self.load_fish_data(name)
            changed = False
            for assay_label in fish_data:
                if (name, assay_label) in fish_assay_pairs_to_ignore:
                    continue
                try:
                    if not force_reload and ('caudal_fin_likelihoods' in fish_data[assay_label] \
                            and fish_data[assay_label]['caudal_fin_likelihoods'] is not None \
                            and len(fish_data[assay_label]['caudal_fin_likelihoods']) > 0):
                        continue
                    fin_coords, fin_likelihoods = self.read_caudal_fin_pose_file(
                        name, assay_label)
                    if fin_coords is None:
                        logger.warning('No caudal fin file found for %s %d wpi', name, assay_label)
                        continue
                    fish_data[assay_label]['caudal_fin_coordinates'] = fin_coords
                    fish_data[assay_label]['caudal_fin_likelihoods'] = fin_likelihoods
                    changed = True
                except Exception as e:
                    logger.error(traceback.format_exc())
                    logger.error(
                        'Cannot get caudal fin data for %s %dwpi due to %s',
                        name, assay_label, str(e))
            if changed:
                self.save_fish_data(name, fish_data)

    def cache_all_centerline_pose_files(self, fish_assay_pairs_to_ignore, force_reload=False):
        ''' By default, only loads fish that are not already cached.

        Parameters
        ----------
        force_reload : bool
            whether to reload all annotation files,
            over-writing any existing cache. Default: False
        '''
        logger = loggers.get_dlc_import_logger(__name__)
        logger.info('Loading any centerline pose files...')
        fpaths = self.find_all_dlc_annotation_files(FileLocations.get_dlc_body_outputs_dir())
        final_fpaths = []
        for fpath in fpaths:
            fd = data_utils.parse_details_from_filename(fpath.name)[0]
            if (fd.name, fd.assay_label) in fish_assay_pairs_to_ignore:
                continue
            final_fpaths.append(fpath)
        self._read_single_animal_files(final_fpaths, force_reload)

    def cache_all_combined_annotation_files(self, force_reload=False):
        ''' Load annotation files located in the default directory.
        '''
        annot_files = self.find_all_dlc_annotation_files(FileLocations.get_dlc_outputs_dir())
        file_details = [data_utils.parse_details_from_filename(f)[0] for f in annot_files]
        fish_assay_pairs = [(fd.name, fd.assay_label) for fd in file_details]
        logger = loggers.get_dlc_import_logger(__name__)
        logger.info('Loading pose annotation files...')
        self._read_single_animal_files(annot_files, force_reload)
        return fish_assay_pairs

    @staticmethod
    def find_all_dlc_annotation_files(rootdir: pathlib.Path):
        ''' Locate all annotation files in the directory.
        Prefer .h5, but also allow .csv
        '''
        # Prefer h5
        fpaths = list(rootdir.glob('*.h5'))
        fds = [data_utils.parse_details_from_filename(f.name)[0] for f in fpaths]
        for fpath in rootdir.glob('*.csv'):
            # Allow csv
            fd = data_utils.parse_details_from_filename(fpath.name)[0]
            if not numpy.any([fd == fd2 for fd2 in fds]):
                fpaths.append(fpath)
        return fpaths

    @staticmethod
    def read_caudal_fin_pose_file(fish_name, assay_label, dlc_save_root=None):
        ''' Get the caudal fin coordinates for a fish and assay.

        Returns
        -------
        (raw_coordinates: numpy.ndarray, likelihoods: numpy.ndarray)
        '''
        if dlc_save_root is None:
            dlc_save_root = FileLocations.get_dlc_caudal_fin_outputs_dir()
        if not dlc_save_root.exists():
            return None, None
        fin_file_matches = list(filter(
            lambda f: data_utils.parse_details_from_filename(f)[0].matches(fish_name, assay_label),
            dlc_save_root.glob('*.h5')))
        if not fin_file_matches:
            return None, None
        with CacheContext(fin_file_matches[0]) as cache:
            dlc_df = cache.getContents()
        raw_coordinates, likelihoods = _FishManager._dlc_output_to_coordinates_likelihoods(dlc_df)
        return raw_coordinates, likelihoods

    def _read_single_animal_files(
            self,
            fnames: list,
            force_reload):
        ''' Handles annotation files that either have centerline with caudal tail tip
        or just the centerline without the caudal tail tip.

        This, unfortunately, cannot be run in parallel (via WorkerSwarm).
        '''
        if not fnames:
            return

        name_to_file_map = self._get_fish_to_file_list_dict(fnames)
        with progress.Progress(sum([len(v) for v in name_to_file_map.values()])):
            for fish_name, vlist in name_to_file_map.items():
                existing_cache = self._get_fish_cache_name(fish_name)
                fish_data = self.load_fish_data(fish_name) if existing_cache.exists() else {}
                changed = False
                for fname in vlist:
                    progress.increment()
                    assay_label = data_utils.parse_details_from_filename(fname)[0].assay_label
                    if assay_label not in fish_data or force_reload:
                        swim_data = self._read_dlc_output_file(fname)
                        if swim_data is not None:
                            fish_data[assay_label] = swim_data
                            changed = True
                if changed:
                    self.save_fish_data(fish_name, fish_data)

    @staticmethod
    def _read_dlc_output_file(fpath):
        ''' Converts DLC output into dicts of posture data.
        If crop tracker log exists, handles appropriately.

        Annotation files can either have centerline with caudal tail tip
        or just the centerline without the caudal tail tip.

        Returns
        ------
        swim_data : dict
            raw_coordinates
                unfiltered coordinates from the file in the full video space
            likelihoods
                likelihoods from the file
            smoothed_coordinates
                coordinates resampled smoother and TeleportationCleaner cleaned
            smoothed_complete_angles
                complete angle representation of smoothed_coordinates
            caudal_fin_coordinates
                coordinates from the file or None
                May be supplied in later analysis
            caudal_fin_likelihoods
                coordinates from the file or None
                May be supplied in later analysis
            behaviors: None
                This will be supplied in later analysis
            predicted_behaviors: None
                This will be supplied in later analysis
        '''
        fpath = FileLocations.as_absolute_path(fpath)
        raw_coordinates, likelihoods = _FishManager._dlc_output_to_coordinates_likelihoods(fpath)
        caudal_fin_coordinates = None
        caudal_fin_likelihoods = None
        if raw_coordinates.shape[1] == 10:
            raw_body_coordinates = raw_coordinates
            body_likelihoods = likelihoods
        elif raw_coordinates.shape[1] == 11:
            raw_body_coordinates = raw_coordinates[:, :10, :]
            body_likelihoods = likelihoods[:, :10]
            caudal_fin_coordinates = raw_coordinates[:, 10, :]
            caudal_fin_likelihoods = likelihoods[:, 10]
        else:
            raise RuntimeError(' '.join([
                'Keypoints in file do not match expectation:',
                '10 on body, optional caudal fin at the end.'
            ]))
        name = None
        assay_label = None
        logger = loggers.get_dlc_import_logger(__name__)
        fish_details = data_utils.parse_details_from_filename(fpath.name)[0]
        name = fish_details.name
        assay_label = fish_details.assay_label
        if not name:
            raise RuntimeError(' '.join([
                f'Could not parse fish name from {fpath.name}.']))
        logger.debug(fpath)
        logger.debug(name)
        spline_smoother = SplineSmootherNoVideoAvailable()
        possible_video = FileLocations.find_video(name, assay_label)
        video_available = possible_video is not None and possible_video.exists()
        if video_available:
            logger.debug(' '.join([
                'Video is available.',
                'Will smooth poses with splines then use skeleton fix on extreme results.'
            ]))
            spline_smoother = SplineSmootherVideoAvailable()
        else:
            logger.debug('No video found, so it will only smooth poses using splines.')
        smoothed_coordinates, was_cleaned_mask = XY_CLEANER.clean(spline_smoother.clean(
            poses=raw_body_coordinates,
            likelihoods=body_likelihoods,
            fish_name=name,
            assay_label=assay_label))
        numnan = numpy.where(numpy.all(
            numpy.isnan(smoothed_coordinates),
            axis=(1, 2)))[0].shape[0]
        logger.debug(
            '%d poses needed cleaning, has %d completely nan poses after cleaning, %d poses total',
            was_cleaned_mask.sum(),
            numnan,
            raw_body_coordinates.shape[0])
        complete_angles = pose_conversion.points_to_complete_angles(smoothed_coordinates)
        return {
            'raw_coordinates': raw_body_coordinates,
            'likelihoods': body_likelihoods,
            'smoothed_coordinates': smoothed_coordinates,
            'was_cleaned_mask': was_cleaned_mask,
            'smoothed_complete_angles': complete_angles,
            'caudal_fin_coordinates': caudal_fin_coordinates, # slot placeholder,
            'caudal_fin_likelihoods': caudal_fin_likelihoods, # slot placeholder,
            'behaviors': None, # slot placeholder,
            'predicted_behaviors': None # slot placeholder
        }

    @staticmethod
    def _dlc_to_coordinates_likelihoods_raw(dlc_df: pandas.DataFrame, individual: str):
        has_likelihoods = 'likelihood' in pandas.unique(
            [c[-1] for c in dlc_df.columns.values])
        # NOTE: indexing the DataFrame is in this format:
        #    dataframe.loc[frame_numbers, idx[individuals, bodyparts, x_y_likelihood]]
        # Then, use .values.tolist() for efficient transfer to numpy array (no copy)
        idx = pandas.IndexSlice
        num_points = dlc_df.shape[1] // 3
        raw_coordinates = numpy.empty((dlc_df.values.shape[0], num_points, 2))
        raw_coordinates[:, :, 0] = numpy.asarray(
            dlc_df.loc[:, idx[individual, :, 'x']].values.tolist())
        raw_coordinates[:, :, 1] = numpy.asarray(
            dlc_df.loc[:, idx[individual, :, 'y']].values.tolist())
        likelihoods = numpy.asarray(
            dlc_df.loc[:, idx[:, :, ['likelihood']]].values.tolist()
        ) if has_likelihoods else None
        return raw_coordinates, likelihoods

    @staticmethod
    def _choose_csv_header_arr(fpath: pathlib.Path) -> List[int]:
        ''' Gets the number of header lines in the csv file.
        The final header must begin with the word "coords".
        The resulting array is to be passed into the
        "header" argument of pandas.read_csv

        There can only be at most 4 header lines.
        '''
        found_final_line = False
        header_arr = []
        with open(fpath, 'rt') as fh:
            for i, line in enumerate(fh):
                header_arr.append(i)
                if len(line) > 6 and line[:6] == 'coords':
                    found_final_line = True
                    break
                if i > 3:
                    break
        if not found_final_line:
            header_arr = None
        return header_arr

    @staticmethod
    def _dlc_output_to_dataframe(fpath) -> pandas.DataFrame:
        ''' Reads an h5 or csv file, returns it as a pandas DataFrame.
        '''
        fpath = FileLocations.as_absolute_path(fpath)
        extension = fpath.name.split('.')[-1]
        dlc_df = pandas.DataFrame()
        if extension == 'csv':
            header = _FishManager._choose_csv_header_arr(fpath)
            dlc_df = pandas.read_csv(fpath.as_posix(), header=header, index_col=[0])
        elif extension == 'h5':
            store = pandas.HDFStore(fpath.as_posix(), mode='r')
            dlc_df = pandas.read_hdf(store)
            store.close()
        else:
            raise ValueError(f'Unable to determine file type (csv or h5) from {fpath.name}')
        if len(dlc_df.columns[0]) == 4:
            # If there are 4 columns, this is a multi-animal h5 file.
            # By indexing past the scorer, the first column unique values
            # correspond to individuals in this file.
            # This script can only handle single-animal files, anyway.
            dlc_df = dlc_df[dlc_df.columns[0][0]]
        return dlc_df

    @staticmethod
    def _dlc_output_to_coordinates_likelihoods(fpath: pathlib.Path):
        ''' Reads an h5 or csv file, returns the raw coordiantes and likelihoods
        If a CropTracker logfile is found, coordinates will be transposed
        back to the original video space.

        Note: If there are four header lines, it removes the top level header.
            That way, the first column values are unique to individual animals in the DataFrame.
            Only single animal annotation files are supported at this time.
        '''
        dlc_df = _FishManager._dlc_output_to_dataframe(fpath)
        raw_coordinates, likelihoods = _FishManager._dlc_to_coordinates_likelihoods_raw(
            dlc_df, _FishManager._get_individuals_from_dlc_df(dlc_df)[0]
        )
        fd = data_utils.parse_details_from_filename(fpath.name)[0]
        raw_coordinates, likelihoods = _FishManager._to_full_video_using_crop_tracker(
            raw_coordinates, likelihoods, FileLocations.find_video(fd.name, fd.assay_label))
        return raw_coordinates, likelihoods

    @staticmethod
    def _get_individuals_from_dlc_df(dlc_df: pandas.DataFrame):
        return pandas.unique([c[0] for c in dlc_df.columns.values])

    def _get_fish_to_file_list_dict(self, annotation_files):
        '''
        Returns
        -------
        csv_map : dict
            from fish name to list of files that match the fish name
        '''
        has_quality = FileLocations.get_qualitative_score_file().exists()
        csv_map = defaultdict(list)
        for fname in annotation_files:
            for fd in data_utils.parse_details_from_filename(fname):
                if has_quality and self.get_quality(fd.name, fd.assay_label) == 0:
                    # Skip dead fish, if we still accidentally processed an empty swim video.
                    continue
                csv_map[fd.name].append(fname)
        return csv_map

    @staticmethod
    def _to_full_video_using_crop_tracker(raw_coordinates, likelihoods, video_path: pathlib.Path):
        ''' If a crop tracker logfile exists, uses that logfile to expand DLC annotations
        to the full length of the video. Also expands likelihoods.

        Parameters
        ----------
        raw_coordinates : numpy.ndarray
            From DLC output, length of video (could be crop-tracked,
            and thus shorter than the original).
        likelihoods : numpy.ndarray
            From DLC output, length of video (could be crop-tracked,
            and thus shorter than the original).
        fpath : pathlib.Path

        Returns
        -------
        raw_coordinates : numpy.ndarray
            From DLC output, padded if necessary to be the length of the video.
        likelihoods : numpy.ndarray
            From DLC output, padded if necessary to be the length of the video.
        '''
        possible_log = CropTracker.find_existing_logfile(video_path)
        if possible_log is not None:
            transformed = CropTracker.CropTracker.full_video_coords_to_lab_frame(
                raw_coordinates, possible_log)
            new_likelihoods = numpy.full(transformed.shape[:2], numpy.nan)
            new_likelihoods[
                numpy.where(numpy.all(numpy.logical_not(numpy.isnan(transformed)), axis=(1, 2))), :
            ] = likelihoods
            return transformed, new_likelihoods
        return raw_coordinates, likelihoods

    #    ------ Other I/O ------

    def create_available_assays_csv(self, force_recalculate: bool=False):
        ''' Creates a file with available assays, based on what's saved in the fish objects.
        '''
        csv_file = FileLocations.get_fish_to_assays_csv()
        if csv_file.exists() and not force_recalculate:
            return
        print(f'Creating available assays file: {csv_file.as_posix()}')
        fish_to_assays = {}
        access_lock = threading.Lock()

        def get_assay_labels(fish_name):
            fish_dict = self.load_fish_data(fish_name)
            assay_labels = []
            for k, v in fish_dict.items():
                if isinstance(v, (dict)):
                    assay_labels.append(k)
            with access_lock:
                fish_to_assays[fish_name] = assay_labels
            progress.increment()
            return assay_labels

        with progress.Progress(len(self.get_available_fish_names())):
            with WorkerSwarm(loggers.get_metric_calculation_logger(__name__)) as swarm:
                for fish_name in self.get_available_fish_names():
                    swarm.add_task(lambda name=fish_name: get_assay_labels(name))

        with open(csv_file, 'wt') as fh:
            for fish, assays in fish_to_assays.items():
                assay_str = ','.join([str(x) for x in sorted(assays)])
                fh.write(f'{fish},{assay_str}\n')
        print(f'Done creating available assays file: {csv_file.as_posix()}')
        print('Please check over the above file and be sure its contents look accurate.')
        print('(Sometimes a blank video is accidentally imported.)')

    @staticmethod
    def save_fish_data(name, data):
        ''' Saves fish data to the cache

        Parameters
        ----------
        name : str
        data : dict
            dictionary of dictionaries
        '''
        with CacheContext(_FishManager._get_fish_cache_name(name)) as cache:
            cache.saveContents(data)

    @staticmethod
    def load_fish_data(name):
        ''' Loads fish data from the cache.

        Returns
        -------
            dictionary of information for a Fish
        '''
        swims_dict = None
        with CacheContext(_FishManager._get_fish_cache_name(name)) as cache:
            swims_dict = cache.getContents()
            if swims_dict is None:
                warnings.warn(f'Could not find data for fish {name}', FishWarning)
        return swims_dict

    #    ------ Check for Available Stuff ------
    @staticmethod
    def get_available_fish_names(experiment_name=None):
        ''' Helper to get the names of all available cached fish.
        '''
        if experiment_name is None:
            experiment_name = config.experiment_name
        with AccessContext(experiment_name=experiment_name):
            from swimfunction.FileLocations import get_fish_cache, get_normalized_videos_dir
            fish_cache = get_fish_cache()
            # Prefer fish names that have been collected in the cache.
            fish_names = list(set([
                p.name.split('.')[0]
                for p in fish_cache.glob('*.pickle')
            ]))
            # If none exist, instead use fish names that have videos.
            if not fish_names:
                fish_names = list(set([
                    data_utils.parse_details_from_filename(f)[0].name
                    for f in get_normalized_videos_dir().glob('*.avi')
                ]))
        return fish_names

    @staticmethod
    def get_available_fish_names_by_group() -> dict:
        '''
        Returns
        -------
        dict
            Key: group string, Value: list of names
        '''
        by_group = defaultdict(list)
        for name in _FishManager.get_available_fish_names():
            by_group[data_utils.fish_name_to_group(name)].append(name)
        return by_group

    def get_available_assay_labels(self, fish_name=None, experiment_name=None):
        ''' Get assay labels that exist for the fish.
        Uses contents of a csv file, not video presence.
        '''
        if experiment_name is None:
            experiment_name = config.experiment_name
        options = set()
        with AccessContext(experiment_name=experiment_name):
            csv_file = FileLocations.get_fish_to_assays_csv()
            if not csv_file.exists():
                self.create_available_assays_csv()
            with open(csv_file, 'rt') as fh:
                for line in fh:
                    line = line.strip().split(',')
                    assays = [int(x) for x in line[1:]]
                    if fish_name is not None and line[0] == fish_name:
                        options = options.union(assays)
                        break
                    elif fish_name is None:
                        options = options.union(assays)
        if not options:
            warnings.warn(f'Could not get any available assay labels from \
                {fish_name} in {experiment_name}.\
                Did you create the file yet? If not, try using create_available_assays_csv()')
        return sorted(list(options))

    @staticmethod
    def get_groups(experiment_name=None) -> list:
        ''' If experiment has fish 'F1' and 'M1', returns ['M', 'F'].
        '''
        return list(set([
            data_utils.fish_name_to_group(n)
            for n in _FishManager.get_available_fish_names(experiment_name)]))

    @staticmethod
    def get_crop_tracker_log(fish_name, assay_label) -> CropTracker.CropTrackerLog:
        ''' Get a CropTrackerLog for the given fish and assay
        '''
        video_path = FileLocations.find_video(fish_name, assay_label)
        logpath = CropTracker.find_existing_logfile(video_path)
        if logpath is None or not logpath.exists():
            return None
        return CropTracker.CropTrackerLog().read_from_file(logpath)

    #    ------ Useful Metric Access ------

    def get_scoliosis(self, fish_name, assay):
        ''' Scoliosis for a specific assay
        '''
        from swimfunction.data_access.MetricLogger import MetricLogger
        from swimfunction.recovery_metrics.metric_analyzers\
            .ScoliosisAnalyzer import ScoliosisAnalyzer
        analyzer = ScoliosisAnalyzer()
        if len(analyzer.keys_to_printable) != 1:
            raise RuntimeError(''.join([
                'Check that ScoliosisAnalyzer().keys_to_printable ',
                'only has one item. Otherwise, we need to hardcode ',
                'the correct metric\'s name here.']))
        metric = list(analyzer.keys_to_printable.keys())[0]
        if not MetricLogger.has_analyzer_metrics(analyzer) \
                or MetricLogger.get_metric_value(fish_name, assay, metric) is None:
            MetricLogger.log_analyzer_metrics(analyzer)
        return MetricLogger.get_metric_value(fish_name, assay, metric)

    def get_final_scoliosis(self, fish_name):
        ''' Scoliosis measured in the fish's last assay
        '''
        return self.get_scoliosis(fish_name, self.get_available_assay_labels(fish_name)[-1])

    def get_endurance_time(self, fish_name, assay_label) -> float:
        ''' Get endurance, meaning time to exhaustion, for the fish.
        Requires the "endurance.csv" file to exist.
        '''
        return PrecalculatedMetricsAnalyzer().get_value(
                fish_name, assay_label, key='endurance', missing_value=None)

    def get_quality(self, fish_name, assay_label):
        ''' Get human score of swim quality
        Requires "perceived_quality.csv" file to exist.
        '''
        return PrecalculatedMetricsAnalyzer().get_value(
            fish_name, assay_label, key='perceived_quality', missing_value=None)

    def get_highest_recovered_quality(self, fish_name: str, min_assay_label=5) -> int:
        '''Gets the highest human-assigned score in recovery.
        Sometimes the 8wpi score is lower than 7wpi (4 vs 5),
        so it selects the highest from 5wpi onward.

        If no scores can be found, returns None.

        Parameters
        ----------
        fish_name : str
            Name of fish

        Returns
        -------
        int
            highest recovery quality score (max of 5, 6, 7, 8 wpi)
        '''
        final_assay = max(self.get_available_assay_labels(fish_name))
        scores = list(filter(
            lambda x: x is not None,
            [
                self.get_quality(fish_name, assay_label)
                for assay_label in range(min_assay_label, final_assay+1)
            ]
        ))
        return max(scores) if scores else None

    def get_final_percent_bridging(self, fish_name, missing_val=None):
        ''' Get percent bridging for the last assay of the fish.
        Requires "glial_bridging.csv" file to exist.
        '''
        final_assay = max(self.get_available_assay_labels(fish_name))
        return PrecalculatedMetricsAnalyzer().get_value(
            fish_name,
            final_assay,
            key='glial_bridging',
            missing_value=missing_val)

    def get_final_percent_proximal_axon_recovery(self, fish_name, missing_val=None):
        ''' Get proximal axon regrowth for the last assay of the fish.
        Requires "proximal_axon_regrowth.csv" file to exist.
        '''
        final_assay = max(self.get_available_assay_labels(fish_name))
        return PrecalculatedMetricsAnalyzer().get_value(
            fish_name, final_assay, key='proximal_axon_regrowth', missing_value=missing_val)

    def get_final_percent_distal_axon_recovery(self, fish_name, missing_val=None):
        ''' Get distal axon regrowth for the last assay of the fish.
        Requires "distal_axon_regrowth.csv" file to exist.
        '''
        final_assay = max(self.get_available_assay_labels(fish_name))
        return PrecalculatedMetricsAnalyzer().get_value(
            fish_name, final_assay, key='distal_axon_regrowth', missing_value=missing_val)

    #    ------ Filename helpers ------

    @staticmethod
    def _get_fish_cache_name(name) -> pathlib.Path:
        ''' Get the location of the cache for name.
        NOTE: this is the fish's name
        '''
        return FileLocations.get_fish_cache() / f'{name}.pickle'

    def clean_cache(self):
        ''' Removes assays that do not exist.
        Assumes that the file FileLocations.get_fish_to_assays_csv()
        is the truth.
        '''
        for name in self.get_available_fish_names():
            fd = self.load_fish_data(name)
            swim_keys = list(fd.keys())
            true_swims = self.get_available_assay_labels(name)
            changed = False
            for k in true_swims:
                if k not in swim_keys:
                    print(f'Missing {k} in {name}!')
            for k in swim_keys:
                if k not in true_swims:
                    print(f'Removing {k} from {name}')
                    del fd[k]
                    changed = True
            if changed:
                self.save_fish_data(name, fd)

    @functools.lru_cache(maxsize=600, typed=False)
    def get_corners_of_swim_area(self, fish_name: str, assay: int):
        ''' Finds the height and width of the swimable area,
        returned as an (X, Y) point where X is width and Y is height.
        Prefers to use the video

        The only reason the corners are not full-sized is if there's a divider.
        '''
        video_fname = FileLocations.find_video(fish_name, assay)
        logfile = Extractor.CropTracker.find_existing_logfile(video_fname)
        minx, maxx = 0, config.getint('VIDEO', 'default_video_width')
        miny, maxy = 0, config.getint('VIDEO', 'default_video_height')
        if logfile is None:
            if video_fname is not None and video_fname.exists():
                maxy, maxx = fp_ffmpeg.read_frame(
                    video_fname, 1, transpose_to_match_convention=True).shape[:2]
        elif logfile.exists():
            ct = Extractor.CropTracker.CropTracker.from_logfile(logfile)
            maxx, maxy = ct.full_width, ct.full_height
            tracker_type_file = pathlib.Path(video_fname).parent / 'tracker_type.txt'
            tracker_type = None
            if tracker_type_file.exists():
                with open(tracker_type_file, 'rt') as fh:
                    tracker_type = fh.readline().strip()
            if tracker_type == 'SplitCropTracker':
                side = data_utils.parse_details_from_filename(logfile)[0].side
                if side == 'R':
                    minx = maxx / 2
                else:
                    maxx /= 2
        return minx, miny, maxx, maxy

DataManager = _FishManager()
