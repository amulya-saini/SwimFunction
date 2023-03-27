''' Novelty detection in PC reduced space.
'''
from collections import namedtuple
from typing import Dict
import threading
import traceback
import numpy
from sklearn.neighbors import LocalOutlierFactor as LOF

from swimfunction.data_access import PoseAccess, data_utils
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.SwimAssay import SwimAssay
from swimfunction.context_managers.CacheContext import CacheContext
from swimfunction.pose_processing import pose_filters
from swimfunction.pose_processing import sample_poses
from swimfunction.recovery_metrics.metric_analyzers.AbstractMetricAnalyzer \
    import AbstractMetricAnalyzer
from swimfunction.global_config.config import config
from swimfunction import FileLocations


NUM_PCS = config.getint('POSE_SPACE_TRANSFORMER', 'num_pcs')
LOF_ARGS = dict(n_neighbors=50, n_jobs=-1)
NOVELTY_COUNTS = namedtuple(
    'NoveltyCounts',
    ['num_ordinary_poses', 'num_novel_poses', 'percent_novelty'])



class NullNoveltyCounts:
    ''' Holds dict from SwimAssay key to NOVELTY_COUNTS namedtuple
    '''
    def __init__(self):
        self.assay_to_counts = {}

    def total_novelty(self) -> NOVELTY_COUNTS:
        num_ordinary = sum((counts.num_ordinary_poses for counts in self.assay_to_counts.values()))
        num_novel = sum((counts.num_novel_poses for counts in self.assay_to_counts.values()))
        if num_ordinary == 0 and num_novel == 0:
            return NOVELTY_COUNTS(0, 0, 0)
        return NOVELTY_COUNTS(
            num_ordinary,
            num_novel,
            num_novel / (num_novel + num_ordinary)
        )

    @staticmethod
    def details_to_key(fish_name: str, assay_label: int):
        return f'{config.experiment_name}_{fish_name}_{assay_label}'

    @staticmethod
    def swim_to_key(swim: SwimAssay) -> str:
        return NullNoveltyCounts.details_to_key(swim.fish_name, swim.assay_label)

    def as_dict(self):
        ''' Save the object as a dictionary
        '''
        return {
            swim_key: counts._asdict() for swim_key, counts in self.assay_to_counts.items()
        }

    @staticmethod
    def from_dict(counts_dict: Dict[str, dict]):
        ''' Load the object from a dictionary
        '''
        nnc = NullNoveltyCounts()
        nnc.assay_to_counts = {
            swim_key: NOVELTY_COUNTS(*[counts[field] for field in NOVELTY_COUNTS._fields])
            for swim_key, counts in counts_dict.items()
        }
        return nnc

class PostureNoveltyAnalyzer(AbstractMetricAnalyzer):
    ''' Calculates outliers in the null distribution
    and uses the null distribution (including outliers)
    to create a LocalOutliersFactor model against which
    all other assays are compared. Percent novely
    is the number of novel (outlier) poses in the assay
    divided by the total number of high quality poses.
    '''

    lof_model_dict = {}
    null_novelty_counts_dict = {}
    SETUP_LOCK = threading.Lock()

    __slots__ = ['use_global_resource', 'null_group', 'null_assay', 'pca_result']
    def __init__(self, use_global_resource: bool=True, null_group: str=None, null_assay: int=-1):
        '''
        Parameters
        ----------
        use_global_resource: bool
            If true, other parameters are ignored and the global model is used.
            If false, makes/uses a local resource.
        null_group: str
            Only used if not use_global_resource
            The group that forms the null distribution of the novelty detector.
            Example: -1 for preinjury. Default: -1 (preinjury)
        null_assay: int
            Only used if not use_global_resource
            The assay that forms the null distribution of the novelty detector.
            Example: 'F' for Female. Default: all groups.
        '''
        super().__init__()
        self.null_group = null_group
        self.null_assay = null_assay
        self.use_global_resource = use_global_resource
        if self.use_global_resource:
            self.pca_result = data_utils.get_tracking_experiment_pca()
        else:
            self.pca_result = data_utils.calculate_pca()
        self.keys_to_printable = {
            'posture_novelty': '% Novelty'}

    def analyze_assay(self, swim_assay: SwimAssay) -> dict:
        ''' Get novelty counts for a swim assay based on LOF novelty detector.
        If assay was in the null distribution then returns appropriate
        precalculated value.

        If you forgot to call "load", then it will attempt to load.
        If load fails and not use_global_resource, then it will run self.first_time_setup()
        '''
        self.guarantee_setup_has_completed()

        posture_novelty = numpy.nan
        swim_key = NullNoveltyCounts.swim_to_key(swim_assay)
        if swim_key in self.null_novelty_counts.assay_to_counts:
            posture_novelty = self.null_novelty_counts.assay_to_counts[swim_key].percent_novelty
        else:
            angle_poses = PoseAccess.get_feature_from_assay(
                swim_assay, 'smoothed_angles',
                pose_filters.BASIC_FILTERS, keep_shape=False)
            posture_novelty = self.get_novelty(angle_poses).percent_novelty
        return {
            'posture_novelty': posture_novelty
        }

    @property
    def null_novelty_counts(self) -> NOVELTY_COUNTS:
        ''' Get the stored null distribution counts or empty namedtuple.
        '''
        key = self.get_output_files().null_novelty_counts_location
        if key in self.null_novelty_counts_dict:
            return self.null_novelty_counts_dict[key]
        return NullNoveltyCounts()

    @property
    def lof_model(self) -> LOF:
        ''' Get the stored model or a default LOF model.
        '''
        key = self.get_output_files().model_location
        if key in self.lof_model_dict:
            return self.lof_model_dict[key]
        return LOF(novelty=True, **LOF_ARGS)

    def get_output_files(self):
        ''' Get locations for saved model and null count files
        '''
        tag = ''
        if self.use_global_resource:
            model_location = FileLocations.GlobalFishResourcePaths.novelty_detection_lof_model
            null_novelty_counts_location = \
                FileLocations.GlobalFishResourcePaths.novelty_null_counts_location
        else:
            save_dir = FileLocations.get_posture_novelty_models_dir()
            tag_parts = []
            if self.null_assay != -1:
                tag_parts.append(f'{self.null_assay}')
            if self.null_group is not None:
                tag_parts.append(self.null_group)
            tag = '_'.join(tag_parts)
            model_location = save_dir \
                / '_'.join(filter(len, [tag, 'novelty_detection_lof_model.pickle']))
            null_novelty_counts_location = save_dir \
                / '_'.join(filter(len, [tag, 'novelty_detection_null_counts.pickle']))
        return namedtuple(
            'Locations',
            ['model_location', 'null_novelty_counts_location'])(
                model_location, null_novelty_counts_location)

    def guarantee_setup_has_completed(self):
        ''' Make sure the setup is complete before trying to analyze assays!
        '''
        # If setup needs to happen, all other threads must wait for setup before continuing.
        with PostureNoveltyAnalyzer.SETUP_LOCK:
            try:
                if not self.is_setup():
                    self.load_model_and_counts()
                # still not setup
                if not self.is_setup() and self.use_global_resource:
                    self.logger.warning(
                        'Global resource does not exist, so it will be created: %s',
                        self.get_output_files()[0].as_posix())
                    self.first_time_setup()
            except Exception as e:
                self.logger.error(e)
                self.logger.error(traceback.format_exc())

    def set_control_distribution_count_outliers(self) -> NOVELTY_COUNTS:
        ''' Since LOF cannot be used to find novelties in the training set,
        I must use two LOF models: one to detect novelty in unseen sets,
        and one to detect outliers in the training set.
        I use the same parameters for both models.

        Keeps track of which points in the null distribution came from which assay,
        so we can still use analyze_assay() on assays included in the null distribution.

        Parameters can be the output of PoseAccess.get_feature()

        Example
        -------
        # Sets control distribution as all preinjury poses.
        analyzer.set_control_distribution_count_outliers()
        '''
        self.logger.debug('Collecting all poses for null distribution...')
        null_dist_fish_names = FDM.get_available_fish_names()
        if self.null_group is not None:
            null_dist_fish_names = [
                n for n in null_dist_fish_names \
                    if data_utils.fish_name_to_group(n) == self.null_group]
        angle_poses_list, fish_names, assays = PoseAccess.get_feature(
                fish_names=null_dist_fish_names,
                assay_labels=[self.null_assay],
                feature='smoothed_angles',
                filters=pose_filters.BASIC_FILTERS,
                keep_shape=False
            )
        reduced_angle_poses_list = [self.pca_reduce(l) for l in angle_poses_list]
        ppa = 1000 # poses per assay
        self.logger.debug('Sampling %d poses using k_means stratification...', ppa * len(assays))
        self.logger.info(
            '%f percent of null space used for LOF fitting, those are potentially problematic.',
            100 * ppa * len(assays) / sum(map(len, reduced_angle_poses_list)))
        sampled_angle_poses = numpy.concatenate([
            sample_poses.stratified_sample(
                l,
                count=ppa,
                k_means=12) \
            for l in reduced_angle_poses_list])
        self.logger.info('Training LOF model...')
        self.logger.info(
            'Note regarding RAM: requires two matrices of size %d',
            sampled_angle_poses.shape[0] * LOF_ARGS["n_neighbors"])
        self.lof_model.fit(sampled_angle_poses)
        self.logger.info('Finding outliers in null distribution...')
        for reduced_angle_poses, name, assay in zip(reduced_angle_poses_list, fish_names, assays):
            labels = self.lof_model.predict(reduced_angle_poses)
            inliers = (labels == 1).sum()
            outliers = (labels == -1).sum()
            counts = NOVELTY_COUNTS(inliers, outliers, outliers / (inliers + outliers))
            self.null_novelty_counts.assay_to_counts[
                NullNoveltyCounts.details_to_key(name, assay)] = counts

    def get_novelty(self, angle_poses) -> NOVELTY_COUNTS:
        ''' NOTE: you must NOT use any angle_poses from the control distribution.
        '''
        self.guarantee_setup_has_completed()

        reduced_poses = self.pca_reduce(angle_poses)
        labels = self.lof_model.predict(reduced_poses)
        num_ordinary = (labels == 1).sum()
        num_novelty = (labels == -1).sum()
        return NOVELTY_COUNTS(
            num_ordinary,
            num_novelty,
            num_novelty / (num_novelty + num_ordinary))

    def pca_reduce(self, angle_poses) -> numpy.ndarray:
        return self.pca_result.decompose(angle_poses, NUM_PCS)

    def is_setup(self):
        ''' Whether null_novelty_counts is populated
        '''
        return len(self.null_novelty_counts.assay_to_counts) > 0

    def first_time_setup(self):
        ''' Set up the required model and save the results.
        '''
        self.set_control_distribution_count_outliers()
        self.save_model_and_counts()

    def load_model_and_counts(self):
        ''' Load lof_model and null_novelty_counts
        '''
        locs = self.get_output_files()
        with CacheContext(locs.model_location) as cache:
            model = cache.getContents()
            if model is not None:
                self.lof_model_dict[locs.model_location] = model
                self.logger.debug('Loaded LOF model')
        with CacheContext(locs.null_novelty_counts_location) as cache:
            counts = cache.getContents()
            if counts is not None:
                self.null_novelty_counts_dict[locs.null_novelty_counts_location]\
                      = NullNoveltyCounts.from_dict(counts)
                self.logger.debug('Loaded novelty null counts')
        return self

    def save_model_and_counts(self, overwrite_resource=False):
        '''
        Only saves if the null distribution has content (self.is_setup())
        Only saves the models if resource does not exist unless overwrite_resource
        '''
        if not self.is_setup():
            return self
        locs = self.get_output_files()
        if not locs.model_location.exists() or overwrite_resource:
            with CacheContext(locs.model_location) as cache:
                cache.saveContents(self.lof_model)
                self.logger.info('Saved LOF model')
        if not locs.null_novelty_counts_location.exists() or overwrite_resource:
            with CacheContext(locs.null_novelty_counts_location) as cache:
                cache.saveContents(self.null_novelty_counts.as_dict())
                self.logger.info('Saved novelty null counts')
        return self
