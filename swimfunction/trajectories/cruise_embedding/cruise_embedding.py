''' Embed exemplar cruise episodes using UMAP
Many methods here are similar to Mearns et al. 2019
    https://bitbucket.org/mpinbaierlab/mearns_et_al_2019
'''
from functools import partial
from typing import List, Tuple
import joblib
import numpy
from scipy.spatial.distance import squareform
from sklearn.cluster import AffinityPropagation
from tqdm import tqdm

from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.Fish import Fish
from swimfunction.trajectories.cruise_embedding import dynamic_time_warping
from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
from swimfunction.data_access import PoseAccess, data_utils
from swimfunction.context_managers.CacheContext import CacheContext

from swimfunction import FileLocations
from swimfunction import loggers
from swimfunction import progress

# whether to use a global available PCA model
# otherwise performs experiment-specific PCA.
USE_TRACKING_EXPERIMENT_POSE_PCA = True

# Number of PCs to use for the embedding
N_DIMS = 5

# Fish with scoliosis higher than this
# at the final assay will be omitted completely.
SCOLIOSIS_THRESHOLD = 0.35

if USE_TRACKING_EXPERIMENT_POSE_PCA:
    PCA_RESULT = data_utils.get_tracking_experiment_pca()
PCA_RESULT = data_utils.calculate_pca()

npload = partial(numpy.load, allow_pickle=True)

def get_logger() -> loggers.logging.Logger:
    ''' Get the default metric calculation logger.
    '''
    return loggers.get_cruise_embedding_logger(__name__)

def _pad_to_equal_shapes(
        list_of_ndarray: List[numpy.ndarray],
        pad_value=0) -> List[numpy.ndarray]:
    ''' Ensures all arrays in the list have the same shape
    by padding short arrays with a pad value on the end.
    '''
    longest = max([l.shape[0] for l in list_of_ndarray])
    list_of_ndarray = [
        numpy.pad(l, ((0, longest - l.shape[0]), (0, 0)), 'constant', constant_values=pad_value)
        if l.shape[0] < longest else l
        for l in list_of_ndarray]
    return list_of_ndarray

class EmbeddingStrategy:
    def __init__(self, name: str, unique_tag: str, model):
        '''
        Parameters
        ----------
        name: str
            Example: 'isomap' or 'tsne'
        unique_tag: str
            Short string to identify the parameters used to initialize the model, used for naming output files.
            Example: 'n20_c10'
        model
        '''
        self.name = name
        self.unique_tag = unique_tag
        self.model = model

    def __str__(self):
        return f'{self.name}_{self.unique_tag}'

    def fit_transform(self, X):
        return self.model.fit_transform(X)

class UmapStrategy(EmbeddingStrategy):
    def __init__(self, n_neighbors: int, n_components: int):
        # Since it takes a long time to load,
        # we only import UMAP when absolutely necessary.
        from umap import UMAP
        super().__init__(
            'umap',
            f'n{n_neighbors}_c{n_components}',
            UMAP(
                n_components=n_components,
                metric='precomputed'))

class EmbeddingInterface:
    ''' Stores these three parallel arrays:
        _keys : tuples (fish_name, wpi)
        _exemplars : cruise episodes as PCA decomposed angle poses
        _exemplars_idx : cruise episodes as frame numbers

    Also stores these two parallel arrays:
        _valid_indices : indices of _keys that include fish names
        _embedding : embedding of only exemplars indexed with _valid_indices
    '''

    def __init__(
            self,
            include_scoliotic_fish: bool,
            all_fish_names: list,
            strategy: EmbeddingStrategy):
        '''
        Parameters
        ----------
        include_scoliotic_fish: bool
        all_fish_names: list
            All fish, including scoliotic, that are in the experiment.
            If you want to ignore any fish, omit them before sending to this function.
        strategy: EmbeddingStrategy
        '''
        self.fish_names = all_fish_names
        if not include_scoliotic_fish:
            self.fish_names = []
            for name in all_fish_names:
                score = FDM.get_final_scoliosis(name)
                if score is not None and score < SCOLIOSIS_THRESHOLD:
                    self.fish_names.append(name)
        self.strategy = strategy

        self.save_dir = FileLocations.get_cruise_embedding_dir(
            include_scoliotic_fish) / f'{self.strategy}_embedding'
        self.save_dir.mkdir(exist_ok=True)
        shared_save_dir = FileLocations.get_cruise_embedding_dir(
            include_scoliotic_fish).parent
        self.exemplar_key_file = shared_save_dir / 'exemplar_keys.pickle'
        self.exemplar_file = shared_save_dir / 'exemplars.pickle'
        self.exemplar_idx_file = shared_save_dir / 'exemplars_idxs.pickle'
        self.distances_file = shared_save_dir / 'exemplars_distances.npz'
        self.logger = get_logger()
        # _valid_indices: Where to index into _keys, _exemplars, and _exemplars_idx
        # to get values used in _embedding
        self._valid_indices = None
        self._exemplars = None
        self._keys = None
        self._exemplars_idx = None
        self._embedding = None

    ''' Data Access '''

    @property
    def embedding_file(self):
        ''' File to save the embedding
        '''
        return self.save_dir / f'{self.strategy}_embedding.npz'

    def get_exemplar_indices(self, fish_name: str, assay_label: int):
        ''' Find the indices for exemplars from the fish's assay.
        '''
        where_fish = numpy.asarray([k[0] == fish_name for k in self.keys])
        where_assay = numpy.asarray([k[1] == assay_label for k in self.keys])
        locs = numpy.where(where_fish & where_assay)[0]
        return [self.exemplars_idx[loc] for loc in locs]

    def get_exemplars_embedded(self, fish_name_or_names, assay_label: int) -> numpy.ndarray:
        ''' Get embedded exemplars for the fish (one or many) at the requested assay.
        '''
        if isinstance(fish_name_or_names, list):
            where_names = numpy.asarray(
                [numpy.in1d(k[0], fish_name_or_names)[0] for k in self.keys])
        else:
            where_names = numpy.asarray([k[0] == fish_name_or_names for k in self.keys])
        where_assay = numpy.asarray([k[1] == assay_label for k in self.keys])
        if not self.embedding.size:
            return numpy.asarray([])
        return self.embedding[where_names & where_assay, :]

    def _set_private_precalculated_values(self):
        ''' Get the keys, exemplars, and indices either from a precalculated file,
        or calculate them freshly if the file does not exist.
        '''
        self._keys, self._exemplars, self._exemplars_idx = self._get_exemplars_for_all_fish()
        self._valid_indices = [i for i, k in enumerate(self._keys) if k[0] in self.fish_names]

    @property
    def exemplars(self):
        ''' Get the distances either from a precalculated file,
        or calculate them freshly if the file does not exist.
        '''
        if self._exemplars is None:
            self._set_private_precalculated_values()
        return numpy.asarray([self._exemplars[i] for i in self._valid_indices])

    @property
    def keys(self):
        ''' Get keys either from a precalculated file,
        or calculate them freshly if the file does not exist.
        '''
        if self._keys is None:
            self._set_private_precalculated_values()
        return numpy.asarray([self._keys[i] for i in self._valid_indices])

    @property
    def exemplars_idx(self):
        ''' Get the exemplar indices either from a precalculated file,
        or calculate them freshly if the file does not exist.
        '''
        if self._exemplars_idx is None:
            self._set_private_precalculated_values()
        return numpy.asarray([self._exemplars_idx[i] for i in self._valid_indices])

    @property
    def distances(self) -> numpy.ndarray:
        ''' Get the distances either from a precalculated file,
        or calculate them freshly if the file does not exist.
        '''
        if self._exemplars is None:
            self._set_private_precalculated_values()
        if len(self._exemplars) < 2:
            return numpy.asarray([])
        if not self.distances_file.exists():
            self.logger.info('Calculating distances between all exemplars...')                
            distances = self._get_distance_matrix(_pad_to_equal_shapes(self._exemplars))
            numpy.savez_compressed(self.distances_file, distances)
        distances = npload(self.distances_file)['arr_0']
        return distances[tuple(numpy.meshgrid(self._valid_indices, self._valid_indices))]

    @property
    def embedding(self):
        ''' Get the embedding either from a precalculated file,
        or calculate them freshly if the file does not exist.
        '''
        if self._embedding is None:
            self._embedding = self._calculate_embedding()
        return self._embedding

    def _save_exemplars(self, keys, exemplars, exemplar_idxs):
        ''' Save the keys, exemplars, and indices to a file
        '''
        with CacheContext(self.exemplar_key_file) as cache:
            cache.saveContents(keys)
        with CacheContext(self.exemplar_file) as cache:
            cache.saveContents(exemplars)
        with CacheContext(self.exemplar_idx_file) as cache:
            cache.saveContents(exemplar_idxs)

    def _get_exemplars_for_all_fish(self):
        ''' Get the keys, exemplars, and indices either from a precalculated file,
        or calculate them freshly if the file does not exist.
        '''
        if not self.exemplar_file.exists():
            return self._calculate_exemplars_for_all_fish()
        with CacheContext(self.exemplar_key_file) as cache:
            keys = cache.getContents()
        with CacheContext(self.exemplar_file) as cache:
            exemplars = cache.getContents()
        with CacheContext(self.exemplar_idx_file) as cache:
            exemplar_idxs = cache.getContents()
        return (keys, exemplars, exemplar_idxs)

    ''' Calculations '''

    def _get_distance_matrix(self, episodes: List[numpy.ndarray]) -> numpy.ndarray:
        ''' Similar to Mearns et al. 2019
        https://bitbucket.org/mpinbaierlab/mearns_et_al_2019
        '''
        nthreads = WorkerSwarm.num_allowed_workers
        total = len(episodes)
        if total < 2:
            return None
        sec = total * total / 1780 / nthreads
        time_str = '%d:%02d:%02d:%02d' % (sec // (60 * 60 * 24), sec // (60 * 60), sec // 60, sec % 60)
        self.logger.info(f'{total} episodes makes for {total} x {total} = {total*total} matrix, with {nthreads} threads taking about {time_str} (d:hh:mm:ss)')
        progress.init(total)
        def fill_row(row_i):
            l = [
                dynamic_time_warping.dynamic_time_warp(
                    episodes[row_i], episodes[i], distance_only=True).distance
                for i in range(row_i+1, len(episodes))
            ]
            progress.increment(row_i, total)
            return l
        distance_matrix = joblib.Parallel(nthreads)(
            joblib.delayed(fill_row)(i) for i in range(len(episodes) - 1)
        )
        distances = squareform(numpy.concatenate(distance_matrix))
        progress.finish()
        return distances

    def _calculate_exemplars(self, fish: Fish, assay_label) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
        ''' Similar to Mearns et al. 2019
        https://bitbucket.org/mpinbaierlab/mearns_et_al_2019
        '''
        episodes, episode_indices = PoseAccess.get_decomposed_assay_cruise_episodes(
            fish, assay_label, PCA_RESULT, ndims=N_DIMS)
        if not episodes:
            self.logger.info(f'{fish.name} {assay_label} : no episodes available.')
            return [], []

        if len(episodes) == 1:
            self.logger.info(f'{fish.name} {assay_label} : only one episode available.')
            return episodes, episode_indices

        unique_labels = [-1]
        attempt = 0
        distance_matrix = self._get_distance_matrix(episodes)
        while len(unique_labels) == 1 and -1 in unique_labels:
            attempt += 1
            if attempt < 5:
                clusterer = AffinityPropagation(
                    affinity='precomputed', preference=numpy.median(-distance_matrix))
                clusterer.fit_predict(-distance_matrix)
                cluster_labels = clusterer.labels_
                cluster_centers = clusterer.cluster_centers_indices_
                cluster_size_threshold = 3
                cluster_sizes = numpy.array([
                    numpy.sum(cluster_labels == l)
                    for l in numpy.unique(cluster_labels)])
                big_clusters = numpy.where(cluster_sizes >= cluster_size_threshold)[0]
                unique_labels = set(cluster_labels)
                n_clusters = len(numpy.unique(cluster_labels))
            else:
                self.logger.error(f'{fish.name} {assay_label} : Affinity propagation failed to converge five times. Randomly sampling up to 16 examples.')
                n_clusters = min(len(distance_matrix), 16)
                big_clusters = numpy.arange(n_clusters)
                cluster_centers = numpy.random.choice(numpy.arange(len(distance_matrix)), size=n_clusters, replace=False)
                break
        self.logger.info(f'{fish.name} {assay_label} : Affinity propagation clusters: {big_clusters.size} large enough, {n_clusters} total.')
        return (
            [episodes[i] for i in cluster_centers[big_clusters]],
            [episode_indices[i] for i in cluster_centers[big_clusters]]
        )

    def _calculate_exemplars_for_all_fish(self):
        ''' Get exemplars for each fish, each assay.
        '''
        self.logger.info('Calculating exemplars for each assay...')
        keys = []
        exemplars = []
        exemplar_idxs = []
        pbar_outer = tqdm(FDM.get_available_fish_names())
        for name in pbar_outer:
            pbar_outer.set_description(name)
            pbar_outer.refresh()
            fish = Fish(name).load()
            pbar = tqdm(fish.swim_keys(), leave=False)
            for assay_label in pbar:
                pbar.set_description(f'{name} {assay_label}wpi')
                pbar.refresh()
                ee, ei = self._calculate_exemplars(fish, assay_label)
                if not ee:
                    continue
                keys += [(name, assay_label)] * len(ee)
                exemplars += ee
                exemplar_idxs += ei
        self._save_exemplars(keys, exemplars, exemplar_idxs)
        return keys, exemplars, exemplar_idxs

    def _calculate_embedding(self, recalculate=False) -> numpy.ndarray:
        ''' Calculates an embedding for exemplars
        based on the distance matrix.
        '''
        embedding = numpy.asarray([])
        if self.embedding_file.exists() and not recalculate:
            self.logger.info('Loading precalculated embedding...')
            embedding = npload(self.embedding_file)['arr_0']
            with open(self.save_dir / 'fish_names_used_in_embedding.txt', 'rt') as fh:
                names = fh.readline().strip().split(',')
            if set(names) != set(self.fish_names):
                print('Fish in embedding:', names)
                raise RuntimeError('Fish names must match those fish that were used in the embedding.')
        else:
            self.logger.info(
                'Embedding with %s...'
                %(self.strategy))
            dists = self.distances
            if dists is not None and dists.size:
                embedding = self.strategy.fit_transform(dists)
                numpy.savez_compressed(self.embedding_file, embedding)
            # For logging purposes
            with open(self.save_dir / 'fish_names_used_in_embedding.txt', 'wt') as fh:
                fh.write(','.join(self.fish_names))
        return embedding

if __name__ == '__main__':
    FileLocations.parse_default_args()
