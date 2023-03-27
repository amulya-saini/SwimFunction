''' Helpers for sampling poses, either simple random sampling (SRS)
or stratified random sampling, stratified using K-means.
'''

import numpy
from sklearn.cluster import KMeans

from swimfunction.global_config.config import config

K_MEANS = config.getint('POSE_SPACE_TRANSFORMER', 'k_means_for_stratified_sampling')
POSES_PER_ASSAY = config.getint('POSE_SPACE_TRANSFORMER', 'poses_to_sample_per_assay')

FEATURE = 'smoothed_angles'

def simple_sample(xx, count, return_positions_only: bool=False):
    ''' Simple random sample of array of poses
    '''
    locs = numpy.random.choice(
        numpy.arange(xx.shape[0]),
        size=min(count, xx.shape[0]),
        replace=False)
    if return_positions_only:
        return locs
    return xx[locs, ...]

def stratified_sample(xx, count, k_means: int = 12, return_positions_only: bool=False):
    ''' Stratified sample with k-means clustering.
    Get the k-means clusters, simple random sample from each of the clusters.
    '''
    km = KMeans(n_clusters=k_means)
    klabels = km.fit_predict(xx)
    sampled_locs = []
    for l in numpy.unique(klabels):
        sampled_locs.append(
            simple_sample(
                numpy.where(klabels == l)[0],
                count // k_means))
    if return_positions_only:
        return numpy.concatenate(sampled_locs)
    return xx[numpy.concatenate(sampled_locs), ...]
