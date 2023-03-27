'''
Holds PCA results, but does not handle cache import or export.
Also contains wrappers for most zplib.pca functions.
'''
import numpy
from zplib import pca

from swimfunction.data_models.DotEnabledDict import DotEnabledDict

PCA_KEYS = (
    'mean',
    'pcs',
    'norm_pcs',
    'variances'#,
    # 'positions',  NO LONGER SAVED
    # 'norm_positions'  NO LONGER SAVED
)
INDEX_TO_KEY = dict(zip(range(len(PCA_KEYS)), PCA_KEYS))

class PCAResult(DotEnabledDict):
    '''PCAResult extends DotEnabledDict. Basically a mutable namedtuple.
    Expected attributes:
     - self.mean
     - self.pcs
     - self.norm_pcs
     - self.variances

    We no longer save these attributes:
     - self.positions
     - self.norm_positions
    '''
    __slots__ = PCA_KEYS
    def __init__(self, *args, **kwargs):
        ''' Initialize with 6 arguments, one iterable with 6 items, or nothing.
        Checks for valid assignment, it not valid then

        Raises
        ------
        IOError
            If you didn't provide good arguments.
        '''
        if len(args) == 1:
            super().__init__()
            if isinstance(args[0], (tuple, list)):
                self.from_tuple(args[0])
            elif isinstance(args[0], dict):
                self.from_dict(args[0])
        elif len(args) > 1:
            super().__init__()
            self.from_tuple(args)
        elif not args:
            super().__init__()
            self.from_tuple((None for _ in PCA_KEYS))
        elif kwargs and not args:
            self.from_dict(kwargs)
        else:
            super().__init__(*args, **kwargs)
        self.assert_correct()

    def assert_correct(self):
        ''' Checks for valid assignments. Essentially makes this a mutable namedtuple class.
        '''
        for k in PCA_KEYS:
            if not hasattr(self, k):
                raise IOError(f'PCAResult: must provide key {k}')
        for k in self:
            if k not in PCA_KEYS:
                raise IOError(f'PCAResult: contains key not in PCA_KEYS: {k}')

    def from_dict(self, d):
        for k in list(d.keys()):
            if k not in PCA_KEYS:
                d.pop(k)
        self.update(d)
        return self

    def from_tuple(self, pca_result_tuple):
        ''' Loads from tuple (output of pca.pca)
        '''
        (
            self.mean,
            self.pcs,
            self.norm_pcs,
            self.variances) = list(pca_result_tuple)[:4]
        return self

    def __getitem__(self, index):
        ''' For backwards compatability, can still access attributes as if a tuple.
        '''
        if isinstance(index, (str)):
            return self.__getattr__(index)
        return self.__getattr__(INDEX_TO_KEY[index])

    def decompose(self, data, ndims=None):
        ''' Helpful wrapper for pca.pca_decompose
        '''
        if ndims is None:
            ndims = len(self.pcs)
        return pca.pca_decompose(data, self.pcs[:ndims], self.mean)

    def pca(self, data, verbose=False):
        ''' Helpful wrapper for pca.pca
        '''
        self.from_tuple(pca.pca(data))
        if verbose:
            first_three_var = 100 * sum(self.variances[:3]) / sum(self.variances)
            print('Shape:', data.shape)
            print(','.join((f'{100*x:.2f}%' for x in self.variances[:3] / sum(self.variances))))
            print(f'{first_three_var:.2f}% in first 3 components')
        return self

    def reconstruct(self, positions, ndims=None):
        ''' Helpful wrapper for pca.pca_reconstruct
        '''
        if ndims is None:
            ndims = len(self.pcs)
        if len(positions.shape) == 1:
            positions = positions.reshape((-1, 1))
        return pca.pca_reconstruct(positions, self.pcs[:ndims, ...], self.mean)

    def get_percent_variance_explained(self):
        ''' Get percent variance explained by each PC.
        Values range from 0 to 1.
        '''
        return self.variances / self.variances.sum()

    def get_npcs_for_required_variance(self, required_variance: float) -> int:
        ''' Fewest number of pcs required to meet the amount of variance requested.
        '''
        if required_variance > 1 or required_variance < 0:
            raise RuntimeError('Required variance must be a float between 0 and 1.')
        return sorted(
            numpy.where(
                (numpy.cumsum(self.get_percent_variance_explained()) - required_variance) > 0
            )[0])[0] + 1

    def report_percent_variance(self):
        ''' Report and return percent variance and cumulative variance explained by each PC
        where 100 is the total variance (values range from 0 to 100).
        '''
        percents = self.get_percent_variance_explained() * 100
        print('\n'.join([
            f'\tPC{i} : {x:.1f}% : {cumulative:.1f}% cumulative'
            for i, (x, cumulative) in enumerate(zip(percents, percents.cumsum())) ]))
        return percents, percents.cumsum()

    def plot_variance(self, title=None, ax=None, legend: bool=True, percent_text: bool=True, cumulative_plot_kws: dict={}, bar_kws: dict={}):
        ''' Scree plot with variance and percent variance of pcs
        '''
        # Late import for speed. Most PCAResult do not need matplotlib.
        from matplotlib import pyplot as plt
        from matplotlib import ticker as mtick
        var, cumulative = self.report_percent_variance()
        if ax is None:
            _fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.bar(range(len(var)), var, **bar_kws)
        cp_kws = dict(linestyle='-', marker='.', clip_on=False, color='red', label='Cumulative variance')
        cp_kws.update(cumulative_plot_kws)
        ax.plot(range(len(var)), cumulative, **cp_kws)
        if percent_text:
            for pc_num, percent_var in zip(range(len(var)), cumulative):
                ax.text(pc_num, percent_var, f'{percent_var:.1f}%', fontsize=8)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=cumulative.max(), symbol=None))
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xticks(range(len(var)))
        ax.set_xticklabels(range(1, len(var) + 1), fontsize=8)
        ax.tick_params(axis='x', which='both', length=0)
        ax.set_xlabel('Principal Components (PCs)', fontsize=8)
        ax.set_ylabel('Explained Variance (%)', fontsize=8)
        if legend:
            ax.legend(loc='center')
