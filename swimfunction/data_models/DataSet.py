''' Basic definition of DataSet that works well with our data.
'''
from sklearn import model_selection
from sklearn import utils as skutils
from imblearn.over_sampling import SMOTE

import numpy
import pandas

class DataSet:
    ''' Holds data, labels for the data, and meta labels.

    Note on the types:
        data is numpy.ndarray
        labels is numpy.ndarray
        meta_labels is pandas.DataFrame
    '''
    __slots__ = ['data', 'labels', 'meta_labels']

    def __init__(
            self,
            data: numpy.ndarray=None,
            labels: numpy.ndarray=None,
            meta_labels: pandas.DataFrame=None):
        ''' Holds data, labels, and meta_labels

        Parameters
        ----------
        data : numpy.ndarray, default=None
        labels : numpy.ndarray, default=None
        meta_labels : pandas.DataFrame, default=None
            Pandas can name columns for convenience
        '''
        if data is None:
            data = numpy.asarray([])
        if labels is None:
            labels = numpy.asarray([])
        if meta_labels is None:
            meta_labels = pandas.DataFrame()
        self.data = numpy.asarray(data)
        self.labels = numpy.asarray(labels)
        self.meta_labels = meta_labels

    def as_dict(self):
        ''' Converts to a dictionary (for pickling)
        '''
        return {
            'data': self.data,
            'labels': self.labels,
            'meta_labels': self.meta_labels
        }

    def as_equally_represented_upsampled(self):
        ''' Upsamples over-represented data labels
        such that labels have equal representation in the DataSet.

        Returns
        -------
        new : DataSet
            A new DataSet with labels equally represented.
        The data for each label takes the length of the least-represented label.
        Over-represented labels are subsampled without replacement.
        '''
        sm = SMOTE(k_neighbors=20, n_jobs=-1)
        new_data, new_labels = sm.fit_resample(self.data, self.labels)
        return self.__class__(
            data=new_data,
            labels=new_labels
        )

    def as_equally_represented_downsampled(self):
        ''' Downsamples over-represented data labels
        such that labels have equal representation in the DataSet.

        Returns
        -------
        new : DataSet
            A new DataSet with labels equally represented.
        The data for each label takes the length of the least-represented label.
        Over-represented labels are subsampled without replacement.
        '''
        counts = dict(zip(*numpy.unique(self.labels, return_counts=True)))
        smallest_count = min(counts.values())
        data = []
        labels = []
        meta_labels = []
        for label in counts.keys():
            positions = numpy.where(self.labels == label)[0]
            take_positions = numpy.random.choice(positions, size=smallest_count, replace=False)
            data.append(numpy.take(self.data, take_positions, axis=0))
            labels.append(numpy.take(self.labels, take_positions, axis=0))
            meta_labels.append(self.meta_labels.take(take_positions, axis=0))
        return self.__class__(
            data=numpy.concatenate(data),
            labels=numpy.concatenate(labels),
            meta_labels=pandas.concat(meta_labels) if meta_labels else pandas.DataFrame()
        )

    def drop_nan(self):
        ''' Remove nan data
        '''
        has_nan_mask = numpy.any(numpy.isnan(self.data), axis=1)
        self.data = self.data[~has_nan_mask]
        self.labels = self.labels[~has_nan_mask]
        self.meta_labels = self.meta_labels.iloc[~has_nan_mask]
        return self

    def drop_labels(self, labels_to_drop: list):
        ''' Removes all items from the dataset that have the given labels.

        Returns
        -------
        self
        '''
        for label in labels_to_drop:
            positions = numpy.where(self.labels != label)[0]
            self.data = numpy.take(self.data, positions, axis=0)
            self.labels = numpy.take(self.labels, positions, axis=0)
            self.meta_labels = self.meta_labels.take(positions, axis=0)
        return self

    def split_for_cross_validation(self, percent_test: int) -> tuple:
        '''
        Returns
        -------
        train_test : tuple
            two DataSets (train, test) split from self.
            The returned datasets are shuffled and stratified.

        '''
        data_set = self
        data = data_set.data
        labels = data_set.labels
        meta_labels = data_set.meta_labels
        num_points = labels.shape[0]
        num_test_points = int(num_points*(percent_test/100))
        train_d, test_d, train_l, test_l, train_ml, test_ml = model_selection.train_test_split(
            data,
            labels,
            meta_labels,
            test_size=num_test_points,
            shuffle=True,
            stratify=labels)
        # Creates child classes if this function is inherited.
        return (
            self.__class__(
                train_d,
                train_l,
                train_ml),
            self.__class__(
                test_d,
                test_l,
                test_ml)
        )

    def as_subsampled(self, max_size=100000):
        ''' Subsamples to max_size, or returns if already less than max size.
        '''
        if self.labels.shape[0] <= max_size:
            return self
        new_metalabels = None
        if self.meta_labels is not None and self.meta_labels.shape[0] == self.labels.shape[0]:
            new_data, new_labels, new_metalabels = skutils.resample(
                self.data, self.labels, self.meta_labels,
                n_samples=max_size, replace=False, stratify=self.labels,
                random_state=0)
        else:
            new_data, new_labels = skutils.resample(
                self.data, self.labels,
                n_samples=max_size, replace=False, stratify=self.labels,
                random_state=0)
        return self.__class__(
            new_data,
            new_labels,
            new_metalabels)

    def subset_by_meta_label(self, feature, value):
        ''' Get the portion of the data where the meta feature equals a value.
        '''
        loc = numpy.where(self.meta_labels[feature] == value)
        return self.__class__(
            self.data[loc],
            self.labels[loc],
            self.meta_labels[self.meta_labels[feature] == value])

    def subset_by_meta_labels(self, **kwargs):
        ''' Get the portion of the data where each meta feature equals its respective value.
        '''
        data_set = self
        for feature, value in kwargs.items():
            data_set = data_set.subset_by_meta_label(feature, value)
            if not data_set.labels.size:
                break
        return data_set

    def subset_by_labels(self, labels_to_keep):
        ''' Select the data according to label values.

        Parameters
        ----------
        labels_to_keep: tuple, list, numpy.ndarray

        Returns
        -------
        new : DataSet
            A new DataSet containing only labels in labels_to_keep.
        '''
        data = []
        labels = []
        meta_labels = []
        for label in labels_to_keep:
            positions = numpy.where(self.labels == label)[0]
            data.append(numpy.take(self.data, positions, axis=0))
            labels.append(numpy.take(self.labels, positions, axis=0))
            if self.meta_labels is not None and self.meta_labels.size:
                meta_labels.append(self.meta_labels.take(positions, axis=0))
        return self.__class__(
            data=numpy.concatenate(data),
            labels=numpy.concatenate(labels),
            meta_labels=pandas.concat(meta_labels) if meta_labels else pandas.DataFrame()
        )

    @staticmethod
    def cast(ds):
        return DataSet(ds.data, ds.labels, ds.meta_labels)

    @staticmethod
    def from_dict(ds_dict):
        ''' Creates a DataSet from a dict.
        '''
        return DataSet(
            data=ds_dict['data'] if 'data' in ds_dict else None,
            labels=ds_dict['labels'] if 'labels' in ds_dict else None,
            meta_labels=ds_dict['meta_labels'] if 'meta_labels' in ds_dict else None
        )

    def copy(self):
        return self.__class__(
            data=self.data.copy() if self.data is not None else None,
            labels=self.labels.copy() if self.labels is not None else None,
            meta_labels=self.meta_labels.copy(deep=True) if self.meta_labels is not None else None)
