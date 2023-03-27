''' Predict behaviors using UMAP for reduction, then SVM for prediction.
'''

import numpy
from typing import Tuple

from swimfunction.data_access \
    import BehaviorAnnotationDataManager as behavior_data_manager, data_utils
from swimfunction.data_models import WindowsDataSet, DataSet
from swimfunction.data_models.PCAResult import PCAResult

from swimfunction.global_config.config import config
from swimfunction import FileLocations
from swimfunction.plotting import matplotlib_helpers as mpl_helpers
from swimfunction.pose_processing.pose_conversion import ANGLE_POSE_SIZE

BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
BEHAVIORS_NAMES = config.getstringdict('BEHAVIORS', 'symbols', 'BEHAVIORS', 'names')
BEHAVIOR_COLORS_DICT = config.getfloatdict('BEHAVIORS', 'symbols', 'BEHAVIORS', 'colors')

def plot_with_labels(data, labels, outfile=None):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

    unique_labels = numpy.unique(labels)
    for l, color in zip(unique_labels, mpl_helpers.get_cmap_values('tab10', len(unique_labels))):
        if l in BEHAVIOR_COLORS_DICT:
            color = BEHAVIOR_COLORS_DICT[l]
        xx = data[labels == l, :]
        ax.plot(xx[:, 0], xx[:, 1], '.', label=l, color=color,
            markersize=1, linewidth=1, fillstyle='full',
            markeredgewidth=0.0)
    fig.legend()
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile)
        plt.close(fig)
    else:
        plt.show()

def plot_pca(dataset: DataSet.DataSet, outfile=None):
    if isinstance(dataset, WindowsDataSet.WindowsDataSet):
        dataset = dataset.to_pose_dataset()
        pca = PCAResult().pca(dataset.data)
    else:
        pca = data_utils.calculate_pca(
            group=None,
            feature='smoothed_angles',
            force_recalculate=False,
            verbose=False)
    decomp = pca.decompose(dataset.data, ndims=2)
    plot_with_labels(decomp, dataset.labels, outfile)

class AbstractBehaviorPredictor:
    ### Abstract methods
    def train_models(self, *args, **kwargs):
        raise NotImplementedError('Abstract method')

    def predict_behaviors(self, sequential_poses: numpy.ndarray):
        raise NotImplementedError('Abstract method')

    def classify_windows(self, pose_windows: numpy.ndarray):
        raise NotImplementedError('Abstract method')

    ### Instance methods
    def get_raw_training_data(
                self, percent_test, use_cached_training_data, feature, save
            ) -> Tuple[WindowsDataSet.WindowsDataSet, WindowsDataSet.WindowsDataSet]:
        ''' Get all annotated train and test data, no modifications.
        '''
        original_train, original_test = behavior_data_manager.get_data_for_training(
            percent_test,
            use_cached_training_data,
            feature)
        if save:
            behavior_data_manager.save_training_data(original_train, original_test, feature)
        return original_train, original_test

    def get_processed_training_data(
            self, original_train, original_test, feature,
            labels_to_predict, enforce_equal_representations,
            use_cached_training_data, save, max_size_for_subsampling=None):
        '''
            1. Drops "course_correction" and any other labels that you aren't predicting.
            3. Makes labels numeric.
            2. If enforce_equal_representations: Upsamples labels to make equally represented data set.
            4. If max_size_for_resampling: Reduces training size to specific amount if needed.
            5. If save: Saves train/test datasets and plots train and test sets in PCA and saves to behaviors model directory.
        '''
        feature_tag = f'{feature}_processed'
        train, test = None, None
        if use_cached_training_data:
            train, test = behavior_data_manager.load_training_data(feature_tag)
        if train is None:
            train = self.mask_unwanted_behaviors(original_train.copy(), labels_to_predict)
            test = self.mask_unwanted_behaviors(original_test.copy(), labels_to_predict)
            train.drop_labels([BEHAVIORS.unknown, BEHAVIORS.course_correction])
            test.drop_labels([BEHAVIORS.unknown, BEHAVIORS.course_correction])
            train = self.labels_to_numeric(train)
            test = self.labels_to_numeric(test)
            if enforce_equal_representations:
                train = train.as_equally_represented_upsampled()
                test = test.as_equally_represented_upsampled()
            if max_size_for_subsampling is not None:
                train = train.as_subsampled(max_size=max_size_for_subsampling)
                test = test.as_subsampled(max_size=max_size_for_subsampling)
            if save:
                behavior_data_manager.save_training_data(train, test, feature_tag)
        if save:
            plot_pca(self.labels_from_numeric(test.to_pose_dataset()), FileLocations.get_behaviors_model_dir() / 'test_set_pca.png')
            plot_pca(self.labels_from_numeric(train.to_pose_dataset()), FileLocations.get_behaviors_model_dir() / 'train_set_pca.png')
        return train, test

    def evaluate(self, dataset: WindowsDataSet.WindowsDataSet, do_plot: bool, outfile=None):
        ''' Predict labels then compare to known labels.
        '''
        print('Predicting validation labels...')
        predicted_labels = self.classify_windows(dataset.data)
        print('Plotting validation results...')
        self.verify(
            self.labels_from_numeric(dataset.labels),
            predicted_labels,
            do_plot=do_plot,
            outfile=outfile)
        return predicted_labels

    def confirm_is_angle_pose_list(self, angle_poses):
        if angle_poses.shape[-1] != ANGLE_POSE_SIZE:
            msg = ''.join(['Attempting to predict behaviors ',
                f'for poses of size {angle_poses.shape[-1]}',
                ' is not allowed. Angle poses are size ',
                f'{ANGLE_POSE_SIZE}.'])
            raise RuntimeError(msg)

    def confirm_is_pose_window_list(self, pose_windows):
        if pose_windows.shape[-1] != WindowsDataSet.WindowsDataSet.WINDOW_SIZE:
            msg = ''.join(['Attempting to predict behaviors ',
                f'for pose windows of size {pose_windows.shape[-1]}',
                ' is not allowed. Pose windows are size ',
                f'{WindowsDataSet.WindowsDataSet.WINDOW_SIZE}.'])
            raise RuntimeError(msg)

    ### Static methods
    @staticmethod
    def mask_unwanted_behaviors(data_set: DataSet.DataSet, labels_to_predict):
        for l in numpy.unique(data_set.labels):
            if l not in labels_to_predict:
                data_set.labels[numpy.where(data_set.labels == l)] = BEHAVIORS.unknown
        return data_set

    @staticmethod
    def unmask_unwanted_behaviors(data_set: DataSet.DataSet):
        if data_set.meta_labels.size == 0:
            return data_set
        data_set.labels = numpy.asarray(data_set.meta_labels.label.values.tolist())
        return data_set

    @staticmethod
    def labels_to_numeric(X):
        ''' Converts list of labels into numeric labels if not already numeric.
        Necessary for current versions of sklearn/umap

        Parameters
        ----------
        X : DataSet or numpy.ndarray

        Returns
        -------
        DataSet or numpy.ndarray
            same type as X, where labels are replaced with numeric labels
        '''
        tmp = X
        if isinstance(X, DataSet.DataSet):
            tmp = X.labels
        tmp = numpy.asarray(tmp)
        if numpy.issubdtype(tmp.dtype, numpy.number):
            return X
        LUT = dict(zip(BEHAVIORS, range(len(BEHAVIORS))))
        numeric_labels = numpy.asarray([LUT[b] for b in tmp], dtype=numpy.uint8)
        if isinstance(X, DataSet.DataSet):
            X.labels = numeric_labels
        else:
            X = numeric_labels
        return X

    @staticmethod
    def labels_from_numeric(X):
        ''' Converts labels from numeric labels to BEHAVIORS values,
        which are probably not numeric.

        Parameters
        ----------
        X : DataSet or numpy.ndarray

        Returns
        -------
        DataSet or numpy.ndarray
            same type as X, where numeric labels are replaced with string labels
        '''
        tmp = X
        if isinstance(X, DataSet.DataSet):
            tmp = X.labels
        tmp = numpy.asarray(tmp)
        if not numpy.issubdtype(tmp.dtype, numpy.number):
            return X
        LUT = dict(zip(range(len(BEHAVIORS)), BEHAVIORS))
        character_labels = numpy.asarray([LUT[int(b)] for b in tmp])
        if isinstance(X, DataSet.DataSet):
            X.labels = character_labels
        else:
            X = character_labels
        return X

    @staticmethod
    def verify(true_labels, predicted_labels, do_plot: bool=True, outfile=None):
        ''' Compare true labels to predicted labels
        '''
        AbstractBehaviorPredictor.print_confusion(true_labels, predicted_labels)
        if do_plot:
            AbstractBehaviorPredictor.plot_confusion(
                true_labels,
                predicted_labels,
                'Behavior Prediction Confusion',
                outfile)

    @staticmethod
    def print_confusion(labels, predicted_labels):
        ''' Prints true and false positive rates for the predictions.
        '''
        labels_enum = list(set(numpy.unique(labels)).union(numpy.unique(predicted_labels)))
        confusion_matrix = AbstractBehaviorPredictor.get_confusion_matrix(labels, predicted_labels, labels_enum)
        total = confusion_matrix.sum()
        correct = numpy.trace(confusion_matrix)
        if total > 0:
            print(f'\nTotal Accuracy: {100 * correct / total:.2f}%')
        print('% true positive for behavior B is (# correctly labeled as B) / (# total true B)')
        print('% false positive for behavior B is (# total predicted B - # correctly labeled as B) / (# total predicted B)')
        print('Behavior\t%true positive\t%false positive')
        for i, behavior_name in enumerate(labels_enum):
            class_correct = confusion_matrix[i, i]
            class_total = confusion_matrix[i, :].sum()
            class_accuracy = 0 if class_total == 0 else 100 * class_correct / class_total
            predicted_total = confusion_matrix[:, i].sum()
            percent_false_positive = 0 if predicted_total == 0 else 100 * (predicted_total - class_correct) / predicted_total
            print(f'{behavior_name}\t{class_accuracy:.2f}%\t{percent_false_positive:.2f}%')
        print('')

    @staticmethod
    def plot_confusion(labels, predicted_labels, name, outfile=None):
        from matplotlib import pyplot as plt
        labels_enum = list(set(numpy.unique(labels)).union(numpy.unique(predicted_labels)))
        labels_enum = sorted(labels_enum, key=lambda x: x == 'U')
        confusion_matrix = AbstractBehaviorPredictor.get_confusion_matrix(labels, predicted_labels, labels_enum)
        fig, ax = plt.subplots()
        plt.title(name)
        plt.imshow(confusion_matrix, cmap='copper')
        labels_enum_names = list(map(lambda x: BEHAVIORS_NAMES[x], labels_enum))
        plt.xticks(ticks=range(len(labels_enum)), labels=labels_enum_names)
        plt.yticks(ticks=range(len(labels_enum)), labels=labels_enum_names)
        ax.xaxis.tick_top()
        for i in range(len(labels_enum)):
            for j in range(len(labels_enum)):
                ax.text(j, i, confusion_matrix[i, j],
                    ha="center", va="center", color="w")
        if outfile is not None:
            fig.savefig(outfile)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def get_confusion_matrix(expected, predicted, labels_enum):
        ''' Returns matrix where row is the expected, column is predicted.

        For example, row 5 contains all inputs with label 5.
        Column 5 contains everything that was predicted to be 5.
        '''
        confusion_matrix = numpy.zeros((len(labels_enum), len(labels_enum)), dtype=int)
        b_to_i = {
            b: labels_enum.index(b) for b in labels_enum
        }
        for e, p in zip(expected, predicted):
            confusion_matrix[b_to_i[e], b_to_i[p]] += 1
        return confusion_matrix

