import warnings
import traceback
import numpy
from sklearn import ensemble

from swimfunction import FileLocations
from swimfunction.data_access import BehaviorAnnotationDataManager as behavior_data_manager
from swimfunction.data_models import WindowsDataSet, DataSet
from swimfunction.behavior_annotation.AbstractBehaviorPredictor import AbstractBehaviorPredictor, BEHAVIORS, plot_with_labels
from swimfunction.global_config.config import config

PPW = config.getint('BEHAVIOR_ANNOTATION', 'poses_per_window')

class UmapClassifier(AbstractBehaviorPredictor):
    __slots__ = ['umap_mapper', 'classifying_model']
    def __init__(self):
        super().__init__()
        self.umap_mapper = None
        self.classifying_model = None

    def load_models(self):
        self.classifying_model = behavior_data_manager.load_classifier()
        self.umap_mapper = behavior_data_manager.load_umap()

    def train_umap(self, training_data: DataSet, save=True, force_retrain=False):
        if self.umap_mapper is None or force_retrain:
            warnings.filterwarnings('ignore', category=FutureWarning) # ignore UMAP internal warnings
            import umap # because it takes a long time to import, I only import it when absolutely necessary.
            self.umap_mapper = umap.UMAP(n_neighbors=50, n_components=2, metric='euclidean')
            try:
                print(f'Training umap with {len(training_data.labels)} labels')
                st_umap_transformed = self.umap_mapper.fit_transform(training_data.data)
                plot_with_labels(
                    st_umap_transformed,
                    self.labels_from_numeric(training_data.labels),
                    FileLocations.get_behaviors_model_dir() / 'train_set_umap_transformed.png')
            except TypeError as e:
                print(e)
                print(traceback.format_exc())
                print('If you get "TypeError: a bytes-like object is required,\
                    not \'list\'", then you probably need to install pynndescent')
            if save:
                behavior_data_manager.save_umap(self.umap_mapper)

    def train_classifier(self, training_data: DataSet, save=True, force_retrain=False):
        if self.classifying_model is None or force_retrain:
            print(f'Training classifier with {len(training_data.labels)} labels')
            self.classifying_model = ensemble.RandomForestClassifier(min_samples_leaf=5)
            self.classifying_model.fit(
                self.umap_mapper.transform(training_data.data),
                training_data.labels)
            if save:
                behavior_data_manager.save_classifier(self.classifying_model)

    # Implementing abstract methods
    def train_models(self,
        feature: str='smoothed_angles',
        labels_to_predict: list=None,
        percent_test: int=25,
        enforce_equal_representations: bool=True,
        use_cached_training_data=True,
        save: bool=True,
        do_plot: bool=True):
        ''' Trains and cross-validates

        Parameters
        ----------
        feature : str, default='smoothed_complete_angles'
            feature to use as data
        labels_to_predict : list, default=None
            labels that should be predicted. Other labels are masked to Unknown.
        percent_test : int, default=25
            integer percent of dataset to reserve for testing.
        enforce_equal_representations : bool, default=True
            whether each label should be represented equally
        use_cached_training_data : bool, default=True
            whether to keep existing models and train/test datasets
        save : bool, default=True
            whether to save the models and train/test datasets
        do_plot : bool, default=True
            whether to show confusion matrix plot
        '''
        if labels_to_predict is None:
            labels_to_predict = list(BEHAVIORS)
        if use_cached_training_data:
            self.load_models()
        original_train, original_test = self.get_raw_training_data(
            percent_test, use_cached_training_data, feature, save)
        behavior_data_manager.report_train_test_summary(original_train, original_test)
        train, test = self.get_processed_training_data(
            original_train, original_test, feature,
            labels_to_predict, enforce_equal_representations, use_cached_training_data,
            save, max_size_for_subsampling=100000)
        # Only train if the model is none, or if we ask it to retrain.
        if not use_cached_training_data or self.umap_mapper is None or self.classifying_model is None:
            self.train_umap(train, save, force_retrain=False)
            self.train_classifier(train, save, force_retrain=False)
        print('Evaluating against the raw test set')
        predicted_test_labels = self.evaluate(
            original_test,
            do_plot=do_plot,
            outfile=FileLocations.get_behaviors_model_dir() / 'test_set_confusion.png')
        print('Evaluating against the raw training set')
        predicted_train_labels = self.evaluate(
            original_train,
            do_plot=do_plot,
            outfile=FileLocations.get_behaviors_model_dir() / 'train_set_confusion.png')
        if save:
            plot_with_labels(
                self.umap_mapper.transform(test.data),
                self.labels_from_numeric(test.labels),
                FileLocations.get_behaviors_model_dir() / 'processed_test_set_umap_transformed.png')
            plot_with_labels(
                self.umap_mapper.transform(original_test.data),
                predicted_test_labels,
                FileLocations.get_behaviors_model_dir() / 'test_set_predicted.png')
            plot_with_labels(
                self.umap_mapper.transform(original_train.data),
                predicted_train_labels,
                FileLocations.get_behaviors_model_dir() / 'train_set_predicted.png')
        print('Training complete!')
        return True

    def classify_windows(self, windows: numpy.ndarray):
        self.confirm_is_pose_window_list(windows)
        return self.labels_from_numeric(self.classifying_model.predict(
            self.umap_mapper.transform(windows)))

    def predict_behaviors(self, sequential_poses: numpy.ndarray):
        ''' Given poses, predict the behavior.
        Internally, poses are concatenated into pose windows.
        Since we want all pose-window stuff to be handled internally,
        the inputs to this function are simply poses.

        Parameters
        ----------
        poses : numpy.ndarray
            NxW array, N poses, each pose is length W

        Returns
        -------
        numpy.ndarray of behaviors (same length as poses)
        '''
        self.confirm_is_angle_pose_list(sequential_poses)
        return self._buffer_predictions(
            self._windows_to_predictions(
                WindowsDataSet.WindowsDataSet().from_poses(sequential_poses)
            )
        )

    def _buffer_predictions(self, predictions, buffer_val=BEHAVIORS.unknown):
        ''' Because windows are sliding, cannot predict the first and last frames,
        this function adds numpy.nan to the beginning and end of predictions so that it is the same length as frames.

        Parameters
        ----------
        predictions : list or numpy.ndarray
            list of predicted behaviors

        Returns
        -------
        numpy.ndarray
        '''
        buffer = [buffer_val for _ in range(PPW//2)]
        return numpy.concatenate((buffer, predictions, buffer))

    def _windows_to_predictions(self, PW):
        ''' Uses an SVM on UMAP-transformed windows to predict behaviors.

        Parameters
        ----------
        PW : WindowsDataSet

        Returns
        -------
        numpy.ndarray
            list of behaviors
        '''
        predictions = numpy.full(PW.data.shape[0], BEHAVIORS.unknown)
        valid_window_locations = numpy.all(numpy.logical_not(numpy.isnan(PW.data)), axis=1)
        if len(PW.data[valid_window_locations]) > 0:
            predictions[valid_window_locations] = self.classify_windows(PW.data[valid_window_locations])
        return predictions

    def labels_from_numeric(self, X):
        return super().labels_from_numeric(X)
