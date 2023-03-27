from tqdm import tqdm
import numpy

from swimfunction.data_access \
    import BehaviorAnnotationDataManager as behavior_data_manager, PoseAccess
from swimfunction.behavior_annotation.RestPredictor import RestPredictor
from swimfunction.behavior_annotation.UmapClassifier import UmapClassifier
from swimfunction.context_managers.AccessContext import AccessContext
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.Fish import Fish
from swimfunction.global_config.config import config
from swimfunction.pose_processing import pose_filters
from swimfunction import FileLocations

PREDICT_FEATURE = config.get('MACHINE_LEARNING', 'predict_feature')
BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')

PERCENT_TEST = 25

ENFORCE_EQUAL_LABEL_REPRESENTATION = True

def train_umap_classifier_predictor():
    ''' Train the UMAP + SVM behavior classifier
    '''
    USE_CACHED_TRAINING_DATA = True
    SAVE_MODELS = True
    BP = UmapClassifier()
    print('Training models')
    BP.train_models(
        feature=PREDICT_FEATURE,
        labels_to_predict=[
            BEHAVIORS.cruise,
            BEHAVIORS.rest,
            BEHAVIORS.turn_ccw,
            BEHAVIORS.turn_cw],
        percent_test=PERCENT_TEST,
        enforce_equal_representations=ENFORCE_EQUAL_LABEL_REPRESENTATION,
        use_cached_training_data=USE_CACHED_TRAINING_DATA,
        save=SAVE_MODELS,
        do_plot=True)

def train_rest_predictor():
    ''' Train the rest predictor
    '''
    USE_CACHED_TRAINING_DATA = True
    BP = RestPredictor(PREDICT_FEATURE)
    print('Training rest model')
    BP.train_models(
        feature=PREDICT_FEATURE,
        labels_to_predict=[BEHAVIORS.rest],
        percent_test=PERCENT_TEST,
        enforce_equal_representations=ENFORCE_EQUAL_LABEL_REPRESENTATION,
        use_cached_training_data=USE_CACHED_TRAINING_DATA,
        save=False,
        do_plot=True)
    print(f'Rest epsilon: {BP.rest_epsilon}')

def classify_behaviors_from_sequential_poses(
        sequential_poses: numpy.ndarray,
        ml_predictor,
        rest_predictor):
    ''' Predict the behavior of sequential poses
    (our PREDICT_FEATURE was smoothed_angles, meaning angle poses)
    '''
    behaviors = ml_predictor.predict_behaviors(sequential_poses)
    rest_mask = rest_predictor.find_rests_angles(sequential_poses)
    behaviors[numpy.where(rest_mask)] = BEHAVIORS.rest
    return behaviors

def predict_rests_and_cruises_all_experiments():
    ''' Predict behaviors for all experiments
    '''
    rest_predictor = RestPredictor(PREDICT_FEATURE)
    ml_predictor = UmapClassifier()
    ml_predictor.load_models()
    experiments = config.getstringlist('EXPERIMENT DETAILS', 'names')
    print('Predicting all rests and cruises...')
    for experiment in experiments:
        with AccessContext(experiment_name=experiment):
            print('\t', experiment)
            for fish_name in tqdm(FDM.get_available_fish_names()):
                fish = Fish(name=fish_name).load()
                for assay in tqdm(FDM.get_available_assay_labels(fish_name), leave=False):
                    sequential_poses = PoseAccess.get_feature_from_assay(
                        fish[assay], PREDICT_FEATURE, pose_filters.BASIC_FILTERS, True)
                    fish[assay].predicted_behaviors = classify_behaviors_from_sequential_poses(
                        sequential_poses,
                        ml_predictor,
                        rest_predictor)
                fish.save()
    print('Done!')

def evaluate_models():
    ''' Perform a simple evaluation of the models.
    '''
    print('Loading models...')
    RP = RestPredictor(PREDICT_FEATURE)
    CP = UmapClassifier()
    CP.load_models()

    print('Loading data...')
    train, test = behavior_data_manager.get_data_for_training(
        percent_test=PERCENT_TEST,
        use_cached_training_data=True,
        feature=PREDICT_FEATURE)

    desired_behaviors = (BEHAVIORS.rest, BEHAVIORS.cruise, BEHAVIORS.turn_cw, BEHAVIORS.turn_ccw)
    train = RP.mask_unwanted_behaviors(train, desired_behaviors)
    test = RP.mask_unwanted_behaviors(test, desired_behaviors)


    print('Classifying training data...')
    train_b = CP.classify_windows(train.data)
    train_b[RP.classify_windows(train.data) == BEHAVIORS.rest] = BEHAVIORS.rest

    print('Classifying test data...')
    test_b = CP.classify_windows(test.data)
    test_b[RP.classify_windows(test.data) == BEHAVIORS.rest] = BEHAVIORS.rest

    print('Reporting results...')
    CP.verify(
        train.labels, train_b, True,
        FileLocations.get_behaviors_model_dir() / 'train_confusion_umap_with_rest_thresholded.png')
    CP.verify(
        test.labels, test_b, True,
        FileLocations.get_behaviors_model_dir() / 'test_confusion_umap_with_rest_thresholded.png')
    print('Evaluation complete.')

if __name__ == '__main__':
    FileLocations.parse_default_args()
    ## Uncomment below if you want to retrain.
    ##   Be sure to back up your old models
    ##       in case you need to revert!
    # train_umap_classifier_predictor()
    # train_rest_predictor()
    evaluate_models()
    # predict_rests_and_cruises_all_experiments()
