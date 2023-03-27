import numpy

from swimfunction.behavior_annotation.RestPredictor import RestPredictor
from swimfunction.behavior_annotation.UmapClassifier import UmapClassifier, BEHAVIORS

_ML_BEHAVIOR_CLASSIFIER = UmapClassifier()

def classify_behaviors_from_sequential_poses(sequential_poses: numpy.ndarray):
    if _ML_BEHAVIOR_CLASSIFIER.classifying_model is None:
        _ML_BEHAVIOR_CLASSIFIER.load_models()
    behaviors = _ML_BEHAVIOR_CLASSIFIER.predict_behaviors(sequential_poses)
    rest_mask = RestPredictor().find_rests_angles(sequential_poses)
    behaviors[numpy.where(rest_mask)] = BEHAVIORS.rest
    return behaviors

