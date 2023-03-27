import traceback
from matplotlib import pyplot as plt
from tqdm import tqdm

from swimfunction.pose_processing import pose_filters
from swimfunction.global_config.config import config
from swimfunction.data_access import PoseAccess
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.Fish import Fish
from swimfunction.qc_scripts import pose_qc, behavior_qc
from swimfunction import loggers
from swimfunction import FileLocations
from swimfunction.main_scripts.metrics import \
    populate_data_storage_structures, predict_behaviors_as_required

def get_logger() -> loggers.logging.Logger:
    ''' Get the default metric calculation logger.
    '''
    return loggers.get_metric_calculation_logger(__name__)

def header(msg):
    ''' Log the message loud and clear.
    '''
    get_logger().info('\n------- %s -------\n', msg)

def print_experiment_stats():
    ''' Report key numbers about fish, frames, and poses.
    '''
    header('Getting experiment stats')
    nfish = len(FDM.get_available_fish_names())
    nassays = 0
    nframes = 0
    nquality_poses = 0
    for name in tqdm(FDM.get_available_fish_names()):
        fish = Fish(name).load()
        for assay in fish.swim_keys():
            nassays += 1
            nframes += fish[assay].raw_coordinates.shape[0]
            nquality_poses += PoseAccess.get_feature_from_assay(
                fish[assay],
                'smoothed_coordinates',
                filters=pose_filters.BASIC_FILTERS,
                keep_shape=False).shape[0]
    get_logger().info(config.experiment_name)
    get_logger().info('\t%d fish', nfish)
    get_logger().info('\t%d total individual swims', nassays)
    get_logger().info('\t%d total frames', nframes)
    get_logger().info('\t%d total high quality poses', nquality_poses)

def qc_main():
    # Import all data into a useable structure.
    populate_data_storage_structures()

    print_experiment_stats()

    # pose quality control and comparison
    header('Pose QC')
    pose_qc.qc(overwrite_existing=True)

    predict_behaviors_as_required()

    header('Behavior QC')
    behavior_qc.qc(overwrite_existing=True)

if __name__ == '__main__':
    plt.switch_backend('agg') # No gui will be created. Safer for distributed processing.
    ARGS = FileLocations.parse_default_args()
    try:
        qc_main()
    except Exception as _E:
        get_logger().error('Exception occurred in %s', __name__)
        get_logger().error(_E)
        get_logger().error(traceback.format_exc())
