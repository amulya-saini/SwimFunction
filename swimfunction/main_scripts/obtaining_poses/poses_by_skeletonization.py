'''
IMPORTANT: You must run `preprocess_videos.py` before running this script.
Gets poses by skeletonization.
'''
from swimfunction.pose_annotation import skeleton_pose_annotate
from swimfunction import FileLocations

def skeletonize_videos():
    ''' Gets skeleton poses for all frames in all videos.
    '''
    skeleton_pose_annotate.annotate_all_videos(
        FileLocations.get_normalized_videos_dir(),
        FileLocations.mkdir_and_return(FileLocations.get_dlc_body_outputs_dir()))

if __name__ == '__main__':
    FileLocations.parse_default_args()
    skeletonize_videos()
