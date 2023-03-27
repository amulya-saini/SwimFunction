''' Annotate poses in one or all videos using skeletonization.
'''
import pathlib

from swimfunction.context_managers.FileLock import FileLock
from swimfunction.pose_annotation import get_skeleton_pose
from swimfunction.video_processing import fp_ffmpeg
from swimfunction import FileLocations
from swimfunction import progress

# Example DLC output CSV format given below:
# scorer,DLC_resnet50_tracked_fishDec16shuffle1_180000,...
# bodyparts,p0,p0,p0,p1,p1,p1
# coords,x,y,likelihood,x,y,likelihood
# 0,24.581830978393555,197.81887817382812,1.0,27.444324493408203,187.75238037109375,1.0

def annotate_video(vfile, output_path):
    ''' Skeletonizes one video.
    '''
    poses = []
    for frame in fp_ffmpeg.read_video(vfile):
        poses.append(get_skeleton_pose.frame_to_skeleton_points(frame))
    ppp = poses[0].shape[0] #points per pose
    with open(output_path, 'wt') as writer:
        scorers = ','.join(['skeletonization' for _ in range(3*ppp)])
        writer.write(f'scorer,{scorers}\n')
        bodyparts = ','.join([f'p{i},p{i},p{i}' for i in range(ppp)])
        writer.write(f'bodyparts,{bodyparts}\n')
        keypoints = ','.join(['x,y,likelihood' for _ in range(ppp)])
        writer.write(f'coords,{keypoints}\n')
        for i, pose in enumerate(poses):
            parts = [f'{p[0]},{p[1]},nan' for p in pose]
            writer.write(f'{i},{",".join(parts)}\n')

def annotate_all_videos(videos_root, output_dir):
    ''' Skeletonizes all videos.
    '''
    videos_root = pathlib.Path(videos_root)
    output_dir = pathlib.Path(output_dir)
    video_files = list(FileLocations.find_video_files(root_path=videos_root))
    for vfile in video_files:
        progress.init(len(video_files))
        for i, fname in enumerate(video_files):
            fname = pathlib.Path(fname)
            progress.progress(i, fname.name)
            output_path = output_dir /  f'{fname.name}_skeletonized.csv'
            with FileLock(
                    lock_dir=output_dir,
                    lock_name=fname.name,
                    output_path=output_path) as fl:
                if fl:
                    annotate_video(vfile, output_path)
