''' After setting up the directory structure
and populating it with normalized videos,
then you can create a training set.
'''
from swimfunction.pose_annotation import extract_training_frames

from swimfunction import FileLocations
import pathlib

from swimfunction.pose_annotation.human_annotation import annotate_with_elegant

def main(videos_dir, training_root_dir, scorer):
    extract_training_frames.extract_training_frames(
        videos_dir,
        training_root_dir,
        total_frame_output=10,
        split_videos=False)
    pa = annotate_with_elegant.PoseAnnotator(
        pathlib.Path(training_root_dir).expanduser().resolve(),
        scorer)
    pa.run()

if __name__ == '__main__':
    ARGS = FileLocations.parse_default_args(
        lambda p: p.add_argument('scorer', help='Your name (the name of the person annotating the poses).')
    )
    main(FileLocations.get_normalized_videos_dir(),
        FileLocations.get_training_root_dir(),
        ARGS.scorer)
