from swimfunction.pose_annotation import get_skeleton_pose
from swimfunction.pose_annotation.human_annotation import annotations_to_csv_h5
from swimfunction.video_processing import fp_ffmpeg
from zplib.curve import interpolate

import glob
import numpy
import pathlib
import pickle

def refine_skeleton_with_keypoints(skeleton_points, keypoints):
    skeleton_points = numpy.asarray(skeleton_points)
    keypoints = numpy.asarray(keypoints)
    head_distances = numpy.linalg.norm(skeleton_points - keypoints[0, :], axis=1)
    tail_distances = numpy.linalg.norm(skeleton_points - keypoints[-1, :], axis=1)
    head_i = head_distances.argmin()
    tail_i = tail_distances.argmin()
    if head_i < tail_i:
        skeleton_points = skeleton_points[head_i:tail_i, :]
    else:
        skeleton_points = skeleton_points[tail_i:head_i:-1, :]
    skeleton_points = numpy.concatenate(([keypoints[0, :]], skeleton_points, [keypoints[-1, :]]))
    return skeleton_points

def get_refined_skeleton(img, head_pt, tail_pt):
    ''' Intended as a helpful wrapper.
    '''
    if isinstance(img, (str)) or isinstance(img, (pathlib.Path)):
        img = fp_ffmpeg.freeimage_read(pathlib.Path(img).expanduser().as_posix(), transpose_to_match_convention=True)
    return refine_skeleton_with_keypoints(get_skeleton_pose.frame_to_skeleton_points(img), [head_pt, tail_pt])

def get_width_tck(ann):
    choices = [ann[1][k]['pose'][1] for k in ann[1].keys()]
    choices = [x for x in choices if x is not None]
    if len(choices) == 0:
        return None
    return choices[0]

def make_csv(train_root, scorer):
    train_root = pathlib.Path(train_root).expanduser()
    with open(train_root / 'annotations' / '00.pickle', 'rb') as fh:
        ann = pickle.load(fh)
    with open(train_root / 'annotations' / '00_old.pickle', 'wb') as fh:
        pickle.dump(ann, fh)
    
    image_fnames = glob.glob((train_root / '*.png').as_posix())
    width_tck = get_width_tck(ann)
    for img_f in image_fnames:
        img_f = pathlib.Path(img_f).expanduser()
        img_k = img_f.name.split(' ')[0].split('.png')[0]
        head = ann[1][img_k]['keypoints']['head']
        tail = ann[1][img_k]['keypoints']['tail']
        refined = get_refined_skeleton(img_f, head, tail)
        center_tck = interpolate.fit_spline(refined)
        ann[1][img_k]['pose'] = (center_tck, width_tck)
        
    with open(train_root / 'annotations' / '00.pickle','wb') as fh:
        ann = pickle.dump(ann, fh)
    
    annotations_to_csv_h5.main(
        train_root / 'annotations' / '00.pickle',
        scorer=scorer,
        npoints=10,
        output_dir=train_root)

if __name__=='__main__':
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('train_root', help='Location of training images.')
    PARSER.add_argument('scorer', help='The scorer (should be same scorer in your DeepLabCut config).')
    ARGS = PARSER.parse_args()
    make_csv(ARGS.train_root, ARGS.scorer)

