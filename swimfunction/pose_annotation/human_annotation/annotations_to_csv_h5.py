''' Takes annotations and creates CSV and H5 files.
Annotations format:
{
    IMG.png: [(center_tck, width_tck), (center_tck, width_tck), ...],
    IMG2.png: [(center_tck, width_tck)]
}

Creates CSV like this:

    scorer,Jensen,Jensen,Jensen,Jensen, ...
    bodyparts,p0,p0,p1,p1, ...
    coords,x,y,x,y, ...
    labeled-data/train_images/img00084.png,578.9,1196.4,574.8,1173.3, ...

or like this for multi-animal:

    scorer,Jensen,Jensen,Jensen,Jensen, ...
    individual,individual1,individual1,individual1,individual1, ...
    bodyparts,p0,p0,p1,p1, ...
    coords,x,y,x,y, ...
    labeled-data/train_images/img00084.png,578.9,1196.4,574.8,1173.3, ...

'''

from collections import defaultdict
from swimfunction.context_managers.CacheContext import CacheContext
from zplib.curve import interpolate
import argparse
import numpy
import pathlib
import pandas

def img_file_to_label(img_file, folder_name):
    if folder_name is None:
        return f'labeled-data/{pathlib.Path(img_file).name}'
    return f'labeled-data/{folder_name}/{pathlib.Path(img_file).name}'

def get_data_with_rownames(annotations, npoints, folder_name):
    data = []
    row_names = []
    for fpath, poses in annotations.items():
        row_names.append(img_file_to_label(fpath, folder_name))
        row = numpy.full(2 * npoints * len(poses), numpy.nan, dtype=float)
        for i, (center_tck, _width_tck) in enumerate(poses):
            if center_tck is not None:
                pts = interpolate.spline_interpolate(center_tck, npoints)
                start = i * (2 * npoints)
                row[start:start+(2*npoints):2] = pts[:, 0]
                row[start+1:start+(2*npoints):2] = pts[:, 1]
        data.append(row)
    return data, row_names

def make_dataframe(annotations, scorer, npoints, folder_name):
    ''' Converts annotations from Elegant into a pandas DataFrame
    '''
    data, row_names = get_data_with_rownames(annotations, npoints, folder_name)
    bodyparts = [f'p{i}' for i in range(npoints)]
    num_animals = 1
    pose_counts = [len(poses) for poses in annotations.values()]
    if len(pose_counts) > 0:
        num_animals = max(pose_counts)
    if num_animals == 1:
        columns_multi_index = pandas.MultiIndex.from_product(
            [[scorer], bodyparts, ['x', 'y']],
            names=['scorer', 'bodyparts', 'coords'])
    else:
        individuals = [f'individual{i+1}' for i in range(num_animals)]
        columns_multi_index = pandas.MultiIndex.from_product(
            [[scorer], individuals, bodyparts, ['x', 'y']],
            names=['scorer', 'individuals', 'bodyparts', 'coords'])
    df = pandas.DataFrame(data, columns=columns_multi_index, index=row_names)
    return df

def trim_poseless(annotations):
    ''' Returns a new annotations dictionary
    where every item has at least one pose.
    Does not modify the original annotations object.
    '''
    keys = list(annotations.keys())
    rv = {}
    for k in keys:
        has_pose = numpy.any([p[0] is not None for p in annotations[k]])
        if has_pose:
            rv[k] = annotations[k]
    return rv

def save_dlc_df(df: pandas.DataFrame, filepath: pathlib.Path):
    '''
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to save as csv and h5
    filepath : pathlib.Path
        Path including basename. Basename can have any suffix.
        Suffix will be replaced with .csv and .h5 as needed.
    '''
    print('Writing csv...')
    df.to_csv(filepath.with_suffix('.csv'), mode='w')
    print('Writing h5...')
    df.to_hdf(filepath.with_suffix('.h5'), key='df_with_missing', format='table', mode='w')
    print('Done!')

def annotations_to_csv_h5(
        annotations,
        scorer,
        npoints,
        output_dir,
        ignore_poseless_files=True):
    ''' Combines annotations into a csv file that DeepLabCut can use for training.
    '''
    output_dir = pathlib.Path(output_dir)
    folder_name = output_dir.name
    if ignore_poseless_files:
        annotations = trim_poseless(annotations)
    df = make_dataframe(annotations, scorer, npoints, folder_name)
    save_dlc_df(df, output_dir / f'CollectedData_{scorer}.csv')

def detect_headers(csv_fname):
    header = []
    valid_header_keys = ['scorer', 'individuals', 'bodyparts', 'coords']
    with open(csv_fname, 'rt') as fh:
        for i, l in enumerate(fh):
            if l.split(',')[0] in valid_header_keys:
                header.append(i)
            else:
                break
    if len(header) > len(valid_header_keys):
        raise ValueError('Header line number could not be determined.')
    return header

def replace_scorer(h5_fname, scorer):
    ''' Replaces old scorer with new scorer. Overwrites both csv and h5 files.
    '''
    fpath = pathlib.Path(h5_fname)
    df = pandas.read_hdf(h5_fname)
    if df.columns.levels[0][0] == scorer:
        print('Nothing to do. Already correct.')
        return
    ofstem = fpath.stem.replace(df.columns.levels[0][0], scorer)
    df.columns = df.columns.set_levels(df.columns.levels[0].str.replace(df.columns.levels[0][0], scorer), level=0)
    print('Writing csv...')
    df.to_csv(fpath.parent / f'{ofstem}.csv', mode='w')
    print('Writing h5...')
    df.to_hdf(fpath.parent / f'{ofstem}.h5', key='df_with_missing', format='table', mode='w')
    print('Done!')

def dataframe_to_pose_map(df: pandas.DataFrame):
    ''' Converts dataframe into a dictionary like { 'image_relative_path': [list, of, poses] }
    '''
    points_per_pose = len(df.columns.levels[-2])
    # NOTE: indexing the DataFrame is in this format:
    #        dataframe.loc[frame_numbers, idx[individuals, bodyparts, x_y_likelihood]]
    # Then, use .values.tolist() for efficient transfer to numpy array (no copy)
    idx = pandas.IndexSlice
    scorer = df.columns[0][0]
    annotations = defaultdict(list)
    for fpath in df.index:
        df[scorer].loc[fpath]
        for individual in pandas.unique(df.columns.get_level_values('individuals')):
            raw_coordinates = numpy.empty((points_per_pose, 2))
            raw_coordinates[:, 0] = numpy.array(df.loc[fpath, idx[scorer, individual, :, 'x']].values.tolist())
            raw_coordinates[:, 1] = numpy.array(df.loc[fpath, idx[scorer, individual, :, 'y']].values.tolist())
            if not numpy.all(numpy.isnan(raw_coordinates)):
                annotations[fpath].append(raw_coordinates)
    return annotations

def read_csv_to_multianimal_dataframe(csv_h5_fname):
    ''' Takes a single-animal csv or h5 file (does not have the "individuals" header line)
            and makes it multi-animal compatible (creates both CSV and H5 files).
            After running this, confirm it's correct and change their name
            by removing the "multianimal_" prefix.
    '''
    extension = pathlib.Path(csv_h5_fname).suffix
    df = None
    if extension == '.csv':
        header = detect_headers(csv_h5_fname)
        df = pandas.read_csv(csv_h5_fname, header=header, index_col=[0])
    elif extension == '.h5':
        df = pandas.read_hdf(csv_h5_fname)
    else:
        raise ValueError('Extension must be .csv or .h5')
    if len(df.columns.levels) == 4:
        return df
    individuals = ['individual1']
    npoints = len(df.columns) // 2
    bodyparts = [f'p{i}' for i in range(npoints)]
    scorer_arr = [df.columns[0][0] ]
    columns_multi_index = pandas.MultiIndex.from_product(
            [scorer_arr, individuals, bodyparts, ['x', 'y']],
            names=['scorer', 'individuals', 'bodyparts', 'coords'])
    df2 = pandas.DataFrame(df.values, columns=columns_multi_index, index=df.index)
    return df2

def main(annotations_fname, scorer, npoints, output_dir):
    ann = None
    print('Reading annotations...')
    with CacheContext(annotations_fname) as cache:
        ann = cache.getContents()
    annotations_to_csv_h5(ann, scorer, npoints, output_dir)
    return True

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='location of annotations file (probably called annotations.pickle)')
    parser.add_argument('scorer', help='name of scorer, or training set annotator, (required for DeepLabCut) example: Jensen')
    parser.add_argument('output_dir', help='directory to save the output file called CollectedData_{scorer}.csv')
    parser.add_argument('-n', '--npoints', type=int, help='number of points along centerline to sample, default: 10', default=10)
    args = parser.parse_args()
    main(args.fname, args.scorer, args.npoints, args.output_dir)
