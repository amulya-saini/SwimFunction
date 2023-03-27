''' If you have annotations for the body keypoints
separate from the caudal fin tail keypoint,
this script will combine matching pairs into one dataframe.
'''

import pandas
import pathlib
from tqdm import tqdm

from swimfunction import FileLocations
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_access import data_utils

class AnnotationCombiner:
    ''' I trained two DLC models:
    1) ten keypoints on the dorsal centerline
    2) caudal tail tip

    We must combine the outputs into one file to be input into swimfunction.
    '''
    def __init__(
            self,
            dlc_body_annot_dir: pathlib.Path,
            dlc_caudal_annot_dir: pathlib.Path,
            combined_output_dir: pathlib.Path):
        self.dlc_body_annot_dir = FileLocations.as_absolute_path(dlc_body_annot_dir)
        self.dlc_caudal_annot_dir = FileLocations.as_absolute_path(dlc_caudal_annot_dir)
        self.combined_output_dir = FileLocations.as_absolute_path(combined_output_dir)

    @staticmethod
    def write_df(combined_df: pandas.DataFrame, fpath):
        ''' Write DataFrame to file
        '''
        if fpath.suffix == '.csv':
            combined_df.to_csv(fpath.with_suffix('.csv'), mode='w')
        elif fpath.suffix == '.h5':
            combined_df.to_hdf(
                fpath.with_suffix('.h5'),
                key='df_with_missing',
                format='table',
                mode='w')

    @staticmethod
    def get_npoints_in_df(df: pandas.DataFrame):
        ''' The number of annotated keypoints
        '''
        return len(df.columns.get_level_values(-2).unique())

    @staticmethod
    def combine_dfs(body_df: pandas.DataFrame, caudal_fin_df: pandas.DataFrame):
        ''' Join the DataFrame keypoints with body on left, tail on right.
        '''
        num_p_in_body = AnnotationCombiner.get_npoints_in_df(body_df)
        num_p_in_tail = AnnotationCombiner.get_npoints_in_df(caudal_fin_df)
        if num_p_in_body != 10 or num_p_in_tail != 1:
                raise RuntimeError(' '.join([
                    f'Body dataframe has {num_p_in_body} keypoints, must have exactly 10.',
                    f'Caudal fin dataframe has {num_p_in_tail} keypoints, must have exactly 1.',
                    'Please check that the input directories contain only body and caudal files',
                    'and that they were typed in the correct order.',
                    'Use the -h flag for help.']))
        combined = body_df.join(caudal_fin_df)
        ctuples = []
        for c in combined.columns:
            ctuples.append(('animal', *c[1:]))
        combined.columns = pandas.MultiIndex.from_tuples(ctuples)
        return combined

    def find_all_body_annotation_files(self):
        ''' Locate all body annotation files.
        '''
        return list(self.dlc_body_annot_dir.glob('*.csv')) \
            + list(self.dlc_body_annot_dir.glob('*.h5'))

    def find_all_tail_annotation_files(self):
        ''' Locate all tail annotation files.
        '''
        return list(self.dlc_caudal_annot_dir.glob('*.csv')) \
            + list(self.dlc_caudal_annot_dir.glob('*.h5'))

    def get_files_to_combine(self):
        ''' Get pairs of files that can be combined.
        '''
        files_to_combine = {}
        for fpath in self.find_all_body_annotation_files():
            fd = data_utils.parse_details_from_filename(fpath)[0]
            fish = f'{fd.name}{fd.side}'
            k = (fish, fd.assay_label)
            files_to_combine[k] = [fpath, None]
        for fpath in self.find_all_tail_annotation_files():
            fd = data_utils.parse_details_from_filename(fpath)[0]
            fish = f'{fd.name}{fd.side}'
            k = (fish, fd.assay_label)
            if k in files_to_combine:
                files_to_combine[k][1] = fpath
        files_to_combine = {
            k: v for k, v in files_to_combine.items() if v[0] is not None and v[1] is not None
        }
        return files_to_combine

    def combine_body_caudal_dlc_outputs(self):
        ''' Combine all annotations.
        '''
        files_to_combine = self.get_files_to_combine()
        for (fish, assay), (bodyf, caudalf) in tqdm(list(files_to_combine.items())):
            df1 = FDM._dlc_output_to_dataframe(bodyf)
            df2 = FDM._dlc_output_to_dataframe(caudalf)
            combined_df = self.combine_dfs(df1, df2)
            outfpath = self.combined_output_dir / f'dlc_body_and_caudal_fin_{assay}wpi_{fish}.h5'
            self.write_df(combined_df, outfpath)

if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('dlc_body_annot_dir')
    PARSER.add_argument('dlc_caudal_annot_dir')
    PARSER.add_argument('combined_output_dir')
    ARGS = PARSER.parse_args()
    AnnotationCombiner(
        ARGS.dlc_body_annot_dir,
        ARGS.dlc_caudal_annot_dir,
        ARGS.combined_output_dir).combine_body_caudal_dlc_outputs()
