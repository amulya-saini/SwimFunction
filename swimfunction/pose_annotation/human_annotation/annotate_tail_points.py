''' After creating a training set for centerline,
this script can help annotate tail points for another DLC model.
'''
from matplotlib import pyplot as plt
from swimfunction.pose_annotation.human_annotation.annotations_to_csv_h5 import save_dlc_df
from swimfunction.video_processing import fp_ffmpeg

from swimfunction import FileLocations
import numpy
import pathlib
import pandas
from tqdm import tqdm

FIN_TIP_NAME = 'caudal_fin'

def get_img_file(folder: pathlib.Path, relpath: str) -> pathlib.Path:
    img_file = folder / pathlib.Path(relpath).name
    if img_file.exists():
        return img_file
    return None

class TailAnnotationPlotter:
    __slots__ = ['fin_x', 'fin_y', 'caudal_scatter']
    def __init__(self):
        self.fin_x = None
        self.fin_y = None
        self.caudal_scatter = None

    def get_annotation(self, sub_df: pandas.DataFrame, img: numpy.ndarray):
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.scatter(sub_df.iloc[:20:2].values, sub_df.iloc[1:20:2].values, s=0.5)
        if FIN_TIP_NAME in sub_df.index.get_level_values(1):
            self.fin_x, self.fin_y = sub_df[sub_df.index.get_level_values(1) == FIN_TIP_NAME].values
            self.caudal_scatter = ax.scatter(self.fin_x, self.fin_y, s=0.5, c='red')
        def onclick(event, this=self, _fig=fig):
            this.fin_x, this.fin_y = event.xdata, event.ydata
            if this.caudal_scatter is not None:
                this.caudal_scatter.remove()
            this.caudal_scatter = ax.scatter(this.fin_x, this.fin_y, s=0.5, c='red')
            _fig.canvas.draw()
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        return self.fin_x, self.fin_y

def get_tail_annotations(folder: pathlib.Path, old_df: pandas.DataFrame) -> pandas.DataFrame:
    relpaths = old_df.index
    fin_data = [
        TailAnnotationPlotter().get_annotation(
            old_df.loc[relpath],
            fp_ffmpeg.freeimage_read(
                get_img_file(folder, relpath),
                transpose_to_match_convention=True))
        for relpath in tqdm(relpaths, desc=folder.name)
    ]
    return get_new_df(old_df, relpaths, fin_data)

def get_annotations_path(folder: pathlib.Path) -> pathlib.Path:
    res = list(folder.glob('CollectedData*.h5'))
    annotations_file = None
    if res:
        annotations_file = res[0]
    return annotations_file

def folder_is_usable(folder: pathlib.Path, old_df: pandas.DataFrame) -> bool:
    ''' Check that image files exist
    '''
    for relpath in old_df.index:
        if get_img_file(folder, relpath) is None:
            return False
    return True

def get_new_df(old_df, relpaths, fin_data):
    scorer = old_df.columns.get_level_values(0)[0]
    columns_multi_index = pandas.MultiIndex.from_product(
            [[scorer], [FIN_TIP_NAME], ['x', 'y']],
            names=['scorer', 'bodyparts', 'coords'])
    return pandas.DataFrame(fin_data, columns=columns_multi_index, index=relpaths)

def annotate_tails(folder: pathlib.Path):
    # Load annotations
    annotations_file = get_annotations_path(folder)
    if annotations_file is not None:
        old_df = pandas.read_hdf(annotations_file)
        new_annotations_file = annotations_file.parent / f'fin_{annotations_file.name}'
        if not new_annotations_file.exists() or True:
            if folder_is_usable(folder, old_df):
                # Add annotations
                new_df = get_tail_annotations(
                    folder,
                    pandas.concat((old_df, pandas.read_hdf(new_annotations_file)), axis=1, copy=False)
                        if new_annotations_file.exists() else old_df)
                save_dlc_df(new_df, new_annotations_file)
    # Recurse
    for item in folder.glob('*'):
        if item.is_dir():
            annotate_tails(item)

if __name__ == '__main__':
    FileLocations.parse_default_args()
    annotate_tails(FileLocations.get_training_root_dir())
