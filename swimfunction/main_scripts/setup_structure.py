''' Initializes a directory structure and
prompts the user to import pose annotation files
and other data into the experiment directory structure.
'''
from swimfunction.global_config.config import config

from swimfunction import FileLocations
from swimfunction import loggers
import pathlib
import shutil

def prompt_yes(question):
    res = '?'
    while res not in ['Y', 'N']:
        res = input(f'{question} (Y/N) ').strip().upper()
    return res == 'Y'

def find_files(directory: pathlib.Path, suffixes: list, dirnames_to_ignore: list) -> list:
    ''' Recursively finds files in the directory with given suffixes.
    '''
    logger = loggers.get_video_processing_logger(__name__)
    files = []
    for suffix in suffixes:
        files = files + list(directory.rglob(f'*{suffix}'))
    if not files:
        logger.warning(f'No {suffixes} files found in {directory}')
    if dirnames_to_ignore is not None:
        files = list(filter(lambda f: pathlib.Path(f).parent.name not in dirnames_to_ignore, files))
    return files

def prompt_locate_files(suffixes: list, dirnames_to_ignore: list) -> list:
    ''' Prompt user to locate files with suffixes.
    '''
    loc = input(f'Which directory contains files with these suffixes {suffixes}?').strip()
    files = find_files(pathlib.Path(loc), suffixes, dirnames_to_ignore)
    while not files:
        print('Didn\'t find any files there.')
        if not prompt_yes('Would you like to ignore this step?'):
            files = []
            break
        loc = input(f'Which directory contains files with these suffixes {suffixes}?').strip()
        files = find_files(pathlib.Path(loc), suffixes, dirnames_to_ignore)
    return files

def populate_directory(files: list, target_dir: pathlib.Path, do_copy: bool):
    ''' Copy in, or link to, files.
    '''
    for filepath in files:
        filepath = pathlib.Path(filepath).expanduser().resolve()
        newfilepath = target_dir / filepath.name
        if newfilepath.exists():
            continue
        if filepath.is_dir():
            print(f'Skipping directory {filepath.as_posix()}')
            continue
        if do_copy:
            shutil.copy(filepath.as_posix(), newfilepath.as_posix())
        else:
            newfilepath.symlink_to(filepath)

def setup_directory(target_dir: pathlib.Path, suffixes: list, file_description, dirnames_to_ignore: list):
    ''' Sets up a directory structure given user input.
    '''
    files = find_files(target_dir, suffixes)
    if files:
        print(f'Found {len(files)} {file_description} files already in {target_dir.as_posix()}')
    if prompt_yes(f'Would you like to import {file_description} files?'):
        files = prompt_locate_files(suffixes, dirnames_to_ignore)
        if files:
            do_copy = prompt_yes('Would you like to copy the files? If no, will create symlink.')
            populate_directory(files, target_dir, do_copy)

def setup_structure():
    ''' Sets up the directory structure given user input.
    '''
    print('Experiment name: ', config.experiment_name)
    print(
        'Experiment location on disk: ',
        FileLocations.get_experiment_cache_root().as_posix())
    if prompt_yes('Do you have pose annotation files to import?'):
        setup_directory(
            FileLocations.get_dlc_outputs_dir(),
            ['.csv', '.h5'],
            'pose annotation',
            []
        )
    if prompt_yes('Do want to import videos matching the pose files? ',
                  'Including the videos can help ensure posture annotation is accurate.'):
        setup_directory(
            FileLocations.get_normalized_videos_dir(),
            ['.avi', '.mp4', '.ctlog', '.ctlog.gz'],
            'video',
            dirnames_to_ignore=['median_frames', 'crop_tracked'])
    if prompt_yes('Do you have precalculated scores in csv format to import?'):
        setup_directory(
            FileLocations.get_precalculated_metrics_dir(),
            ['.csv'],
            'precalculated score',
            []
        )

if __name__ == '__main__':
    FileLocations.parse_default_args()
    setup_structure()
