'''
This is just an example of how to run DeepLabCut
I cannot guarantee that this will work as written.
Please read the DeepLabCut documentation to be sure.

You need to first set a flag allowing cli (no GUI) runtime,
and activate the conda environment, assuming you have it set up with conda.

    >> export DLClight=True # we also do this in-script for your convenience
    >> conda activate dlc-macOS-CPU # for mac
    >> conda activate dlc-ubuntu-GPU # for ubuntu
'''
import os
import pathlib
# ensure this variable is set before importing dlc
os.environ['DLClight'] = 'true'
import deeplabcut

def dlc_analyze(config_path, videos_dir, output_dir):
    # Local import so that we can still generate html from docstrings
    # without loading the DLC conda environment.
    videos = [p.as_posix() for p in pathlib.Path(videos_dir).expanduser().resolve().glob('*.avi')]
    deeplabcut.analyze_videos(
        config_path,
        videos,
        gputouse=0,
        save_as_csv=True,
        destfolder=output_dir)

if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('config_path')
    PARSER.add_argument('videos_dir')
    PARSER.add_argument('output_dir')
    ARGS = PARSER.parse_args()
    dlc_analyze(ARGS.config_path, ARGS.videos_dir, ARGS.output_dir)
