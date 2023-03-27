'''
Just an example of how to run DeepLabCut
I cannot guarantee that this will work as written.
Use `swimfunction/README.md` as a guide.
Please read the DeepLabCut documentation to be sure.

You need to first set a flag allowing cli (no GUI) runtime,
and activate the conda environment.

    >> export DLClight=True # we also do this in-script for your convenience
    >> conda activate dlc-macOS-CPU # for mac
    >> conda activate dlc-ubuntu-GPU # for ubuntu
'''
import os
os.environ['DLClight'] = 'true'

def train_dlc(config_path, training_dir):
    # Local import so that we can still generate html from docstrings
    # without loading the DLC conda environment.
    import deeplabcut

    config_path = '/home/zplab/Jensen_Nick/wt_uninjured-Jensen-2019-12-19/config.yaml'

    # train_network does what it says on the tin.
    deeplabcut.train_network(config_path, saveiters=100000, displayiters=500, maxiters=1500000, max_snapshots_to_keep=12, gputouse=0)

if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('config_path')
    PARSER.add_argument('training_dir')
    ARGS = PARSER.parse_args()
    train_dlc(ARGS.config_path, ARGS.training_dir)
