#! /usr/bin/env python3

import argparse
import numpy
import pathlib
from skimage import color, io

ARCHIVE_FOLDER_NAME = 'colors'

def read_image(fname):
    return numpy.float32(io.imread(fname))/255.

def write_image(img, fname):
    io.imsave(fname, img)

def to_grayscale(img):
    if img.shape[-1] != 3:
        return img
    return color.rgb2lab(img)[:, :, 0].astype(numpy.uint8)

def all_images_to_grayscale(images_dir):
    images_dir = pathlib.Path(images_dir)
    archive_dir = images_dir / ARCHIVE_FOLDER_NAME
    archive_dir.mkdir()
    image_paths = images_dir.glob('*.png')
    if len(image_paths) == 0:
        image_paths = images_dir.glob('*.bmp')
    if len(image_paths) == 0:
        print('Run this file inside the directory of RGB images.')
        return False
    for fpath in image_paths:
        gray_fname = fpath.as_posix()
        if gray_fname[-4:] == '.bmp':
            gray_fname = gray_fname.replace('.bmp','.png')
        img = read_image(fpath.as_posix())
        original_fname = archive_dir / fpath.name
        print(f'Original located at: {original_fname.as_posix()}')
        print(f'Grayscale located at: {gray_fname}')
        write_image(img, original_fname.as_posix())
        write_image(to_grayscale(img), gray_fname)
    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', help='Location of images to turn into grayscale.')
    args = parser.parse_args()
    all_images_to_grayscale(args.images_dir)
