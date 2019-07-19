#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from PIL import Image


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(root_dir, sequences, extensions):
    """Search and store the paths of all images as well as the image number

    :param root_dir: the root dir that contains the folders of all sequences
    :param sequences: the sequence names
    :param extensions: the supported image file extensions
    :return: a dict that stores the image file path with key=(seq_name, image_name)
             a dict that stores the image number of a sequence with key=seq_name
    """
    images = {}
    image_nums = {}
    for seq in sorted(sequences):
        image_num = 0
        d = os.path.join(root_dir, seq)
        if not os.path.isdir(d):
            continue

        for root, subfolders, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (seq, fname)
                    images[item] = path
                    image_num += 1
        image_nums[seq] = image_num

    return images, image_nums


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ACRV(object):
    def __init__(self, data_dir, transform=None):
        # data_dir: the folder that contains sequences of frames
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']
        sequences = self._find_sequences(data_dir)
        samples, sample_nums = make_dataset(data_dir, sequences, IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + data_dir + "\n"
                                "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.data_dir = data_dir
        self.sequences = sequences
        self.samples = samples
        self.sample_keys = sorted(self.samples.keys())
        self.sample_nums = sample_nums
        self.transform = transform
        self.selected_seq = None  # choose a certain sequence

    def __getitem__(self, idx):
        # given an idx, read the image
        # if idx is an int, read the image in order of seq_name first and then image_name;
        # when the selected_seq is given, read the image: selected_seq/idx;
        # if idx is a tuple (seq_name, image_name), read the image: seq_name/image_name
        if isinstance(idx, int):
            if not self.selected_seq:
                key = self.sample_keys[idx]
                img_dir = self.samples[key]
                img = pil_loader(img_dir)
                img.name = self.sample_keys[idx]
            else:
                # ends the loop when idx >= the number of images in this sequence
                if idx >= self.sample_nums[self.selected_seq]:
                    raise IndexError()
                image_name = '{0:06d}'.format(idx) + '.png'
                key = (self.selected_seq, image_name)
                img_dir = self.samples[key]
                img = pil_loader(img_dir)
                img.name = key

        elif isinstance(idx, tuple) and len(idx) == 2:
            seq_name = idx[0]
            image_name = idx[1]
            if len(str(seq_name)) < 6:
                seq_name = '{0:06d}'.format(int(seq_name))
            if len(str(image_name)) < 6:
                image_name = '{0:06d}'.format(int(image_name)) + '.png'
            key = (seq_name, image_name)
            img_dir = self.samples[key]
            img = pil_loader(img_dir)
            img.name = key
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        if self.selected_seq:
            return self.sample_nums[self.selected_seq]
        return len(self.samples)

    def __call__(self, seq_name=None):
        # initialize the selected_seq and iterate images just in this sequence
        if isinstance(seq_name, int) or len(str(seq_name)) < 6:
            self.selected_seq = '{0:06d}'.format(int(seq_name))
        else:
            self.selected_seq = seq_name

    def _find_sequences(self, dir):
        """
        Finds the sequence folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            sequence names:

        Ensures:
            No sequence folder is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            sequences = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            sequences = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        sequences.sort()
        return sequences

    def get_image_name(self, idx):
        # get the image name, given the index
        if isinstance(idx, int):
            return self.sample_keys[idx][0] + '/' + self.sample_keys[idx][1]
        else:
            seq_name = idx[0]
            image_name = idx[1]
            if len(str(seq_name)) < 6:
                seq_name = '{0:06d}'.format(int(seq_name))
            if len(str(image_name)) < 6:
                image_name = '{0:06d}'.format(int(image_name)) + '.png'
            return seq_name + '/' + image_name


if __name__ == '__main__':
    # replace the data_dir with your own and run the code to learn the usage
    data_dir = '/home/wind/Data/Isaac_sim/AIUE_V01_001/'
    acrv_testdata = ACRV(data_dir)
    idx = 8
    img = acrv_testdata[idx]
    print(img.name)
    img = acrv_testdata[('000001', idx)]
    print(img.name)
    acrv_testdata('000002')
    img = acrv_testdata[idx]
    print(img.name)
    for image in acrv_testdata:
        print(image.name)
    print('end')
