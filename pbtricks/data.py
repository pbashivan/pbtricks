from __future__ import print_function
import pickle
from imageio import imread
import os
import h5py
import numpy as np

npa = np.array


def unpickle(file):
  """
  Load contents of a pickle file
  :param file:
  :return:
  """
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict


def pack_images_h5(images_path, output_path=None, file_type='png', image_names=None):
  """
  Packs a set of images in the given directory
  :param images_path: path to directory containing the images
  :param output_path: path to save the output file
  :param file_type: file extension name: "png"
  :param image_names: list containing name of all images (None: all images)
  :return: None
  """
  images = []
  if image_names is None:
    files_list = sorted(os.listdir(images_path))
    for filename in files_list:
      if filename.endswith("." + file_type):
        im = imread(os.path.join(images_path, filename))
        if np.ndim(im) == 2:
          im = np.repeat(np.expand_dims(im, -1), 3, axis=-1)
        images.append(im)
  else:
    for idx in image_names:
      filename = os.path.join(images_path, idx + '.' + file_type)
      im = imread(filename)
      if np.ndim(im) == 2:
        im = np.repeat(np.expand_dims(im, -1), 3, axis=-1)
      images.append(im)

  if output_path is None:
    output_path = images_path
  with h5py.File(os.path.join(output_path, 'images.h5'), 'w') as h5file:
    h5file.create_dataset('images', data=np.array(images))

  print('Finished saving.')


def load_h5file(path, keys=None):
  """
  Load contents of a hdf5 file
  :param path: path to h5 file
  :param keys: keys to load
  :return:
  """

  with h5py.File(path, 'r') as h5file:
    try:
      if keys is None:
        keys = []
        h5file.visit(keys.append)
      return {k: npa(h5file[k]) for k in keys}
    except KeyError:
      print('Existing keys are: ', h5file.keys())
      raise KeyError


