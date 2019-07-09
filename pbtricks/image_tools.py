from scipy.interpolate import interp2d
import numpy as np
from PIL import Image, ImageOps


def pad_to_square(input_image, desired_size, fill="grey"):
  """
  pads an image to the desired size while keeping the aspect ratio
  :param input_image: (uint8) numpy array
  :param desired_size: (int) desired square size
  :param fill: padding fill color
  :return:
  """
  im = Image.fromarray(input_image)
  old_size = im.size

  ratio = float(desired_size) / max(old_size)
  new_size = tuple([int(x * ratio) for x in old_size])
  im = im.resize(new_size, Image.ANTIALIAS)

  delta_w = desired_size - new_size[0]
  delta_h = desired_size - new_size[1]
  padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
  return np.array(ImageOps.expand(im, padding, fill=fill))


def resize_mat(mat, new_size):
  """
  Resize a matrix to the desired size. Input size is [num_channels, num_pixels, num_pixels]"""
  if mat.ndim == 2:
    mat = np.expand_dims(mat, axis=0)
  num_ch, _, num_pix = np.array(mat).shape

  x = np.arange(0, num_pix)
  y = np.arange(0, num_pix)
  ratio = (new_size - 1.) / (num_pix - 1)

  x_new = np.arange(0, new_size)
  y_new = np.arange(0, new_size)

  output = []
  for i in range(num_ch):
    resized_rf_func = interp2d(x * ratio, y * ratio, mat[i], kind='cubic')
    tmp_out = resized_rf_func(x_new, y_new)
    output.append(tmp_out)

  return np.squeeze(output)
