from scipy.interpolate import interp2d
import numpy as np


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
