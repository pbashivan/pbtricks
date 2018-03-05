import numpy as np
from sklearn.preprocessing import scale


def reps_to_array(reps):
  """
  reps: dictionary containing the reps for each variation level
  """
  max_reps = np.max([reps[i].shape[0] for i in reps.keys()], axis=0)
  hvm_neural = np.zeros((max_reps, 5760, reps['V0'].shape[2]))
  hvm_neural[...] = np.NaN

  c = 0
  for key in reps:
    shape = reps[key].shape
    hvm_neural[:shape[0], c:c + shape[1], :] = reps[key]
    c += shape[1]
  return hvm_neural


def concat_reps(rep_list):
  """
  """
  max_reps = np.max([r.shape[0] for r in rep_list])
  resized_reps = []
  for r in rep_list:
    tmp = np.zeros((max_reps, r.shape[1], r.shape[2]))
    tmp[...] = np.NaN
    tmp[:r.shape[0], :, :] = r
    resized_reps.append(tmp)
  return np.concatenate(resized_reps, axis=-1)


def fix_nan_reps(reps):
  """Some of the entries in neural reps might be nan.
  Substitute those values by the average response of
  corresponding neurons to all images over all valid reps.
  reps = [n_reps, n_samples, n_neurons]
  """
  if np.any(np.isnan(reps)):
    # find the indices of nan neurons
    nan_ind = np.isnan(reps)
    _, _, nan_neu_ind = np.nonzero(nan_ind)

    corrected_reps = reps
    for n in np.unique(nan_neu_ind):
      # create a mask of all nan values for a neuron
      mask = np.zeros(shape=nan_ind.shape, dtype=bool)
      mask[:, :, n] = True
      masked_nan_ind = nan_ind & mask

      # substitue all nan values of neuron by average neuron response
      av_neuron_act = np.nanmean(reps[:, :, n])
      corrected_reps[masked_nan_ind] = av_neuron_act
    return corrected_reps
  else:
    return reps


def project_reps(input_reps, W_mat):
  """Project each rep of neural data using the projection matrix
  input_reps = [n_reps, n_samples, n_neurons]"""
  input_reps = fix_nan_reps(input_reps)
  reps = []
  for rep in input_reps:
    reps.append(scale(rep))
  comp_reps = np.tensordot(reps, W_mat, axes=1)
  return comp_reps