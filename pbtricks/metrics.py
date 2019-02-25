from __future__ import print_function

import itertools
import numpy as np
import h5py
from functools import reduce
from scipy.stats import pearsonr

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import scale
from joblib import Parallel, delayed

np.random.seed([0])
npa = np.array


def object_averages(x, meta):
  """
  :param x: Computes the average responses to each object sorted by categories and objects.
  :param meta:
  :return: (np array) a (n_samples, n_features) matrix
  """
  m = meta
  if hasattr(m['category'], 'unique'):
    objects_by_category = itertools.chain(
      *[np.unique(m[m['category'] == c]['obj']) for c in m['category'].unique()])
  else:
    objects_by_category = itertools.chain(
      *[np.unique(m[m['category'] == c]['obj']) for c in m.categories])
  x_obj = np.array([x[np.array(m['obj'] == o), :].mean(0)
                   for o in objects_by_category])
  return x_obj


def compute_rdm(x, meta, method='pearson', type='obj'):
  """
  Computes RDM for a feature matrix.
  :param x: (array) feature matrix of size (n_images, n_features)
  :param method: (str) similarity measure to use ('pearson', 'spearman')
  :param type: (str) type of RDM matrix to compute ('image', 'obj'). For 'obj'
                     the average response of each feature to each object is
                     first computed and then the RDM is computed. For 'image'
                     RDM is computed between images.
  :return: (float)
  """
  if type == 'obj':
    x_av = object_averages(x, meta)
  else:
    x_av = x
  if method == 'pearson':
    return 1 - np.corrcoef(x_av)
  elif method == 'spearman':
    return 1 - spearmanr(x_av, axis=1)[0]
  else:
    raise ValueError("Method not supported. ('pearson', 'spearman')")


def _rdm_worker(input_path, meta, layer, seed=None, num_units=1000, rdm_type='obj'):
  """
  Worker function
  :param input_path:
  :param meta:
  :param layer:
  :param seed:
  :param num_units:
  :param rdm_type:
  :return:
  """
  if seed is not None:
    np.random.seed(seed)
  feats = np.array(h5py.File(input_path)[layer])
  if feats.ndim > 2:
    feats = feats.reshape(feats.shape[0], -1)
  tmp_indx_list = np.random.choice(feats.shape[1], np.min([num_units, feats.shape[1]]), replace=False)
  rdm = compute_rdm(feats[:, tmp_indx_list], meta, type=rdm_type)
  return rdm


def indirect_call(instance, name, args=(), kwargs=None):
  """
  Indirect caller for instance methods and multiprocessing. This method should be used to call the worker for
  parallel processing.
  :param instance: class instance
  :param name: name of the worker function to be called
  :param args: args
  :param kwargs: kwargs
  :return: instance of the class function
  """
  if kwargs is None:
    kwargs = {}
  return getattr(instance, name)(*args, **kwargs)


class MetricsExpert(object):
  def __init__(self,
               x,
               y,
               reps=None,
               meta=None,
               ):
    super(MetricsExpert, self).__init__()
    self.meta_hvm = meta

    self._x = x
    self._y = npa(y)
    if reps is not None and x is not None:
      assert np.all(reps.mean(0) == x)
    self._reps = reps

  @staticmethod
  def splithalf_averages(M):
    """
    Splits the M matrix into two splits along the first axis and computes the average of each split.
    :param M: n-dimensional numpy array
    :return: average of values in M split along the first dimension.
    """
    length = M.shape[0]
    ri = range(length)
    np.random.shuffle(ri)
    ri1 = ri[:length // 2]
    ri2 = ri[length // 2:]
    return np.nanmean(M[ri1], axis=0), np.nanmean(M[ri2], axis=0)

  @staticmethod
  def spearman_brown_correction(r, num_folds=2):
    return (num_folds * np.abs(r)) / (1 + (num_folds - 1) * np.abs(r))

  @staticmethod
  def _consistency_worker(reps, seed, metric=pearsonr):
    """
    Computes the internal consistency of neurons given their responses to images.
    :param reps: (list or nd-array) Contains the neural responses to images.
                  (list) list of responses to subset of images with different variation ranges.
                  Each item is of size [n_reps, n_images, n_neurons]
                  (nd_array) Complete neural response matrix of size [n_reps, n_images, n_neurons]
    :param seed: random seed
    :return: correlation (consistency) value
    """
    np.random.seed(seed)

    if isinstance(reps, list):
      rep_list1, rep_list2 = [], []
      for v in range(3):
        f1, f2 = MetricsExpert.splithalf_averages(reps[v])
        rep_list1.extend(f1.ravel())
        rep_list2.extend(f2.ravel())
      r = metric(npa(rep_list1), npa(rep_list2))[0]
      r = MetricsExpert.spearman_brown_correction(r)
      return r
    elif isinstance(reps, np.ndarray):
      f1, f2 = MetricsExpert.splithalf_averages(reps)
      rep_list1 = f1.ravel()
      rep_list2 = f2.ravel()
      r = metric(rep_list1, rep_list2)[0]
      r = MetricsExpert.spearman_brown_correction(r)
      return r
    else:
      raise ValueError("input 'reps' should be of type list or numpy ndarray.")

  @staticmethod
  def compute_consistency(reps, num_resamples, n_jobs=20, rand_seed=0, method='spearman', population=True):
    """
    Compute population consistency by splitting the reps matrix into two folds along its first dimension and
     compute the correlation between the average value for each split.
    :param reps: n-dimensional numpy array [n_reps, n_images, n_neurons]
    :param num_resamples: number of resamples for computing correlation between the splits.
    :param n_jobs: number of simultaneous jobs to create
    :param rand_seed: numpy random seed
    :param method: correlation method to be used ['spearman', 'pearson']
    :param population: whether to check consistency over the population of neurons (used for dim estimation)
    :return: spearman-brown corrected average correlation
             [n_resamples,] if population consistency
             [n_resamples, n_neurons] otherwise
    """
    func_dic = {'pearson': pearsonr,
                'spearman': spearmanr}
    metric = func_dic[method]

    with Parallel(n_jobs=n_jobs) as parallel:
      if population:
        corr_list = parallel(delayed(indirect_call)(MetricsExpert, '_consistency_worker',
                                                    (reps, rand_seed + i)) for i in range(num_resamples))
      else:
        corr_list = []
        if isinstance(reps, list):
          num_neurons = reps[0].shape[-1]
          for n in range(num_neurons):
            corr_list.append(parallel(delayed(indirect_call)(MetricsExpert, '_consistency_worker',
                                                             ([N0[:, :, n] for N0 in reps], rand_seed + r, metric))
                                      for r in range(num_resamples)))
        else:
          num_neurons = reps.shape[-1]
          for n in range(num_neurons):
            corr_list.append(parallel(delayed(indirect_call)(MetricsExpert, '_consistency_worker',
                                                             (reps[:, :, n], rand_seed + r, metric))
                                      for r in range(num_resamples)))

    return corr_list

  @staticmethod
  def estimate_neural_dim(reps, num_resamples=20, n_jobs=50, rand_seed=0):
    # Apply PCA on the average data
    av_response = np.nanmean(reps, axis=0)
    pca = PCA(whiten=True)
    pca.fit(scale(av_response))
    explained_var = [reduce(lambda x, y: x + y, pca.explained_variance_ratio_[:i + 1]) for i in
                     range(len(pca.explained_variance_ratio_))]
    # Find the noise threshold from population consistency
    threshold = np.mean(MetricsExpert.compute_consistency(reps,
                                                          num_resamples=num_resamples,
                                                          n_jobs=n_jobs,
                                                          rand_seed=rand_seed)) ** 2
    dim = np.nonzero(explained_var > threshold)[0][0] + 1
    return dim

  def compute_fit_score(self,
                        num_folds=2,
                        rand_state=0,
                        subsample=False,
                        subset_size=None,
                        reg=LinearRegression(n_jobs=-1)):
    """
    Compute the regression score, given model_features and the response variables
    :param num_folds: Number of folds
    :param rand_state: Random seed to be used
    :param subsample: Flag for whether to subsample the features.
    :param subset_size: size of the subsample
    :param reg: Regression model to fit the data
    :return:
    """
    classifiers = []
    kf = KFold(num_folds, shuffle=True, random_state=rand_state)
    kf.get_n_splits(self._x.shape[0])
    if subsample & (subset_size <= self._x.shape[1]):
      tmp_indx_list = np.random.choice(
        self._x.shape[1], subset_size, replace=False)
      tmp_indx_list.sort()
      model_features_aftersub = self._x[:, tmp_indx_list]
    else:
      model_features_aftersub = self._x
    predictions = np.zeros(self._y.shape)
    for train_ind, test_ind in kf.split(model_features_aftersub):
      now_train_data = model_features_aftersub[train_ind, :]
      now_train_label = self._y[train_ind]
      now_test_data = model_features_aftersub[test_ind, :]
      reg.fit(now_train_data, now_train_label)
      classifiers.append(reg)
      predictions[test_ind] = reg.predict(now_test_data)
    unit_score = r2_score(self._y, predictions, multioutput='raw_values')
    return unit_score, classifiers, predictions

  @staticmethod
  def preprocess_features(features, n_components, iterative=True):
    """
    Applys PCA on the features matrix and selects the top n_components.
    :param features: (nd-array) Features matrix [n_samples, n_features]
    :param n_components: (int) Number of components to retain after PCA
    :param iterative: (bool) If True, will perform iterative PCA
    :return: (nd-array) transformed features matrix
    """
    n_components = min(n_components, features.shape[1])
    batch_size = n_components
    if iterative:
      pca = IncrementalPCA(n_components=n_components, whiten=False, batch_size=batch_size)
    else:
      pca = PCA(n_components=n_components, whiten=False)

    output = np.zeros((features.shape[0], min(n_components, features.shape[1])), dtype=np.float)
    features = scale(features)
    pca.fit(features)

    for c in range(0, features.shape[0], batch_size):
      output[c:c + batch_size] = pca.transform(features[c:c + batch_size])
    return output

  def compute_rdm(self, X, method='pearson', type='obj'):
    """
    Computes RDM for a feature matrix.
    :param X: (array) feature matrix of size (n_images, n_features)
    :param method: (str) similarity measure to use ('pearson', 'spearman')
    :param type: (str) type of RDM matrix to compute ('image', 'obj'). For 'obj'
                       the average response of each feature to each object is
                       first computed and then the RDM is computed. For 'image'
                       RDM is computed between images.
    :return: (float)
    """
    if type == 'obj':
      X_av = self.object_averages(X)
    else:
      X_av = X
    if method == 'pearson':
      return 1 - np.corrcoef(X_av)
    elif method == 'spearman':
      return 1 - spearmanr(X_av, axis=1)[0]
    else:
      raise ValueError("Method not supported. ('pearson', 'spearman')")

  def object_averages(self, x):
    """
    :param x: Computes the average responses to each object sorted by categories and objects.
    :return: (np array) a (n_samples, n_features) matrix
    """
    m = self.meta_hvm
    objects_by_category = itertools.chain(
      *[np.unique(m[m['category'] == c]['obj']) for c in m['category'].unique()])
    x_obj = npa([x[npa(m['obj'] == o), :].mean(0) for o in objects_by_category])
    return x_obj

  def plot_rdm(self, X):
    X_av = self.object_averages(X)
    fig, ax = plt.subplots()
    plt.matshow(np.corrcoef(X_av.T))
    return fig, ax

  def apply_pca(self, features, pca_method='pca'):
    """

    :param features:
    :param pca_method:
    :return:
    """
    if pca_method == 'pca':
      pca = PCA(whiten=True)
    elif pca_method == 'incremental':
      pca = IncrementalPCA(batch_size=200)
    else:
      raise ValueError('PCA mehtod not recognized.')
    pca.fit(features)

    return pca
