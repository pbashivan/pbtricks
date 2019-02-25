"""A class to extract features for a CNN model defined in TF using a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time
import h5py
import scipy.misc
import numpy as np
import tensorflow as tf
import os
import sys
import logging
import tensorflow.contrib.slim as slim

npa = np.array


class Extractor(object):
  def __init__(self, checkpoint_path, model_type, dataset,
               preprocess_type, retina, image_size, batch_size, output_path, model_graph=None,
               log_rate=10, num_classes=1001,
               zoo_path='/braintree/home/bashivan/dropbox/Codes/base_model/Pipeline/Model_zoo/'):
    """Evaluate model on Dataset for a number of steps."""
    # Add zoo_path before checkpoint_dir. It will check the checkpoint_dir before zoo_path
    # to find the model definition file. If it exists along with the checkpoints will be
    # loaded otherwise would retreat to the default model zoo.

    self._setup_logger()
    self._logger.info('Logger setup complete.')

    self._model = model_graph
    self._checkpoint_path = checkpoint_path
    self._model_type = model_type
    self._dataset = dataset
    self._preprocess_type = preprocess_type
    self._retina = retina
    self._image_size = image_size
    self._batch_size = batch_size
    self._log_rate = log_rate
    self._num_classes = num_classes
    self._output_path = output_path
    self._zoo_path = zoo_path
    self._initialized = False
    self._images_placeholder = None
    self._endpoints = None
    self._saver = None
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._sess = tf.Session()

    self._R_MEAN = 123.68
    self._G_MEAN = 116.78
    self._B_MEAN = 103.94

    if self._model is None:
      assert (self._checkpoint_path is not None) & \
             (self._model_type is not None)
      if self._zoo_path not in sys.path:
        sys.path.insert(0, self._zoo_path)
      if os.path.isdir(self._checkpoint_path):
        if self._checkpoint_path not in sys.path:
          sys.path.insert(0, self._checkpoint_path)
      else:
        tmp_path = os.path.dirname(self._checkpoint_path)
        if tmp_path not in sys.path:
          sys.path.insert(0, tmp_path)
      exec('import {0}_model as model_class'.format(self._model_type))

      self._model = eval('model_class.{0}()'.format(self._model_type.title()))

  def _setup_logger(self):
    FORMAT = '%(asctime)-15s - %(message)s'
    logging.basicConfig(format=FORMAT)
    self._logger = logging.getLogger('basiclogger')
    self._logger.setLevel('INFO')

  def _preprocess_images(self, images):
    images_resized = np.zeros((images.shape[0], self._image_size, self._image_size, 3), dtype=np.float32)
    for i, im in enumerate(images):
      images_resized[i] = scipy.misc.imresize(im, (self._image_size, self._image_size, 3)).astype(np.float32)
    if self._preprocess_type in ['simple', 'inception', 'alex'] or 'cifar' in self._dataset:
      images_resized -= 128.
      images_resized /= 128.
      self._logger.info('(-1 - 1)')
    else:
      images_resized -= npa([self._R_MEAN, self._G_MEAN, self._B_MEAN]).reshape(1, 1, 1, 3)
      self._logger.info('(-128 - 128)')
    return images_resized

  def _preprocess_images_retina(self, images):
    if self._preprocess_type == 'vgg':
      import pbtricks.preprocessors.image_processing_vgg_retina as imprep_retina
    elif self._preprocess_type == 'alex':
      import pbtricks.preprocessors.image_processing_alex_retina as imprep_retina
    else:
      import pbtricks.preprocessors.image_processing_alex_retina as imprep_retina
    with self._graph.as_default():
      images_resized = np.zeros((images.shape[0], self._image_size, self._image_size, 3), dtype=np.float32)
      image_holder = tf.placeholder(tf.float32, shape=list(images.shape[1:]))
      image = imprep_retina._central_crop([image_holder], self._image_size, self._image_size)[0]
      image.set_shape([self._image_size, self._image_size, 3])
      image = imprep_retina._mean_image_subtraction(image,
                                                    [imprep_retina._R_MEAN, imprep_retina._G_MEAN,
                                                     imprep_retina._B_MEAN])
      if self._preprocess_type in ['simple', 'inception', 'alex']:
        self._logger.info('(0-1)')
        image = tf.div(image, 128.)
      else:
        self._logger.info('(0-255)')

      print('\n')
      for i, im in enumerate(images):
        # print('resizing image %d / %d' % (i + 1, images.shape[0]), end='')
        images_resized[i] = self._sess.run(image, feed_dict={image_holder: im})
    return images_resized

  def _read_images(self, images):
    # Get images and labels from the dataset.
    assert isinstance(images, np.ndarray)
    if np.ndim(images) == 3:
      images = np.repeat(np.expand_dims(images, 3), 3, axis=3)
    if self._dataset == 'imagenet':
      if self._retina:
        self._logger.info('Preprocessing Retina')
        self.images_resized = self._preprocess_images_retina(images)
      else:
        self._logger.info('Preprocessing Normal')
        self.images_resized = self._preprocess_images(images)
    else:
      self.images_resized = self._preprocess_images(images)
    self._logger.info('Preprocessing done!')

  @staticmethod
  def tf_repeat(images, num_reps=3):
    return tf.tile(tf.expand_dims(images, axis=-1), [1, 1, 1, num_reps])

  @staticmethod
  def iterate_minibatches(images, batchsize, shuffle=False):
    input_len = images.shape[0]

    if shuffle:
      indices = np.arange(input_len)
      np.random.shuffle(indices)
    for start_idx in range(0, input_len, batchsize):
      if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
      else:
        excerpt = slice(start_idx, start_idx + batchsize)
      yield images[excerpt]

  def _load_checkpoint(self):
    init = tf.global_variables_initializer()
    self._sess.run(init)
    if os.path.isdir(self._checkpoint_path):
      ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          self._saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
          # Restores from checkpoint with relative path.
          self._saver.restore(self._sess, os.path.join(self._checkpoint_path,
                                                       ckpt.model_checkpoint_path))

        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        self._logger.info('Succesfully loaded model from %s at step=%s.' %
                         (ckpt.model_checkpoint_path, global_step))
      else:
        self._logger.info('No checkpoint file found')
        return

    else:
      self._saver.restore(self._sess, self._checkpoint_path)
      global_step = self._checkpoint_path.split('/')[-1].split('-')[-1]
      self._logger.info('Succesfully loaded model from %s at step=%s.' %
                       (self._checkpoint_path, global_step))

  def _calculate_activations(self, endpoints_to_extract):
    """
    Extract activations from designated layers.
    :param endpoints_to_extract: name of endpoints for which the activations should be extracted.
    :return:
    """
    if not self._initialized:
      self._load_checkpoint()
      self._initialized = True
    try:
      num_examples = self.images_resized.shape[0]
      num_iter = int(math.ceil(num_examples / self._batch_size))
      step = 0
      start_time = time.time()
      total_examples = 0

      if self._output_path is not None:
        with h5py.File(self._output_path) as h5file:
          # Remove all model features in HDF5
          for key in h5file.keys():
            if 'mdl_' in key:
              h5file.__delitem__(key)

      all_feats = None
      for batch in self.iterate_minibatches(self.images_resized, self._batch_size):
        # if not coord.should_stop():
        total_examples += batch.shape[0]
        feed_dict = {self.images_placeholder: batch}
        feats = self._sess.run(self._endpoints, feed_dict=feed_dict)
        if endpoints_to_extract is None:
          endpoints_to_extract = feats.keys()

        # Store specific feature sets
        if self._output_path is not None:
          with h5py.File(self._output_path) as h5file:
            for key in endpoints_to_extract:
              if 'mdl_' + key.replace('/', '_') not in h5file.keys():
                ds_feat = h5file.create_dataset('mdl_' + key.replace('/', '_'),
                                                shape=self.images_resized.shape[0:1] + feats[key].shape[1:],
                                                dtype=np.float32)
              else:
                ds_feat = h5file['mdl_' + key.replace('/', '_')]
              ds_feat[step * self._batch_size:(step + 1) * self._batch_size, :] = feats[key]
        else:
          if all_feats is None:
            all_feats = feats
          else:
            for key in endpoints_to_extract:
              all_feats[key] = np.concatenate((all_feats[key], feats[key]), axis=0)

        step += 1
        if step % self._log_rate == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / self._log_rate
          examples_per_sec = self._batch_size / sec_per_batch
          self._logger.info('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                           'sec/batch)' % (datetime.now(), step, num_iter,
                                           examples_per_sec, sec_per_batch))
          start_time = time.time()
      self._logger.info('%s: [%d examples]' %
                       (datetime.now(), total_examples))
      if self._output_path is None:
        return all_feats
      else:
        return self._output_path

    except Exception as e:
      raise e

  def extract(self, images, average_vars=False, endpoints_to_extract=None):
    assert images.shape[0] % self._batch_size == 0
    self._read_images(images)
    with self._graph.as_default():
      if not self._initialized:
        self.images_placeholder = tf.placeholder(tf.float32,
                                                 shape=tuple([self._batch_size] + list(self.images_resized.shape[1:])))

        _, self._endpoints = self._model.inference(self.images_placeholder, self._num_classes, for_training=False)
        if average_vars:
          variable_averages = tf.train.ExponentialMovingAverage(
            self._model.MOVING_AVERAGE_DECAY)
          variables_to_restore = variable_averages.variables_to_restore()
        else:
          variables_to_restore = slim.get_variables_to_restore()
        self._saver = tf.train.Saver(variables_to_restore)
      return self._calculate_activations(endpoints_to_extract=endpoints_to_extract)

  def close(self):
    self._sess.close()
