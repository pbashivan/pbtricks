import tensorflow as tf
import sys
from absl import flags


FLAGS = flags.FLAGS

# Top flags
tf.app.flags.DEFINE_string("model_type", "inception", "Model type, currently"
                                                      "inception or alexnet")
tf.app.flags.DEFINE_string('zoo_path',
                           '/braintree/home/bashivan/dropbox/Codes/base_model/Pipeline/Model_zoo/',
                           """Path to the TF model zoo. """
                           )

# Model train flags
tf.app.flags.DEFINE_string('dataset', 'cifar100', 'cifar10, cifar100 or imagenet.')
tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 330000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_string('data_dir', '',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path',
                           '',
                           # '/mindhive/dicarlolab/u/bashivan/Checkpoints/imagenet_dlbase_raw_temp',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 20.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('grad_clip_global_norm', 10,
                          """Global norm value to normalize the gradients.""")


# Flags for training SAGE models
tf.app.flags.DEFINE_integer('seed', 0, """Random seed value.""")
tf.app.flags.DEFINE_integer('depth', 9, """Depth parameter.""")
tf.app.flags.DEFINE_integer('num_filters', 100, """Number of filters.""")
tf.app.flags.DEFINE_integer('kernel_size', 3, """Number of filters.""")


# image processing flags
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")
tf.app.flags.DEFINE_string("preprocess_type", 'simple', "simple, alex, vgg, inception")
tf.app.flags.DEFINE_boolean('retina', False,
                            """Whether to apply retina transformation.""")

# Evaluation flags
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('run_all', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', None,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")


# Feature Extraction flags
tf.app.flags.DEFINE_string('in_file',
                           '/braintree/data2/active/users/bashivan/model_features/HVM_images.hdf5',
                           """Path to HDF5 file containing the images.""")
tf.app.flags.DEFINE_string('out_file',
                           '/om/user/bashivan/Retina/HVM_data/hvm_dlbase_all_raw_remix.hdf5',
                           """Path to HDF5 file which the features are written to.""")
tf.app.flags.DEFINE_boolean('PCA', False, 'Extract only PCA features')
tf.app.flags.DEFINE_boolean('average_vars', True, 'Whether to load checkpoint from moving average estimate of vars.')
tf.app.flags.DEFINE_integer('num_classes', 1001, 'Number of classes.')


# To get rid of the unparsed flags error
# remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
# assert(remaining_args == [sys.argv[0]])

# remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
# assert (remaining_args == [sys.argv[0]])
FLAGS(sys.argv, known_only=True)

