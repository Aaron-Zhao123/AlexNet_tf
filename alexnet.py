import tensorflow as tf
import numpy as np

class AlexNet(object):
    def __init__ (self, x, keep_prob, num_classes, weights_mask, weights_path = 'DEFAULT', new_model = False):
        # -x: tf.placeholder
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.layer_names = []

        if (new_model):
          self.isnew_model = True
          self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
          self.isnew_model = False
          if (weights_path == 'DEFAULT'):
              self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
          else:
              self.WEIGHTS_PATH = weights_path
          # call the create function
        self.create(weights_mask)

    def create(self, weights_mask):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1', mask = weights_mask['conv1'])
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2', mask = weights_mask['conv2'])
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3', mask = weights_mask['conv3'])

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4', mask = weights_mask['conv4'])

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5', mask = weights_mask['conv5'])
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6', mask = weights_mask['fc6'])
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name = 'fc7', mask = weights_mask['fc7'])
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8', mask = weights_mask['fc8'])

    def load_initial_weights(self, session):
      """
      As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
      as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
      dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
      need a special load function
      """
      # Load the weights into memory
      weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()


      # Loop over all layer names stored in the weights dict
      # store the layer names
      self.layer_names = []
      for op_name in weights_dict:
          self.layer_names.append(op_name)
          with tf.variable_scope(op_name, reuse = True):
            # Loop over list of weights/biases and assign them to their corresponding tf variable
            for data in weights_dict[op_name]:
              # Biases
              if len(data.shape) == 1:
                if (self.isnew_model):
                    var = tf.get_variable('biases', trainable = True,
                        initializer=tf.random_normal_initializer())
                else:
                    var = tf.get_variable('biases', trainable = True)
                    session.run(var.assign(data))
              # Weights
              else:
                if (self.isnew_model):
                    var = tf.get_variable('weights', trainable = True,
                        initializer = tf.truncated_normal_initializer())
                else:
                    var = tf.get_variable('weights', trainable = True)
                    session.run(var.assign(data))
    def save_weights(self, file_name = 'base'):
      print('Saving weights..')
      weights_val = {}
      for op_name in self.layer_names:
        with tf.variable_scope(op_name, reuse = True) as scope:
          weights = tf.get_variable('weights')
          biases = tf.get_variable('biases', shape = [num_filters])
          weights_val[op_name] = [weights.eval(), biases.eval()]
      np.save(file_name+'.npy', weights_val,encoding = 'bytes')

    def mask_weights(self, weights_mask, session):
      for op_name in self.layer_names:
        with tf.variable_scope(op_name, reuse = True):
          var = tf.get_variable('weights_mask', trainable = False)
          session.run(var.assign(weights_mask[op_name]))


"""
global layer definitions
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, mask,
         padding='SAME', groups=1):
  """
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k,
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)

  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    new_weights = weights * mask
    biases = tf.get_variable('biases', shape = [num_filters])


    if groups == 1:
      conv = convolve(x, new_weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(value=x,num_or_size_splits=groups,  axis=3)
      weight_groups = tf.split(num_or_size_splits=groups, value=new_weights, axis = 3)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]

      # Concat the convolved output together again
      conv = tf.concat(values = output_groups, axis = 3)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)

    return relu

def fc(x, num_in, num_out, name, mask, relu = True):
  with tf.variable_scope(name) as scope:

    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    new_weights = weights * mask

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, new_weights, biases, name=scope.name)

    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)
      return relu
    else:
      return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)

def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)
