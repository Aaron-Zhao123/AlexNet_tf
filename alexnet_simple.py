import tensorflow as tf
import numpy as np

def initialize_weights_mask(first_time_training, mask_dir, file_name):
    NUM_CHANNELS = 3
    NUM_CLASSES = 1000
    if (first_time_training):
        print('setting initial mask value')
        weights_mask = {
            'conv1': np.ones([11, 11, NUM_CHANNELS, 96]),
            'conv2': np.ones([5, 5, 48, 256]),
            'conv3': np.ones([3, 3, 256, 384]),
            'conv4': np.ones([3, 3, 192, 384]),
            'conv5': np.ones([3, 3, 192, 256]),
            'fc6': np.ones([6 * 6 * 256, 4096]),
            'fc7': np.ones([4096, 4096]),
            'fc8': np.ones([4096, NUM_CLASSES])
        }
        biases_mask = {
            'conv1': np.ones([96]),
            'conv2': np.ones([256]),
            'conv3': np.ones([384]),
            'conv4': np.ones([384]),
            'conv5': np.ones([256]),
            'fc6': np.ones([4096]),
            'fc7': np.ones([4096]),
            'fc8': np.ones([NUM_CLASSES])
        }

        # with open(mask_dir + 'maskcov0cov0fc0fc0fc0.pkl', 'wb') as f:
        #     pickle.dump((weights_mask,biases_mask), f)
    else:
        with open(mask_dir + file_name,'rb') as f:
            (weights_mask, biases_mask) = pickle.load(f)
    print('weights set')
    return (weights_mask, biases_mask)

def initialize_variables(new_model = False, weights_path = 'DEFAULT'):
    NUM_CHANNELS = 3
    NUM_CLASSES = 1000
    keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    if (new_model):
        if (weights_path == 'DEFAULT'):
            WEIGHTS_PATH = 'base.npy'
        else:
            WEIGHTS_PATH = weights_path
    else:
        if (weights_path == 'DEFAULT'):
            WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            WEIGHTS_PATH = weights_path
        # call the create function
    weights = {}
    biases = {}
    if (new_model):
        weights = {
            'conv1': tf.Variable(tf.truncated_normal([11, 11, NUM_CHANNELS, 96], stddev=5e-2)),
            'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=5e-2)),
            'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=5e-2)),
            'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=5e-2)),
            'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=5e-2)),
            'fc6': tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], stddev=0.01)),
            'fc7': tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01)),
            'fc8': tf.Variable(tf.truncated_normal([4096, NUM_CLASSES], stddev=0.01))
        }
        biases = {
            'conv1': tf.Variable(tf.constant(0.1, shape=[96])),
            'conv2': tf.Variable(tf.constant(0.1, shape=[256])),
            'conv3': tf.Variable(tf.constant(0.1, shape=[384])),
            'conv4': tf.Variable(tf.constant(0.1, shape=[384])),
            'conv5': tf.Variable(tf.constant(0.1, shape=[256])),
            'fc6': tf.Variable(tf.constant(0.1, shape=[4096])),
            'fc7': tf.Variable(tf.constant(0.1, shape=[4096])),
            'fc8': tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
        }
    else:
        weights_dict = np.load(WEIGHTS_PATH, encoding = 'bytes').item()
        for key in keys:
            for data in weights_dict[key]:
                if (len(data.shape) == 1):
                    biases[key] = tf.Variable(data)
                else:
                    weights[key] = tf.Vairable(data)
    return (weights, biases)

def conv_network(images, weights, biases, keep_prob, batch_size = 128):
    NUM_CLASSES = 1000
    NUM_CHANNELS = 3
    # conv1
    conv = tf.nn.conv2d(images, weights['conv1'], [1, 4, 4, 1], padding='VALID')
    pre_activation = tf.nn.bias_add(conv, biases['conv1'])
    # conv1 = tf.nn.relu(pre_activation)
    conv1 = tf.nn.relu(tf.reshape(pre_activation,conv.get_shape().as_list()))
    pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
    norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
    print(norm1.get_shape())

    #conv2
    conv = tf.nn.conv2d(norm1, weights['conv2'], [1, 1, 1, 1], padding='VALID')
    pre_activation = tf.nn.bias_add(conv, biases['conv2'])
    # conv2 = tf.nn.relu(pre_activation)
    conv2 = tf.nn.relu(tf.reshape(pre_activation,conv.get_shape().as_list()))
    pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name = 'pool2')
    norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
    print(norm2.get_shape())

    #conv3
    conv = tf.nn.conv2d(norm2, weights['conv3'], [1, 1, 1, 1], padding='VALID')
    pre_activation = tf.nn.bias_add(conv, biases['conv3'])
    # conv3 = tf.nn.relu(pre_activation)
    conv3 = tf.nn.relu(tf.reshape(pre_activation,conv.get_shape().as_list()))

    #conv4
    conv = tf.nn.conv2d(conv3, weights['conv4'], [1, 1, 1, 1], padding='VALID')
    pre_activation = tf.nn.bias_add(conv, biases['conv4'])
    conv4 = tf.nn.relu(tf.reshape(pre_activation,conv.get_shape().as_list()))

    #conv5
    conv = tf.nn.conv2d(conv4, weights['conv5'], [1, 1, 1, 1], padding='VALID')
    pre_activation = tf.nn.bias_add(conv, biases['conv5'])
    conv5 = tf.nn.relu(pre_activation)
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')
    print(pool5.get_shape())

    #fc6
    flattened = tf.reshape(pool5, [-1, 6*6*256])
    fc6 = tf.nn.relu(tf.matmul(flattened, weights['fc6']) + biases['fc6'])
    dropout6 = dropout(fc6, keep_prob)

    # fc7
    fc7 = tf.nn.relu(tf.matmul(fc6, weights['fc7']) + biases['fc7'])
    dropout7 = dropout(fc7, keep_prob)

    fc8 = tf.matmul(fc7, weights['fc8']) + biases['fc8']
    return fc8

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                                                  strides = [1, stride_y, stride_x, 1],
                                                  padding = padding, name = name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                          beta = beta, bias = bias, name = name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
