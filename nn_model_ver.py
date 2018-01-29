
import tensorflow as tf
import tensorflow.contrib.slim as slim

from common import label_list

y_len = len(label_list)

"""
Interface:

In order to interact with nn_model file,
please follow following key points.

1. function name should be start with "get_",
   followed by model name
2. function must take x_shape, learning_rate
3. function must returns hypothesis, train, cost, keep_prob, X, Y

ex) get_spectrogram_nn

"""


def get_cnn_mfcc_noise(x_shape, learning_rate):
    
    global y_len

    print("learning_rate {}".format(learning_rate))

    x_shape[0] = None
    X = tf.placeholder(tf.float32, shape=x_shape)
    Y = tf.placeholder(tf.float32, shape=[None, y_len])

    ## weight and bias variables
    def weight_variable(name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(shape):
        return tf.Variable(tf.random_normal(shape))

    ## define conv and max_pool
    def conv2d(X, W):
        return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')

    def max_pool_2x2(X):
        return tf.nn.max_pool(X, ksize=[1,4,4,1], strides=[1,2,2,1], padding='SAME')

    ## define operation
    ### conv1: depth 16
    W_conv1 = weight_variable("conv1", [5, 5, 1, 16])
    b_conv1 = bias_variable([16])

    x_reshape = [-1, x_shape[1], x_shape[2], 1]
    x_image = tf.reshape(X, x_reshape)
    conv1 = tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1)
    h_conv1 = tf.nn.relu(conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    layer1 = h_pool1
    
    ### conv2: depth 32
    W_conv2 = weight_variable("conv2", [5, 5, 16, 32])
    b_conv2 = bias_variable([32])

    conv2 = tf.nn.bias_add(conv2d(layer1, W_conv2), b_conv2)
    h_conv2 = tf.nn.relu(conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    layer2 = h_pool2
    
    ### conv3: depth 64
    W_conv3 = weight_variable("conv3", [5, 5, 32, 64])
    b_conv3 = bias_variable([64])

    conv3 = tf.nn.bias_add(conv2d(layer2, W_conv3), b_conv3)
    h_conv3 = tf.nn.relu(conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    layer3 = h_pool3
    
    
    ### fc1: 512
    fc1_input_len = 4 * 7 * 64
    W_fc1 = weight_variable("fc1", [fc1_input_len, 512])
    b_fc1 = bias_variable([512])
    layer3_matrix = tf.reshape(layer3, [-1, fc1_input_len])
    matmul_fc1 = tf.matmul(layer3_matrix, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(matmul_fc1)
    layer4 = h_fc1
    
    ### dropout on fc1
    keep_prob = tf.placeholder(tf.float32)
    layer4_drop = tf.nn.dropout(layer4, keep_prob)

    ### fc2: y_len
    W_fc2 = weight_variable("fc2", [512, y_len])
    b_fc2 = bias_variable([y_len])
    matmul_fc2 = tf.matmul(layer4_drop, W_fc2) + b_fc2
    layer5 = tf.nn.softmax(matmul_fc2)
    hypothesis = layer5
    # print("layer5: {}".format(layer5))
    # print("matmul_fc2: {}".format(matmul_fc2))
    # import sys
    # sys.exit()

    ### hypothesis and train
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=layer6)
    cost = tf.reduce_mean(cross_entropy)
    learning_rate = tf.Variable(learning_rate)
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    tf.summary.scalar("loss", cost)
    
    tf.summary.histogram("weights", W_fc2)
    tf.summary.histogram("hypothesis", hypothesis)
    
    return hypothesis, train, cost, keep_prob, X, Y



def get_cnn_nn(x_len, learning_rate):
    
    global y_len

    print("learning_rate {}".format(learning_rate))

    x_len[0] = None
    X = tf.placeholder(tf.float32, shape=x_len)
    Y = tf.placeholder(tf.float32, shape=[None, y_len])

    ## weight and bias variables
    def weight_variable(name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(shape):
        return tf.Variable(tf.random_normal(shape))

    ## define conv and max_pool
    def conv2d(X, W):
        return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')

    def max_pool_2x2(X):
        return tf.nn.max_pool(X, ksize=[1,4,4,1], strides=[1,2,2,1], padding='SAME')

    ## define operation
    ### conv1: depth 8
    W_conv1 = weight_variable("conv1", [5, 5, 1, 8])
    b_conv1 = bias_variable([8])

    x_image = tf.reshape(X, [-1, 99, 41, 1])
    conv1 = tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1)
    h_conv1 = tf.nn.relu(conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    layer1 = h_pool1
    
    ### conv2: depth 16
    W_conv2 = weight_variable("conv2", [5, 5, 8, 16])
    b_conv2 = bias_variable([16])

    conv2 = tf.nn.bias_add(conv2d(layer1, W_conv2), b_conv2)
    h_conv2 = tf.nn.relu(conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    layer2 = h_pool2
    
    ### conv3: depth 32
    W_conv3 = weight_variable("conv3", [5, 5, 16, 32])
    b_conv3 = bias_variable([32])

    conv3 = tf.nn.bias_add(conv2d(layer2, W_conv3), b_conv3)
    h_conv3 = tf.nn.relu(conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    layer3 = h_pool3
    
    ### conv4: depth 64
    W_conv4 = weight_variable("conv4", [5, 5, 32, 64])
    b_conv4 = bias_variable([64])

    conv4 = tf.nn.bias_add(conv2d(layer3, W_conv4), b_conv4)
    h_conv4 = tf.nn.relu(conv4)
    h_pool4 = max_pool_2x2(h_conv4)
    layer4 = h_pool4
    
    ### fc1: 1024
    W_fc1 = weight_variable("fc1", [7*3*64, 512])
    b_fc1 = bias_variable([512])
    layer4_matrix = tf.reshape(layer4, [-1, 7*3*64])
    matmul_fc1 = tf.matmul(layer4_matrix, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(matmul_fc1)
    layer5 = h_fc1
    
    ### dropout on fc1
    keep_prob = tf.placeholder(tf.float32)
    layer5_drop = tf.nn.dropout(layer5, keep_prob)

    ### fc2: y_len
    W_fc2 = weight_variable("fc2", [512, y_len])
    b_fc2 = bias_variable([y_len])
    matmul_fc2 = tf.matmul(layer5_drop, W_fc2) + b_fc2
    layer6 = tf.nn.softmax(matmul_fc2)
    hypothesis = layer6
    # print("layer6: {}".format(layer6))
    # print("matmul_fc2: {}".format(matmul_fc2))
    # import sys
    # sys.exit()

    ### hypothesis and train
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=layer6)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=layer6)
    cost = tf.reduce_mean(cross_entropy)
    #cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))
    learning_rate = tf.Variable(learning_rate)
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    tf.summary.scalar("loss", cost)
    
    tf.summary.histogram("weights", W_fc2)
    tf.summary.histogram("hypothesis", hypothesis)

    ### prediction and accruacy
    # correct_prediction = tf.equal(tf.argmax(layer6, 1), tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    '''
    export_nodes = ['hypothesis', 'train', 'cost', 'keep_prob', 'x', 'y_']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    '''

    return hypothesis, train, cost, keep_prob, X, Y




# nn spectrogram_nn
def get_spectrogram_nn(x_len, learning_rate_val):

    global y_len

    keep_prob = tf.placeholder(tf.float32)
    
    X = tf.placeholder(tf.float32, (None, x_len))
    Y = tf.placeholder(tf.float32, (None, y_len))

    # 5 hidden layer
    with tf.name_scope('layer1') as scope:
        W1 = tf.get_variable("W1", shape=[x_len, x_len], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([x_len]), name='bias')
        layer1 = tf.add(tf.matmul(X, W1), b1, name='regression')
        layer1 = tf.nn.relu(layer1, name="relu")
        layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob, name="dropout")
    
    with tf.name_scope('layer2') as scope:
        W2 = tf.get_variable("W2", shape=[x_len, x_len], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([x_len]), name='bias')
        layer2 = tf.add(tf.matmul(layer1, W2), b2, name='regression')
        layer2 = tf.nn.relu(layer2, name="relu")
        layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob, name="dropout")
    
    with tf.name_scope('layer3') as scope:
        W3 = tf.get_variable("W3", shape=[x_len, y_len], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.Variable(tf.random_normal([y_len]), name='bias')
        layer3 = tf.add(tf.matmul(layer2, W3), b3)  # W*x + b
        hypothesis = tf.nn.softmax(layer3)
        tf.summary.histogram("weights", W3)
        tf.summary.histogram("hypothesis", hypothesis)
    
    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

    # Gradient Descent
    learning_rate = tf.Variable(learning_rate_val)
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # train = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

    tf.summary.scalar("loss", cost)
    
    return hypothesis, train, cost, keep_prob, X, Y






