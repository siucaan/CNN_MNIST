import tensorflow as tf
import numpy as np
import urllib.request

# get tools from remote sever
response = urllib.request.urlopen('http://deeplearning.net/tutorial/code/utils.py')
content = response.read().decode('utf-8')
target = open('utils1.py', 'w')
target.write(content)
target.close()

from utils1 import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image

# start a interactive session
sess = tf.InteractiveSession()

# load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# initial parameter
width = 28
height = 28
flat = width * height
class_output = 10

# create placeholder for input and output
x = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])
# converting images of the data set to tensor
x_image = tf.reshape(x, [-1, 28, 28, 1])

# define kernal weight and bias
# a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# convolutional layer 1
convolve1 = tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
# apply reLU activation function
h_conv1 = tf.nn.relu(convolve1)

# apply the max pooling
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# convolutional layer 2
w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
convolve2 = tf.nn.conv2d(conv1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2
h_conv2 = tf.nn.relu(convolve2)
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Fully connected layer
# flattening the second convolutional layer output to [3136x1]
layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])

# weights and biases between layer 2 and 3
w_fcl = tf.Variable(tf.truncated_normal(shape=[3136, 1024], stddev=0.1))
b_fcl = tf.Variable(tf.constant(0.1, shape=[1024]))

#
fcl = tf.matmul(layer2_matrix, w_fcl) + b_fcl

# apply the reLU activation function
h_fcl = tf.nn.relu(fcl)


# Dropout Layer
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fcl, keep_prob)

# softmat layer
w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
# matrix multiplication
fc = tf.matmul(layer_drop, w_fc2) + b_fc2
# apply softmat activation function
y_cnn = tf.nn.softmax(fc)

# define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_cnn), reduction_indices=[1]))

# define the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# define prediction
correct_prediction = tf.equal(tf.argmax(y_cnn, 1), tf.argmax(y_, 1))

# define accuracy
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run session
sess.run(tf.global_variables_initializer())
for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_acc = acc.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, float(train_acc)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7})

print('test accuracy %g' % acc.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# look at all the filters
kernels = sess.run(tf.reshape(tf.transpose(w_conv1, perm=[2, 3, 0, 1]), [32, -1]))
### get tools from remote sever
# %matplotlib inline
image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5) ,tile_shape=(4, 8), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')

# see the output of an image passing through first convolution layer
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage, [28, 28]), cmap='gray')
ActivatedUnits = sess.run(convolve1, feed_dict={x:np.reshape(sampleimage, [1, 784], order='F'), keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20, 20))
n_columns = 6
n_rows = np.math.ceil(filters/n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filters' + str(i))
    plt.imshow(ActivatedUnits[0, :, :, i], interpolation = 'nearest', cmap='gray')


# see the output of an image passing through second convolution layer
ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage, [1,784], order='F'), keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0, :, :, i], interpolation="nearest", cmap="gray")

sess.close()  # finish the session`