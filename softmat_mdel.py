import tensorflow as tf
# import MNIST dataset using tensoflow build-in function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# creating a interactive session
sess = tf.InteractiveSession()

# creating placeholder for input and output
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# assign bias and weight to null tensors
w = tf.Variable(initial_value=tf.zeros([784,10], dtype=tf.float32))
b = tf.Variable(initial_value=tf.zeros([10],dtype=tf.float32))

#excute the assignment operation
sess.run(tf.global_variables_initializer())

# softmax regression
y = tf.nn.softmax(tf.matmul(x,w) + b)

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

# optimizaiton
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# mini- batch setting
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})

# test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})*100
print("The final accuracy for the simple ANN model is: {} % ".format(acc))

# close session
sess.close()