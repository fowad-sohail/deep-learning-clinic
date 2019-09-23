

import tensorflow as tf
tf.set_random_seed(42)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
tf.set_random_seed(42)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


print('Training Data:')
print('Number of training examples: ', mnist.train.num_examples)
print('Shape of training inputs: ', mnist.train.images.shape)
print('Shape of training labels: ', mnist.train.labels.shape)

print('Validation Data:')
print('Number of validation examples: ', mnist.validation.num_examples)
print('Shape of validation inputs: ', mnist.validation.images.shape)
print('Shape of validation labels: ', mnist.validation.labels.shape)

print('Test Data:')
print('Number of test examples: ', mnist.test.num_examples)
print('Shape of test inputs: ', mnist.test.images.shape)
print('Shape of labels labels: ', mnist.test.labels.shape)

# Iniatialize all variables and palceholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape = [100]))

W2 = tf.Variable(tf.truncated_normal([100, 30], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape = [30]))

W3 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape = [10]))


# Layer 1 - Wx+b
z1 = tf.matmul(W1, x) + b1
a1 = tf.nn.relu(z1)

# Layer 2 Wx+b
z2 = tf.matmul(W2, x) + b2
a2 = tf.nn.relu(z2)

# Prediction
y_hat = tf.matmul(W3, x) + b3

# Define loss
loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_hat))

# Trainig
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss_cross_entropy)

# Inilialize the Tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y: batch[1]})
  

    correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('test accuracy:', accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))



# Tasks
# 1. Run the code
# 2. What is this code doing?

