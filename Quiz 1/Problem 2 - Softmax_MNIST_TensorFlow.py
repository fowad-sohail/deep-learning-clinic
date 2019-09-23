
import tensorflow as tf
tf.set_random_seed(42)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print(mnist)


print()
# Display Training Data
print('Training Data:',)
print('Number of training examples: ', mnist.train.num_examples)
print('Shape of training inputs: ', mnist.train.images.shape)
print('Shape of training labels: ', mnist.train.labels.shape)

print()

# Display validation data
print('Validation Data:',)
print('Number of validation examples: ', mnist.validation.num_examples)
print('Shape of validation inputs: ', mnist.validation.images.shape)
print('Shape of validation labels: ', mnist.validation.labels.shape)

print()

print('Test Data:',)
print('Number of test examples: ', mnist.test.num_examples)
print('Shape of test inputs: ', mnist.test.images.shape)
print('Shape of labels labels: ', mnist.test.labels.shape)

print()

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# Initialize to zero
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Hypothesis
y_hat = tf.matmul(x,W) + b

# Calculate loss
loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_hat))

# Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss_cross_entropy)

# Execute the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y: batch[1]})
  

    correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('test accuracy:', accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))


# Tasks
# 1.    Make sure your code runs
# 2.    Change batch size to 100, 200, 500. Report your accuracy
# 3.    Run for various number of iterations, 10, 20, 1000, 1000, 10,000. Report your accuracy
# 4.    What is this code doing? What type of classifer is this?