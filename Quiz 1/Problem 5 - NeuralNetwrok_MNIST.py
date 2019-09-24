
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.set_random_seed(42)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)


X_train = mnist.train.images.T
X_test = mnist.test.images.T

y_train = mnist.train.labels
y_test = mnist.test.labels


# Select a subset of traing examples
X_train = X_train[:,0:5000]
y_train = y_train[0:5000]

# Select a subset of test examples
X_test = X_test[:,0:1000]
y_test = y_test[0:1000]

n_hidden = 10                                         # Number of hidden units
alpha = 0.1                                           # Learning rate
n_iterations = 200
num_classes = 10

# Data info
m = X_train.shape[1]
num_features = X_train.shape[0]


print ("Number of training examples: " + str(m))
print ("Number of test examples: " + str(y_test.shape[0]))
print ("Number of features = " + str(num_features))
print ("Number of classes = " + str(num_classes))


# initialize parameters randomly
np.random.seed(42)                                      # control the random initializatioon
W1 = 0.01 * np.random.randn(n_hidden, num_features)
b1 = np.zeros((n_hidden, 1))
W2 = 0.01 * np.random.randn(num_classes, n_hidden)
b2 = np.zeros((num_classes, 1))

costs=[]
acc_train = []
acc_test = []

# Function to calcualte accuracies, i.e., just the forward pass
def fGetAccuracy(W1, W2, b1, b2, X_data, y_data):
    scores = np.dot(W2, np.maximum(0, np.dot(W1, X_data) + b1)) + b2
    predicted_class = np.argmax(scores, axis=0)
    accuracy = np.mean(predicted_class == y_data)
    return accuracy

# derivative of ReLU
def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

# Start iterations
for i in range(n_iterations):
  
    Z1 = np.dot(W1, X_train) + b1
    A1 = np.maximum(0, Z1)                                                        # Relu nonlinearity
    Z2 = np.dot(W2, A1) + b2
    
    # Softmax calculation
    # compute the class probabilities
    exp_scores = np.exp(Z2)
    probs_all = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)            # Get all probabilities
    probs_correct = probs_all.T[range(m),y_train]                                 # Keep only that we want to maximize
    logprobs_correct = -np.log(probs_correct)                        
    
    loss = np.sum(logprobs_correct)/m                                        
    
    costs.append(loss)
    if i % 100 == 0:
        print('------------------------------------')
        print("iteration %d: loss %f" % (i, loss))
        
        accuracy = fGetAccuracy(W1, W2, b1, b2, X_train, y_train)
        print('training accuracy: %.2f' % accuracy)
        acc_train.append(accuracy) 

        accuracy = fGetAccuracy(W1, W2, b1, b2, X_test, y_test)
        print('test accuracy: %.2f' % accuracy)
        acc_test.append(accuracy)
        
    # Backpropagation
    dZ2 = exp_scores - y_train
    dW2 = 1/m * np.dot(dZ2, A1.T)
    dZ1 = np.dot(W2.T, dZ2) * np.dot(reluDerivative(X_train), Z1.T)
    db2 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    
    dW1 = np.dot(dZ1, X_train.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    
    # perform a parameter update
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    
    
    
# evaluate accuracies
print('------------------------------------')
print('Accuracies')
accuracy = fGetAccuracy(W1, W2, b1, b2, X_train, y_train)
print('training accuracy: %.2f' % accuracy)

accuracy = fGetAccuracy(W1, W2, b1, b2, X_test, y_test)
print('training accuracy: %.2f' % accuracy)

plt.figure(0)
plt.plot(costs)
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.figure(1)
plt.plot(acc_train, label='train')
plt.plot(acc_test, label='test')
plt.title('Training Accuraacy')
plt.ylabel('accuracy')
plt.xlabel('epoch*100')

plt.legend()



# Tasks
# Run the code.
# Report your train and test accuracy/loss plots
