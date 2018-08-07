from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import os
import pandas as pd
import itertools
from scipy.interpolate import interp1d
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data
import random

repeat_times = 5
spline_time = 225

subjects = ['Shohei', 'Kazusi', 'Tatsuki']
movementNames = ['a', 'b', 'c', 'd', 'e']

def get_WBB_data():
    cwd = os.getcwd()
    ii =0
    data_base = np.zeros((225, 7, 75))
    for i in range(len(subjects)):
        for j in range(len(movementNames)):
            for k in range(repeat_times):
                pathBase = cwd + '\\Data\\{0}\\{1}_{2}.txt'.format(subjects[i], movementNames[j], k+1)
                one_data = pd.read_csv(pathBase, delimiter=",").as_matrix()
                one_data = one_data[100:325, :]
                data_base[:, :, ii] = one_data
                ii = ii+1
    data_base_matrix = data_base
    data_base_matrix = data_base_matrix[:, 1:7, :]

    return data_base_matrix

def get_Data():
    # obtaining test_x
    test_x = data_base_matrix[:, :, 0:75:5]
    c = np.zeros((1, spline_time, 6))
    for ii in range(15):
        a = test_x[:, :, ii]
        a = a.reshape(1, spline_time, 6)
        c = np.vstack((c, a))
    c = np.delete(c, 0, axis=0)
    test_x = c
    test_x = test_x[:, :, :, np.newaxis]

    # obtaining train_x
    train_x = np.zeros((spline_time, 6, 60))
    ii = 0
    for i in range(75):
        if (i % 5 != 0):
            train_x[:, :, ii] = data_base_matrix[:, :, i]
            ii = ii + 1
    c = np.zeros((1, spline_time, 6))
    for ii in range(60):
        a = train_x[:, :, ii]
        a = a.reshape(1, spline_time, 6)
        c = np.vstack((c, a))
    c = np.delete(c, 0, axis=0)
    train_x = c
    train_x = train_x[:, :, :, np.newaxis]

    # obtaining one_hots_test
    eye = np.eye(5)
    eye2 = np.vstack((eye, eye))
    one_hots_test = np.vstack((eye2, eye))

    # obtaining one_hots_train
    vec1 = [1.0, 0.0, 0.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0, 0.0, 0.0]
    vec3 = [0.0, 0.0, 1.0, 0.0, 0.0]
    vec4 = [0.0, 0.0, 0.0, 1.0, 0.0]
    vec5 = [0.0, 0.0, 0.0, 0.0, 1.0]
    one_hots_train = np.zeros((1, 5))
    for j in range(60):
        if (j % 20 <= 3):
            one_hots_train = np.vstack((one_hots_train, vec1))
        elif (4 <= j % 20 <= 7):
            one_hots_train = np.vstack((one_hots_train, vec2))
        elif (8 <= j % 20 <= 11):
            one_hots_train = np.vstack((one_hots_train, vec3))
        elif (12 <= j % 20 <= 15):
            one_hots_train = np.vstack((one_hots_train, vec4))
        elif (16 <= j % 20 <= 19):
            one_hots_train = np.vstack((one_hots_train, vec5))
    one_hots_train = np.delete(one_hots_train, 0, axis=0)

    return train_x, one_hots_train, test_x, one_hots_test

def dense(input, name, in_size, out_size, activation="relu"):

    with tf.variable_scope(name, reuse=False):
        w = tf.get_variable("w", shape=[in_size, out_size],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable("b", shape=[out_size], initializer=tf.constant_initializer(0.0))
        p = 0.5
        l = tf.add(tf.matmul(input, w), b)
        #l = tf.add(tf.matmul(tf.nn.dropout(input, keep_prob=p), w) + b)

        if activation == "relu":
            l = tf.nn.relu(l)
        elif activation == "sigmoid":
            l = tf.nn.sigmoid(l)
        elif activation == "tanh":
            l = tf.nn.tanh(l)
        else:
            l = l
        print(l)
    return l

def scope(y, y_, learning_rate=10^-5):

    #Learning rate
    learning_rate = tf.Variable(learning_rate,  trainable=False)

    # Loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=y_), name="loss")

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       name="optimizer").minimize(loss)

    # Evaluate the model
    correct = tf.equal(tf.cast(tf.argmax(y_, 1), tf.int32),
                       tf.cast(tf.argmax(y, 1), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    #  Tensorboard
    writer = tf.summary.FileWriter('./Tensorboard/')
    # run this command in the terminal to launch tensorboard:
    # tensorboard --logdir=./Tensorboard/
    writer.add_graph(graph=sess.graph)

    return loss, accuracy, optimizer, writer

def confusion_matrix(cm, accuracy):

    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size=15)

data_base_matrix = get_WBB_data()

train_x, one_hots_train, test_x, one_hots_test = get_Data()

number_test = [one_hots_test[i, :].argmax() for i in range(0, one_hots_test.shape[0])]

n_label = 5  # Number of class
height = spline_time
width = 6

# Session and context manager
tf.reset_default_graph()
sess = tf.Session()

with tf.variable_scope(tf.get_variable_scope()):

    # Placeholders
    x = tf.placeholder(tf.float32, [None, height, width, 1], name='X')
    y = tf.placeholder(tf.float32, [None, n_label], name='Y')

    # Convolutional Neural network
    c1 = tf.layers.conv2d(inputs=x, kernel_size=[10, 1], strides=[2, 1],
                          filters=16, padding='SAME', activation=tf.nn.relu,
                          name="conv_1")
    print(c1)

    c1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[6, 1],
                                 strides=[1, 1], padding='SAME')
    print(c1)

    c2 = tf.layers.conv2d(inputs=c1, kernel_size=[5, 1], strides=[2, 1],
                          filters=32, padding='SAME', activation=tf.nn.relu,
                          name="conv_2")
    print(c2)

    c2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[3, 1],
                                 strides=[1, 1], padding='SAME')
    print(c2)

    # Reshape to a fully connected layers
    size = c2.get_shape().as_list()

    l1 = tf.reshape(c2, [-1, size[1] * size[2] * size[3]], name='reshape_to_fully')

    l2 = dense(input=l1, name='layer_1',
               in_size=l1.get_shape().as_list()[1], out_size=8192, activation='relu')

    l3 = dense(input=l2, name="layer_2", in_size=8192, out_size=4096, activation="relu")
    l4 = dense(input=l3, name="layer_3", in_size=4096, out_size=2048, activation="relu")
    l5 = dense(input=l4, name="layer_4", in_size=2048, out_size=1024, activation="relu")
    l6 = dense(input=l5, name="layer_5", in_size=1024, out_size=512, activation="relu")
    l7 = dense(input=l6, name="layer_6", in_size=512, out_size=64, activation="relu")
    l8 = dense(input=l7, name="layer_7", in_size=64, out_size=16, activation="relu")
    l9 = dense(input=l8, name="output_layer", in_size=16, out_size=n_label, activation="None")


    # Softmax layer
    y_ = tf.nn.softmax(l9, name="softmax")

    # Scope
    loss, accuracy, optimizer, writer = scope(y, y_, learning_rate=10^-5)

    # Initialize the Neural Network
    sess.run(tf.global_variables_initializer())

    # Train the Neural Network
    loss_history = []
    acc_history = []
    epoch = 50
    train_data = {x: train_x, y: one_hots_train}

    for e in range(epoch):

        _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict=train_data)

        loss_history.append(l)
        acc_history.append(acc)

        print("Epoch " + str(e) + " - Loss: " + str(l) + " - " + str(acc))

plt.figure()
plt.plot(acc_history)

# Test the trained Neural Network

test_data = {x: test_x, y: one_hots_test}
l, acc = sess.run([loss, accuracy], feed_dict=test_data)
print("Test - Loss: " + str(l) + " - " + str(acc))

# Confusion matrix

predictions = y_.eval(feed_dict=test_data, session=sess)
predictions_int = (predictions == predictions.max(axis=1, keepdims=True)).astype(int)
predictions_numbers = [predictions_int[i, :].argmax() for i in range(0, predictions_int.shape[0])]

cm = metrics.confusion_matrix(number_test, predictions_numbers)
print(cm)
confusion_matrix(cm=cm, accuracy=acc)
cmN = cm / cm.sum(axis=0)
confusion_matrix(cm=cmN, accuracy=acc)