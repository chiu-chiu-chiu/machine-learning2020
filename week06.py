# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:24:49 2020

@author: Acc
"""

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
import math as m
from keras import regularizers

num = int(3e5)
g = 9.807
velocity = np.random.uniform(1, 100, [num, 1])
theta = np.random.uniform(0.01, m.pi / 2, [num, 1])
height = (velocity * np.sin(theta)) ** 2 / (2 * g)
distance = velocity ** 2 * np.sin(2 * theta) / g
x = np.hstack((velocity, theta))
y = np.hstack((height, distance))
from sklearn import preprocessing


def norm(data):
    a = preprocessing.StandardScaler().fit(data)
    d = a.transform(data)
    m = a.mean_
    s = a.scale_
    v = a.var_
    return d, m, v, s


x1, m1, v1, s1 = norm(x.reshape(-1, 2))
y1, m2, v2, s2 = norm(y.reshape(-1, 2))

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(2,)))
# model.add(BatchNormalization())                                   BN
# model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))       L2
# model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2))

import keras.backend as K


def tran(y_true, y_pred):
    re1 = tf.math.add(tf.math.multiply(tf.math.divide(y_true, s2), v2), m2)
    re2 = tf.math.add(tf.math.multiply(tf.math.divide(y_pred, s2), v2), m2)
    return tf.math.reduce_mean(tf.math.square((tf.math.subtract(re1, re2))))


index = np.arange(num)
np.random.shuffle(index)
model.compile(loss='mae', optimizer='adam',
              metrics=[tran])
model.fit(x1[index[int(num * 0.2):]], y1[index[int(num * 0.2):]],
          validation_data=(x1[index[:int(num * 0.2)]], y1[index[:int(num * 0.2)]]),
          epochs=100, batch_size=128)
# %%
tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=10, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None,
)
# %%
