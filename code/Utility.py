#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

import numpy as np
from keras.utils import to_categorical
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as K
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import exp_configurations as Conf
import itertools



class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)

    def next(self):     # Py2
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def create_configuration(hyperparameter_list, value_list):
    
    configurations = []
    
    for names, values in zip(hyperparameter_list, value_list):
        v_list = []
        for name in names:
            v_list.append(values[name])
        
        configurations += list(itertools.product(*v_list))
   
    configurations = list(set(configurations))
    configurations.sort()
    return configurations


def flatten(data, mode):
    
    ndim = data.ndim
    new_axes = list(range(mode, ndim)) + list(range(mode))
    
    data = np.transpose(data, axes=new_axes)
    
    old_shape = data.shape
    data = np.reshape(data, (old_shape[0], -1))
    
    return data, old_shape

def nmodeproduct(data, projection, mode):
    
    data, old_shape = flatten(data, mode)
    
    data = np.dot(projection.T, data)
    
    new_shape = list(old_shape)
    
    new_shape[0] = projection.shape[-1]
    
    
    data = np.reshape(data, new_shape)
    
    new_axes = list(range(data.ndim))
    new_axes = list(new_axes[-mode:]) + list(new_axes[:-mode])
    
    data = np.transpose(data, new_axes)
    
    return data
    
def MSE(a,b):
    return np.mean((a.flatten()-b.flatten())**2)
    
def HOSVD(data, h, w, d, centering=True, iteration=100, threshold=1e-4, regularization=0.0):

    
    N, H, W, _ = data.shape
        
    W1 = np.random.rand(H, h)
    W2 = np.random.rand(W,w)
    W3 = np.random.rand(3,d)
    
    if centering:
        mean_tensor = np.mean(data, axis=0, keepdims=True)
    else:
        mean_tensor = np.zeros(data.shape[1:])
        mean_tensor = np.expand_dims(mean_tensor, axis=0)
    
    data -= mean_tensor
    
    for i in range(iteration):
        print('iteration: %s' % str(i))
        """ compute W1 by fixing W2, W3"""
        # project mode 2, 3 -> data: N x H x w x 2
        data_tmp = nmodeproduct(data, W2, 2)
        data_tmp = nmodeproduct(data_tmp, W3, 3)
        
        # flatten to H x (N*w*2)
        data_tmp, _ = flatten(data, 1)
        cov = np.dot(data_tmp, data_tmp.T) + regularization*np.eye(H)
        U, _, _ = np.linalg.svd(cov)
        W1_new = U[:, :h]
        
        """ compute W2 by fixing W1, W3"""
        # project mode 1, 3 -> data: N x h x W x 2
        data_tmp = nmodeproduct(data, W1_new, 1)
        data_tmp = nmodeproduct(data_tmp, W3, 3)
        
        # flatten to W x (N*h*2)
        data_tmp, _ = flatten(data, 2)
        cov = np.dot(data_tmp, data_tmp.T) + regularization*np.eye(W)
        U, _, _ = np.linalg.svd(cov)
        W2_new = U[:, :w]
        
        """ compute W3 by fixing W1, W2"""
        # project mode 1, 2 -> data: N x h x w x 3
        data_tmp = nmodeproduct(data, W1_new, 1)
        data_tmp = nmodeproduct(data_tmp, W2_new, 2)
        
        # flatten to 3 x (N*h*w)
        data_tmp, _ = flatten(data, 3)
        cov = np.dot(data_tmp, data_tmp.T) + regularization*np.eye(3)
        U, _, _ = np.linalg.svd(cov)
        W3_new = U[:, :d]
        
        """ calculate error """
        data_tmp = nmodeproduct(data, W1, 1)
        data_tmp = nmodeproduct(data_tmp, W2, 2)
        data_tmp = nmodeproduct(data_tmp, W3, 3)
        
        data_tmp = nmodeproduct(data_tmp, W1.T, 1)
        data_tmp = nmodeproduct(data_tmp, W2.T, 2)
        data_tmp = nmodeproduct(data_tmp, W3.T, 3)
        
        print('Residual error: %.4f' % MSE(data_tmp, data))
        
        projection_error = MSE(W1, W1_new) + MSE(W2, W2_new) + MSE(W3, W3_new)
        print('Projection error: %.4f' % projection_error)
        
        W1 = W1_new
        W2 = W2_new
        W3 = W3_new
        
        if projection_error  < threshold:
            break
        
    return W1, W2, W3, mean_tensor
        
def load_data(name):
    prefix = Conf.data_dir
    x_train = np.load(os.path.join(prefix, name + '_x_train.npy'))
    y_train = np.load(os.path.join(prefix, name + '_y_train.npy'))
    x_val = np.load(os.path.join(prefix, name + '_x_val.npy'))
    y_val = np.load(os.path.join(prefix, name + '_y_val.npy'))
    x_test = np.load(os.path.join(prefix, name + '_x_test.npy'))
    y_test = np.load(os.path.join(prefix, name + '_y_test.npy'))

    
    n_class = np.unique(y_train).size
    
    y_train = to_categorical(y_train, n_class)
    y_val = to_categorical(y_val, n_class)
    y_test = to_categorical(y_test, n_class)
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_HOSVD_matrix(dataset, height, width, depth, centering):
    path = os.path.join(Conf.data_dir, 'HOSVD_%s_%s_%s_%s_%s.pickle' % (dataset, str(height), str(width), str(depth), str(centering)))
    
    if not os.path.exists(path):
        x_train = np.load(os.path.join(Conf.data_dir, dataset + '_x_train.npy'))
        W1, W2, W3, mean_tensor = HOSVD(x_train, height, width, depth, centering)
        fid = open(path, 'wb')
        projection = {'W1': W1, 'W2': W2, 'W3': W3, 'mean_tensor': mean_tensor}
        pickle.dump(projection, fid)
        fid.close()
        
    else:
        fid = open(path, 'rb')
        projection = pickle.load(fid)
        fid.close()
        W1, W2, W3, mean_tensor = projection['W1'], projection['W2'], projection['W3'], projection['mean_tensor']
    
    return W1, W2, W3, mean_tensor



def least_square(x, y):
    # x : N x D
    # y: N x d
    
    D = x.shape[-1]
    
    xTx = np.dot(x.T, x) + np.eye(D, D, dtype=np.float32)
    xTy = np.dot(x.T, y)
    W = np.dot(np.linalg.pinv(xTx), xTy)

    return W
    
def tensor_regression(x, model, input_shape, encode_shape, iterations=50):
    # x: N x H x W x D
    # y: N x h x w x d
    y = model.predict(x)
    h, w, d = encode_shape
    H, W, D = input_shape
    W1 = np.random.rand(H, h)
    W2 = np.random.rand(W, w)
    W3 = np.random.rand(D, d)
    
    y_pred = nmodeproduct(nmodeproduct(nmodeproduct(x, W1, 1), W2, 2), W3, 3)
    error = np.mean((y_pred.flatten()-y.flatten())**2)
    
    for i in range(iterations):
        """ solve for W1 """
        x_tmp = nmodeproduct(x, W2, 2)
        x_tmp = nmodeproduct(x_tmp, W3, 3) # x_tmp: N x H x w x d
        x_tmp = np.transpose(x_tmp, axes=(1, 0, 2, 3)) # x_tmp : H x N x w x d
        x_tmp = np.reshape(x_tmp, (H, -1))
        y_tmp = np.transpose(y, axes=(1, 0, 2, 3)) # y_tmp: h x N x w x d
        y_tmp = np.reshape(y_tmp, (h, -1))
        W1_new = least_square(x_tmp.T, y_tmp.T)
        
        """ solve for W2 """
        x_tmp = nmodeproduct(x, W1_new, 1)
        x_tmp = nmodeproduct(x_tmp, W3, 3) # x_tmp: N x h x W x d
        x_tmp = np.transpose(x_tmp, axes=(2, 0, 1, 3)) # x_tmp : H x N x w x d
        x_tmp = np.reshape(x_tmp, (W, -1))
        y_tmp = np.transpose(y, axes=(2, 0, 1, 3)) # y_tmp: h x N x w x d
        y_tmp = np.reshape(y_tmp, (w, -1))
        W2_new = least_square(x_tmp.T, y_tmp.T)
        
        """ solve for W3 """
        x_tmp = nmodeproduct(x, W1_new, 1)
        x_tmp = nmodeproduct(x_tmp, W2_new, 2) # x_tmp: N x h x W x d
        x_tmp = np.transpose(x_tmp, axes=(3, 0, 1, 2)) # x_tmp : H x N x w x d
        x_tmp = np.reshape(x_tmp, (D, -1))
        y_tmp = np.transpose(y, axes=(3, 0, 1, 2)) # y_tmp: h x N x w x d
        y_tmp = np.reshape(y_tmp, (d, -1))
        W3_new = least_square(x_tmp.T, y_tmp.T)
        
        y_pred = nmodeproduct(nmodeproduct(nmodeproduct(x, W1_new, 1), W2_new, 2), W3_new, 3)
        new_error = np.mean((y_pred.flatten()-y.flatten())**2)
        
        if np.abs(new_error - error) < 1e-4:
            break
    
        W1 = W1_new
        W2 = W2_new
        W3 = W3_new
        error = new_error
    
    return W1, W2, W3


def get_baseline_data_generator(x, y, batch_size=32, shuffle=False, augmentation=False):
    N = x.shape[0]
    steps = int(np.ceil(N/float(batch_size)))
    
    if augmentation:
        gen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    else:
        gen = ImageDataGenerator()
    
    gen.fit(x)
    
    return gen.flow(x, y, batch_size, shuffle), steps
        

def get_autoencoder_data_generator(x, batch_size=32, shuffle=False, augmentation=False):
    N = x.shape[0]
    steps = int(np.ceil(N/float(batch_size)))
    
    if augmentation:
        gen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    else:
        gen = ImageDataGenerator()
    
    gen.fit(x)
    gen_ = gen.flow(x, batch_size=batch_size, shuffle=shuffle)
    
    @threadsafe_generator
    def generator():
        while True:
            for i in range(steps):
                x_batch = next(gen_)
                yield x_batch, x_batch
    
    return generator(), steps


def get_distillation_data_generator(x, batch_size=32, shuffle=False, augmentation=False):
    N = x.shape[0]
    steps = int(np.ceil(N/float(batch_size)))

    if augmentation:
        gen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    else:
        gen = ImageDataGenerator()

    gen.fit(x)
    gen_ = gen.flow(x, batch_size=batch_size, shuffle=shuffle)

    @threadsafe_generator
    def generator():
        while True:
            for i in range(steps):
                x_batch = next(gen_)
                yield x_batch, None

    return generator(), steps


def get_sensing_data_generator(x, y, batch_size=32, shuffle=False, augmentation=False):
    N = x.shape[0]
    steps = int(np.ceil(N/float(batch_size)))
    
    if augmentation:
        gen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    else:
        gen = ImageDataGenerator()
    
    gen.fit(x)
    
    return gen.flow(x, y, batch_size, shuffle), steps






        
    
    
