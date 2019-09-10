#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""
from keras import backend as K
from keras.layers import Layer
from keras import constraints, regularizers


def flatten(data, mode):
    ndim = K.ndim(data)
    new_axes = list(range(mode, ndim)) + list(range(mode))
    data = K.permute_dimensions(data, new_axes)
    old_shape = K.shape(data)
    data = K.reshape(data, (old_shape[0], -1))
    
    return data, old_shape


def mode1_product(data, projection):
    data_shape = K.int_shape(data)
    data = K.permute_dimensions(data, (1, 2, 3, 0))
    data = K.reshape(data, (data_shape[1], -1))
    
    data = K.dot(K.transpose(projection), data)
    
    data = K.reshape(data, (K.int_shape(projection)[-1], data_shape[2], data_shape[3], -1))
    
    data = K.permute_dimensions(data, (3, 0, 1, 2))
    
    return data

def mode2_product(data, projection):
    data_shape = K.int_shape(data)
    data = K.permute_dimensions(data, (2, 3, 0, 1))
    data = K.reshape(data, (data_shape[2], -1))
    
    data = K.dot(K.transpose(projection), data)
    
    data = K.reshape(data, (K.int_shape(projection)[-1], data_shape[3], -1, data_shape[1]))
    
    data = K.permute_dimensions(data, (2, 3, 0, 1))
    
    return data

def mode3_product(data, projection):
    data_shape = K.int_shape(data)
    data = K.permute_dimensions(data, (3, 0, 1, 2))
    data = K.reshape(data, (data_shape[3], -1))
    
    data = K.dot(K.transpose(projection), data)
    
    data = K.reshape(data, (K.int_shape(projection)[-1], -1, data_shape[1], data_shape[2]))
    
    data = K.permute_dimensions(data, (1, 2, 3, 0))
    
    return data

class TensorEncoder(Layer):

    def __init__(self, 
                 encode_shape,
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        
        
        self.h = encode_shape[0]
        self.w = encode_shape[1]
        self.d = encode_shape[2]
        self.constraint = constraint
        self.regularizer = regularizer
        
        super(TensorEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        _, H, W, _ = input_shape

        
        if self.constraint is not None:
            constraint = constraints.max_norm(self.constraint, axis=0)
        else:
            constraint = None
            
        if self.regularizer is not None:
            regularizer = regularizers.l2(self.regularizer)
        else:
            regularizer = None
        
        self.P1_encode = self.add_weight(name='mode1_encode', 
                                          shape=(H, self.h),
                                          initializer='he_normal',
                                          constraint=constraint,
                                          regularizer=regularizer,
                                          trainable=True)
        
        self.P2_encode = self.add_weight(name='mode2_encode', 
                                      shape=(W, self.w),
                                      initializer='he_normal',
                                      constraint=constraint,
                                      regularizer=regularizer,
                                      trainable=True)
        
        self.P3_encode = self.add_weight(name='mode3_encode', 
                                      shape=(3, self.d),
                                      constraint=constraint,
                                      regularizer=regularizer,
                                      initializer='he_normal',
                                      trainable=True)
    

        
        super(TensorEncoder, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        
        encode = mode1_product(x, self.P1_encode)
        encode = mode2_product(encode, self.P2_encode)
        encode = mode3_product(encode, self.P3_encode)
        
        
        return encode

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.h, self.w, self.d)
    
    
class TensorSensing(Layer):

    def __init__(self, 
                 h,
                 w,
                 d,
                 linear_sensing=False,
                 end_to_end=False,
                 separate_decoder=True,
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        
        
        self.h = h
        self.w = w
        self.d = d
        self.linear_sensing = linear_sensing
        self.end_to_end = end_to_end
        self.separate_decoder = separate_decoder
        self.constraint = constraint
        self.regularizer = regularizer
        
        super(TensorSensing, self).__init__(**kwargs)

    def build(self, input_shape):
        _, H, W, _ = input_shape

        
        if self.constraint is not None:
            constraint = constraints.max_norm(self.constraint, axis=0)
        else:
            constraint = None
            
        if self.regularizer is not None:
            regularizer = regularizers.l2(self.regularizer)
        else:
            regularizer = None
        
        self.P1_encode = self.add_weight(name='mode1_encode', 
                                          shape=(H, self.h),
                                          initializer='he_normal',
                                          constraint=constraint,
                                          regularizer=regularizer,
                                          trainable=self.end_to_end)
        
        self.P2_encode = self.add_weight(name='mode2_encode', 
                                      shape=(W, self.w),
                                      initializer='he_normal',
                                      constraint=constraint,
                                      regularizer=regularizer,
                                      trainable=self.end_to_end)
        
        self.P3_encode = self.add_weight(name='mode3_encode', 
                                      shape=(3, self.d),
                                      constraint=constraint,
                                      regularizer=regularizer,
                                      initializer='he_normal',
                                      trainable=self.end_to_end)
    
        if self.separate_decoder:
            self.P1_decode = self.add_weight(name='mode1_decode', 
                                              shape=(self.h, H),
                                              initializer='he_normal',
                                              constraint=constraint,
                                              regularizer=regularizer,
                                              trainable=self.end_to_end)
            
            self.P2_decode = self.add_weight(name='mode2_decode', 
                                          shape=(self.w, W),
                                          initializer='he_normal',
                                          constraint=constraint,
                                          regularizer=regularizer,
                                          trainable=self.end_to_end)
            
            self.P3_decode = self.add_weight(name='mode3_decode', 
                                          shape=(self.d, 3),
                                          constraint=constraint,
                                          regularizer=regularizer,
                                          initializer='he_normal',
                                          trainable=self.end_to_end)
        
        super(TensorSensing, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        
        encode = mode1_product(x, self.P1_encode)
        encode = mode2_product(encode, self.P2_encode)
        encode = mode3_product(encode, self.P3_encode)
        
        if not self.linear_sensing:
            encode = K.relu(encode)
            
        if self.separate_decoder:
            decode = mode1_product(encode, self.P1_decode)
            decode = mode2_product(decode, self.P2_decode)
            decode = mode3_product(decode, self.P3_decode)
        else:
            decode = mode1_product(encode, K.transpose(self.P1_encode))
            decode = mode2_product(decode, K.transpose(self.P2_encode))
            decode = mode3_product(decode, K.transpose(self.P3_encode))
        
        return decode

    def compute_output_shape(self, input_shape):
        return input_shape
    

    

class BL(Layer):
    """
    Bilinear Layer
    """
    def __init__(self, output_dim,
                 kernel_regularizer=None,
                 kernel_constraint=None,**kwargs):

        
        self.output_dim = output_dim
        self.kernel_regularizer=kernel_regularizer
        self.kernel_constraint=kernel_constraint
        
        super(BL, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.kernel_regularizer is not None:
            regularizer = regularizers.l2(self.kernel_regularizer)
        else:
            regularizer = None
            
        if self.kernel_constraint is not None:
            constraint = constraints.max_norm(self.kernel_constraint, axis=0)
        else:
            constraint = None
            
        self.W1 = self.add_weight(name='W1',shape=(input_shape[1], self.output_dim[0]),
                                      initializer='he_uniform',
                                      regularizer=regularizer,
                                      constraint=constraint,
                                      trainable=True)
        self.W2 = self.add_weight(name='W2',shape=(input_shape[2], self.output_dim[1]),
                                      initializer='he_uniform',
                                      regularizer=regularizer,
                                      constraint=constraint,
                                      trainable=True)
        
        self.W3 = self.add_weight(name='W3',shape=(input_shape[3], self.output_dim[2]),
                                      initializer='he_uniform',
                                      regularizer=regularizer,
                                      constraint=constraint,
                                      trainable=True)
        
        super(BL, self).build(input_shape)

    def call(self, x):
        x = mode1_product(x,self.W1)
        x = mode2_product(x,self.W2)
        x = mode3_product(x,self.W3)
        
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1], self.output_dim[2])