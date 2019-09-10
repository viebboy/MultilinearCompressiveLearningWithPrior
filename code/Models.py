#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

import Layers
from keras.layers import Input, Conv2D, Dropout, Activation, Dense, Add, BatchNormalization as BN, AveragePooling2D, Flatten, GlobalAveragePooling2D, Concatenate
from keras.layers import UpSampling2D, Conv2DTranspose, MaxPooling2D
from keras import Model, regularizers, constraints
import tensorflow as tf
from keras.losses import mse as keras_mse
from keras.losses import mae as keras_mae
from keras.losses import kullback_leibler_divergence as keras_kl
from keras.losses import categorical_crossentropy as keras_ce
from keras import backend as K
import numpy as np


def allcnn_module(inputs, input_shape, prefix, regularizer, constraint):
    
    if regularizer is not None:
        conv_regularizer = regularizers.l2(regularizer)
    else:
        conv_regularizer = None

    if constraint is not None:
        conv_constraint = constraints.max_norm(constraint, axis=[0,1,2])
    else:
        conv_constraint= None
        
    if input_shape[0] == 32:
        nb_filter = 36
        filter_shape = (3,3)
        strides = (1,1)
    else:
        nb_filter = 16
        filter_shape = (5,5)
        strides = (2,2)
        
    if prefix is None:
        prefix = ''
    hiddens = Conv2D(nb_filter, filter_shape, strides=strides, padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name=prefix + 'allcnn_conv1_1')(inputs)
    hiddens = BN(name=prefix + 'allcnn_bn1_1')(hiddens)
    hiddens = Activation('relu')(hiddens)
    
    hiddens = Conv2D(96, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name=prefix + 'allcnn_conv1_2')(hiddens)
    hiddens = BN(name=prefix + 'allcnn_bn1_2')(hiddens)
    hiddens = Activation('relu')(hiddens)
    
    hiddens = Conv2D(96, (3,3), strides=(2,2), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name=prefix + 'allcnn_conv1_3')(hiddens)
    hiddens = BN(name=prefix + 'allcnn_bn1_3')(hiddens)
    hiddens = Activation('relu')(hiddens)
    
    #
    hiddens = Dropout(0.2)(hiddens)
    
    hiddens = Conv2D(192, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name=prefix + 'allcnn_conv2_1')(hiddens)
    hiddens = BN(name=prefix + 'allcnn_bn2_1')(hiddens)
    hiddens = Activation('relu')(hiddens)
    
    hiddens = Conv2D(192, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name=prefix + 'allcnn_conv2_2')(hiddens)
    hiddens = BN(name=prefix + 'allcnn_bn2_2')(hiddens)
    hiddens = Activation('relu')(hiddens)
    
    hiddens = Conv2D(192, (3,3), strides=(2,2), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name=prefix + 'allcnn_conv2_3')(hiddens)
    hiddens = BN(name=prefix + 'allcnn_bn2_3')(hiddens)
    hiddens = Activation('relu')(hiddens)

    #
    hiddens = Dropout(0.2)(hiddens)
    
    hiddens = Conv2D(192, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name=prefix + 'allcnn_conv3_1')(hiddens)
    hiddens = BN(name=prefix + 'allcnn_bn3_1')(hiddens)
    hiddens = Activation('relu')(hiddens)
    
    hiddens = Conv2D(192, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name=prefix + 'allcnn_conv3_2')(hiddens)
    hiddens = BN(name=prefix + 'allcnn_bn3_2')(hiddens)
    hiddens = Activation('relu')(hiddens)
    
    return hiddens

def get_baseline_allcnn(input_shape, n_class, regularizer=None, constraint=None):
    
    inputs = Input(input_shape, name='inputs')
    
    if regularizer is not None:
        conv_regularizer = regularizers.l2(regularizer)
    else:
        conv_regularizer = None

    if constraint is not None:
        conv_constraint = constraints.max_norm(constraint, axis=[0,1,2])
    else:
        conv_constraint= None
    
    
    hiddens = allcnn_module(inputs, input_shape, None, regularizer, constraint)
    
    hiddens = Conv2D(n_class, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv3_3')(hiddens)
    hiddens = BN(name='allcnn_bn3_3')(hiddens)
    hiddens = Activation('relu', name='activation3_3')(hiddens)
    
    hiddens = GlobalAveragePooling2D(name='average_pooling')(hiddens)
    
    outputs = Activation('softmax', name='allcnn_prediction')(hiddens)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    
    return model

def tensor_sensing_module(inputs,
                          h, 
                          w, 
                          d,
                          weight_decay,
                          weight_constraint):
    

    
    decode = Layers.TensorSensing(h=h,
                                  w=w,
                                  d=d,
                                  linear_sensing=True, 
                                  end_to_end=True, 
                                  separate_decoder=True,
                                  regularizer=weight_decay, 
                                  constraint=weight_constraint,
                                  name='sensing')(inputs)
        
    return decode

def get_MCL_allcnn(input_shape, 
                   n_class, 
                   h,
                   w,
                   d,
                   projection_regularizer=None,
                   projection_constraint=None, 
                   regularizer=None,
                   constraint=None):
    

    
    if regularizer is not None:
        conv_regularizer = regularizers.l2(regularizer)
    else:
        conv_regularizer = None

    if constraint is not None:
        conv_constraint = constraints.max_norm(constraint, axis=[0,1,2])
    else:
        conv_constraint= None
        
    inputs = Input(input_shape, name='inputs')
    
    decode = tensor_sensing_module(inputs,
                                   h, 
                                   w, 
                                   d,
                                   projection_regularizer,
                                   projection_constraint)
    
    hiddens = allcnn_module(decode, input_shape, None, regularizer, constraint)
    
    hiddens = Conv2D(n_class, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv3_3')(hiddens)
    
    hiddens = BN(name='allcnn_bn3_3')(hiddens)
    
    hiddens = Activation('relu', name='activation3_3')(hiddens)
    
    hiddens = GlobalAveragePooling2D(name='average_pooling')(hiddens)
    
    outputs = Activation('softmax', name='allcnn_prediction')(hiddens)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def get_linear_encoder(inputs, 
                       target_shape,
                       regularizer,
                       constraint):
    
    encode = Layers.TensorEncoder(target_shape, regularizer, constraint, name='linear_encoder')(inputs)
    
    return encode

def get_sensing_model(input_shape, 
                      target_shape):
    inputs = Input(input_shape)
    outputs = Layers.TensorEncoder(target_shape, name='linear_encoder')(inputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def upsampling_block(inputs,
                     nb_conv_filter1,
                     nb_conv_filter2,
                     nb_conv_layer,
                     prefix,
                     suffix,
                     regularizer, 
                     constraint,
                     upsample=True):
    
    if regularizer is not None:
        regularizer = regularizers.l2(regularizer)
    if constraint is not None:
        constraint = constraints.max_norm(constraint, axis=[0,1,2])
    
    for i in range(nb_conv_layer):
        conv_name = 'nonlinear_decoder_conv' + suffix + '_' + str(i) if prefix is None else prefix + 'nonlinear_decoder_conv' + suffix + '_' + str(i)
        bn_name = 'nonlinear_decoder_bn' + suffix + '_' + str(i) if prefix is None else prefix + 'nonlinear_decoder_bn' + suffix + '_' + str(i)
        inputs = Conv2D(nb_conv_filter1, (3,3), strides=(1,1), padding='same',
                         kernel_regularizer=regularizer,
                         kernel_constraint=constraint,
                         name=conv_name)(inputs)
        
        inputs = BN(name=bn_name)(inputs)
        
        inputs = Activation('relu')(inputs)
    
    if upsample:
        name = 'nonlinear_decoder_upsampling' + suffix if prefix is None else prefix + 'nonlinear_decoder_upsampling' + suffix
        outputs = Conv2DTranspose(nb_conv_filter2, (2,2), strides=(2,2), padding='same',
                                  kernel_regularizer=regularizer,
                                  kernel_constraint=constraint,
                                  name=name)(inputs)
    else:
        outputs = inputs
        
    return outputs

def get_nonlinear_decoder(inputs,
                          input_shape,
                          target_shape,
                          prefix=None,
                          complexity='low',
                          conv_regularizer=None,
                          conv_constraint=None,
                          bl_regularizer=None,
                          bl_constraint=None):
    
    h_scale = np.log2(np.floor(float(target_shape[0]) / input_shape[0]))
    w_scale = np.log2(np.floor(float(target_shape[1]) / input_shape[1]))
    
    # number of upsampling step
    nb_upsampling = max(1, int(max(h_scale, w_scale)))
    
    if complexity == 'low':
        nb_conv_per_block = 3 if input_shape[0] == input_shape[1] and input_shape[0] == 6 else 4
    else:
        nb_conv_per_block = 4 if input_shape[0] == input_shape[1] and input_shape[0] == 6 else 3
    
    nb_filter1 = 256
    nb_filter2 = 128
    
    for up_iter in range(nb_upsampling):
        suffix = '_' + str(up_iter)
        inputs = upsampling_block(inputs, 
                                  nb_filter1, 
                                  nb_filter2,
                                  nb_conv_per_block,
                                  prefix,
                                  suffix,
                                  conv_regularizer,
                                  conv_constraint)
        
        nb_filter1 = nb_filter2
        nb_filter2 = max(int(nb_filter2/2), 32)
        
    if nb_upsampling == 1 and complexity == 'high':
        inputs = upsampling_block(inputs, 
                                  nb_filter1, 
                                  nb_filter2,
                                  nb_conv_per_block,
                                  prefix,
                                  '_1',
                                  conv_regularizer,
                                  conv_constraint,
                                  False)
        
    name = 'nonlinear_decoder_bl' if prefix is None else prefix + 'nonlinear_decoder_bl'
    
    outputs = Layers.BL(target_shape, kernel_regularizer=bl_regularizer, kernel_constraint=bl_constraint, name=name)(inputs)
    
    return outputs


def downsampling_block(inputs,
                       nb_conv_filter1,
                       nb_conv_filter2,
                       nb_conv_layer,
                       prefix,
                       suffix,
                       regularizer, 
                       constraint,
                       downsample=True):
    
    if regularizer is not None:
        regularizer = regularizers.l2(regularizer)
    if constraint is not None:
        constraint = constraints.max_norm(constraint, axis=[0,1,2])
    
    for i in range(nb_conv_layer):
        conv_name = 'nonlinear_encoder_conv' + suffix + '_' + str(i) if prefix is None else prefix + 'nonlinear_encoder_conv' + suffix + '_' + str(i)
        bn_name = 'nonlinear_encoder_bn' + suffix + '_' + str(i) if prefix is None else prefix + 'nonlinear_encoder_bn' + suffix + '_' + str(i)
        inputs = Conv2D(nb_conv_filter1, (3,3), strides=(1,1), padding='same',
                         kernel_regularizer=regularizer,
                         kernel_constraint=constraint,
                         name=conv_name)(inputs)
        
        inputs = BN(name=bn_name)(inputs)
        
        inputs = Activation('relu')(inputs)
        
    if downsample:
        outputs = MaxPooling2D(pool_size=(2,2))(inputs)
    else:
        outputs = inputs
        
    return outputs

def get_nonlinear_encoder(inputs,
                          input_shape,
                          target_shape,
                          prefix=None,
                          complexity='low',
                          conv_regularizer=None,
                          conv_constraint=None,
                          bl_regularizer=None,
                          bl_constraint=None):
    
    h_scale = np.log2(np.floor(float(input_shape[0]) / target_shape[0]))
    w_scale = np.log2(np.floor(float(input_shape[1]) / target_shape[1]))
    
    # number of upsampling step
    nb_downsampling = max(1, int(min(h_scale, w_scale)))
    
    if complexity == 'low':
        nb_conv_per_block = 3 if target_shape[0] == target_shape[1] and target_shape[0] == 6 else 4
    else:
        nb_conv_per_block = 4 if target_shape[0] == target_shape[1] and target_shape[0] == 6 else 3
    
    
    nb_filter1 = 32
    nb_filter2 = 64
    
    for down_iter in range(nb_downsampling):
        suffix = '_' + str(down_iter)
        inputs = downsampling_block(inputs, 
                                    nb_filter1, 
                                    nb_filter2,
                                    nb_conv_per_block,
                                    prefix,
                                    suffix,
                                    conv_regularizer,
                                    conv_constraint)
        
        nb_filter1 = nb_filter2
        nb_filter2 = min(nb_filter2*2, 256)
        
    if nb_downsampling == 1 and complexity == 'high':
        inputs = downsampling_block(inputs,
                                    nb_filter1,
                                    nb_filter2,
                                    nb_conv_per_block,
                                    prefix,
                                    '_1',
                                    conv_regularizer,
                                    conv_constraint,
                                    False)

        
    name = 'nonlinear_encoder_bl' if prefix is None else prefix + 'nonlinear_encoder_bl'
    
    outputs = Layers.BL(target_shape, kernel_regularizer=bl_regularizer, kernel_constraint=bl_constraint, name=name)(inputs)
        
    return outputs

def get_autoencoder(input_shape, encode_shape, complexity, conv_regularizer, conv_constraint, bl_regularizer, bl_constraint):
    inputs = Input(input_shape)
    
    encode = get_nonlinear_encoder(inputs, input_shape, encode_shape, None, complexity, conv_regularizer, conv_constraint, bl_regularizer, bl_constraint)
    
    decode = get_nonlinear_decoder(encode, encode_shape, input_shape, None, complexity, conv_regularizer, conv_constraint, bl_regularizer, bl_constraint)
    
    model = Model(inputs=inputs, outputs=decode)
    
    return model

def get_prior_model_allcnn(input_shape, n_class, encode_shape, complexity, conv_regularizer, conv_constraint, bl_regularizer, bl_constraint):
        
    inputs = Input(input_shape, name='inputs')
    
    encode = get_nonlinear_encoder(inputs, input_shape, encode_shape, None, complexity, conv_regularizer, conv_constraint, bl_regularizer, bl_constraint)
    
    decode = get_nonlinear_decoder(encode, encode_shape, input_shape, None, complexity, conv_regularizer, conv_constraint, bl_regularizer, bl_constraint)
    
    hiddens = allcnn_module(decode, input_shape, None, conv_regularizer, conv_constraint)
    
    
    
    if conv_regularizer is not None:
        regularizer = regularizers.l2(conv_regularizer)
    else:
        regularizer = None
    if conv_constraint is not None:
        constraint = constraints.max_norm(conv_constraint, axis=[0,1,2])
    else:
        constraint = None
    hiddens = Conv2D(n_class, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=regularizer,
                     kernel_constraint=constraint,
                     name='allcnn_conv3_3')(hiddens)
    
    hiddens = BN(name='allcnn_bn3_3')(hiddens)
    hiddens = Activation('relu')(hiddens)
    
    hiddens = GlobalAveragePooling2D()(hiddens)
    
    outputs = Activation('softmax', name='allcnn_prediction')(hiddens)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def get_distillation_encoder(input_shape,
                             encode_shape,
                             loss_type,
                             complexity, 
                             conv_regularizer,
                             conv_constraint):
    
    inputs = Input(input_shape, name='inputs')
    
    linear_encode = get_linear_encoder(inputs, encode_shape, conv_regularizer, conv_constraint)
    nonlinear_encode = get_nonlinear_encoder(inputs, input_shape, encode_shape, 'fix_', complexity, None, None, None, None)
    
    model = Model(inputs=inputs, outputs=[linear_encode, nonlinear_encode])
    
    if loss_type == 'mse':
        reconstruction_loss = K.mean(keras_mse(linear_encode, nonlinear_encode))
    else:
        reconstruction_loss = K.mean(keras_mae(linear_encode, nonlinear_encode))
        
    model.add_loss(reconstruction_loss)
    
    return model

def get_distillation_encoder_decoder(input_shape,
                                     encode_shape,
                                     loss_type,
                                     complexity,
                                     conv_regularizer,
                                     conv_constraint,
                                     bl_regularizer,
                                     bl_constraint):
    
    inputs = Input(input_shape, name='inputs')
    
    """ trainable part """
    linear_encode = get_linear_encoder(inputs, encode_shape, conv_regularizer, conv_constraint)
    nonlinear_decode_trainable = get_nonlinear_decoder(linear_encode, 
                                                       encode_shape, 
                                                       input_shape, 
                                                       None, 
                                                       complexity, 
                                                       conv_regularizer, 
                                                       conv_constraint, 
                                                       bl_regularizer, 
                                                       bl_constraint)
    
    
    """ fix part """
    nonlinear_encode_fix = get_nonlinear_encoder(inputs, input_shape, encode_shape, 'fix_', complexity, None, None, None, None)
    nonlinear_decode_fix = get_nonlinear_decoder(nonlinear_encode_fix, encode_shape, input_shape, 'fix_', complexity, None, None, None, None)
    
    
    model = Model(inputs=inputs, outputs=[nonlinear_decode_trainable, nonlinear_decode_fix])
    
    if loss_type == 'mse':
        reconstruction_loss = K.mean(keras_mse(nonlinear_decode_trainable, nonlinear_decode_fix))
    else:
        reconstruction_loss = K.mean(keras_mae(nonlinear_decode_trainable, nonlinear_decode_fix))
        
    model.add_loss(reconstruction_loss)
    
    return model

def get_MCLwP_allcnn(input_shape,
                     n_class, 
                     encode_shape, 
                     distil_prediction,
                     distillation_coef,
                     complexity,
                     conv_regularizer, 
                     conv_constraint,
                     bl_regularizer,
                     bl_constraint):
        
    inputs = Input(input_shape, name='inputs')
    
    """ the trainable part """
    linear_encode = get_linear_encoder(inputs, encode_shape, conv_regularizer, conv_constraint)
    
    nonlinear_decode_trainable = get_nonlinear_decoder(linear_encode, 
                                                       encode_shape, 
                                                       input_shape, 
                                                       None,
                                                       complexity,
                                                       conv_regularizer, 
                                                       conv_constraint,
                                                       bl_regularizer,
                                                       bl_constraint)
    
    hiddens_trainable = allcnn_module(nonlinear_decode_trainable, input_shape, None, conv_regularizer, conv_constraint)
    
    hiddens_trainable = Conv2D(n_class, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=regularizers.l2(conv_regularizer) if conv_regularizer is not None else None,
                     kernel_constraint=constraints.max_norm(conv_constraint, axis=[0, 1, 2]) if conv_constraint is not None else None,
                     name='allcnn_conv3_3')(hiddens_trainable)
    
    hiddens_trainable = BN(name='allcnn_bn3_3')(hiddens_trainable)
    hiddens_trainable = Activation('relu')(hiddens_trainable)
    
    hiddens_trainable = GlobalAveragePooling2D()(hiddens_trainable)
    
    outputs_trainable = Activation('softmax', name='allcnn_prediction')(hiddens_trainable)
    
    """ fix part """
    nonlinear_encode = get_nonlinear_encoder(inputs, input_shape, encode_shape,'fix_', complexity, None, None, None, None)
    
    nonlinear_decode_fix = get_nonlinear_decoder(nonlinear_encode, 
                                                 encode_shape, 
                                                 input_shape, 
                                                 'fix_', 
                                                 complexity,
                                                 None, 
                                                 None,
                                                 None,
                                                 None)
    
    hiddens_fix = allcnn_module(nonlinear_decode_fix, input_shape, 'fix_', None, None)
    
    hiddens_fix = Conv2D(n_class, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=regularizers.l2(conv_regularizer) if conv_regularizer is not None else None,
                     kernel_constraint=constraints.max_norm(conv_constraint, axis=[0, 1, 2]) if conv_constraint is not None else None,
                     name='fix_allcnn_conv3_3')(hiddens_fix)
    
    hiddens_fix = BN(name='fix_allcnn_bn3_3')(hiddens_fix)
    hiddens_fix = Activation('relu')(hiddens_fix)
    
    hiddens_fix = GlobalAveragePooling2D()(hiddens_fix)
    
    outputs_fix = Activation('softmax', name='fix_allcnn_prediction')(hiddens_fix)
    
    model = Model(inputs=inputs, outputs=[outputs_trainable, outputs_fix])
    #model = Model(inputs=inputs, outputs=outputs_trainable)
    
    if distil_prediction:
        distillation_loss = K.mean(keras_kl(outputs_trainable, outputs_fix)) + K.mean(keras_kl(outputs_fix, outputs_trainable))
        
        model.add_loss(distillation_coef*distillation_loss)
    
    return model


def get_hybrid_autoencoder(input_shape,
                           encode_shape,
                           conv_regularizer,
                           conv_constraint,
                           bl_regularizer,
                           bl_constraint):
    
    inputs = Input(input_shape, name='inputs')
    
    linear_encode = get_linear_encoder(inputs, encode_shape, conv_regularizer, conv_constraint)
    
    nonlinear_decode_trainable = get_nonlinear_decoder(linear_encode, 
                                                       encode_shape, 
                                                       input_shape, 
                                                       None,
                                                       'low',
                                                       conv_regularizer, 
                                                       conv_constraint,
                                                       bl_regularizer,
                                                       bl_constraint)
    
    model = Model(inputs=inputs, outputs=nonlinear_decode_trainable)
    
    return model

def get_MCLwoP_allcnn(input_shape,
                      n_class, 
                      encode_shape, 
                      conv_regularizer, 
                      conv_constraint,
                      bl_regularizer,
                      bl_constraint):
        
    inputs = Input(input_shape, name='inputs')
    
    """ the trainable part """
    linear_encode = get_linear_encoder(inputs, encode_shape, conv_regularizer, conv_constraint)
    
    nonlinear_decode_trainable = get_nonlinear_decoder(linear_encode, 
                                                       encode_shape, 
                                                       input_shape, 
                                                       None,
                                                       'low',
                                                       conv_regularizer, 
                                                       conv_constraint,
                                                       bl_regularizer,
                                                       bl_constraint)
    
    hiddens_trainable = allcnn_module(nonlinear_decode_trainable, input_shape, None, conv_regularizer, conv_constraint)
    
    hiddens_trainable = Conv2D(n_class, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=regularizers.l2(conv_regularizer) if conv_regularizer is not None else None,
                     kernel_constraint=constraints.max_norm(conv_constraint, axis=[0, 1, 2]) if conv_constraint is not None else None,
                     name='allcnn_conv3_3')(hiddens_trainable)
    
    hiddens_trainable = BN(name='allcnn_bn3_3')(hiddens_trainable)
    hiddens_trainable = Activation('relu')(hiddens_trainable)
    
    hiddens_trainable = GlobalAveragePooling2D()(hiddens_trainable)
    
    outputs_trainable = Activation('softmax', name='allcnn_prediction')(hiddens_trainable)
    
    
    model = Model(inputs=inputs, outputs=outputs_trainable)
    
    return model
