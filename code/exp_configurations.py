#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""
import os

LR = (1e-3, 1e-3, 1e-4, 1e-5)
EPOCH = (40, 40, 40, 40)

DISTILLATION_LR = (1e-3, 1e-4, 1e-4, 1e-5)
DISTILLATION_EPOCH = (20, 30, 40, 40)

TRANSFER_LR = (1e-3, 1e-4, 1e-4, 1e-5)
TRANSFER_EPOCH = (20, 40, 60, 60)

"""

Configuration for test run

LR = (1e-3, 1e-3, 1e-4, 1e-5)
EPOCH = (1, 1)

DISTILLATION_LR = (1e-3, 1e-4, 1e-4, 1e-5)
DISTILLATION_EPOCH = (1, 1)

TRANSFER_LR = (1e-3, 1e-4, 1e-4, 1e-5)
TRANSFER_EPOCH = (1, 1)

"""

log_dir = os.path.join(os.path.dirname(os.getcwd()), 'log')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    
    
tmp_dir = os.path.join(os.path.dirname(os.getcwd()), 'tmpdir')
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
    
baseline_dir = os.path.join(os.path.dirname(os.getcwd()), 'baseline_models')
if not os.path.exists(baseline_dir):
    os.mkdir(baseline_dir)
    
output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')

assert os.path.exists(data_dir)

LOG_INTERVAL = 3
VAL_MEASURE_INTERVAL = 1
BATCH_SIZE = 32

BASELINE_REGULARIZATION = (1e-4, None)
PRIOR_REGULARIZATION = ((None, 6.0, None, 6.0))

baseline_names = ['dataset', 'classifier', 'regularizer', 'LR', 'Epoch']

baseline_conf = {'dataset': ['cifar10', 'cifar100', 'celebA32x32_100C', 'celebA32x32_200C', 'celebA32x32_500C', 
                             'cifar10_subset', 'cifar100_subset', 'celebA32x32_500C_subset'],
                 'classifier': ['allcnn',],
                 'regularizer': [BASELINE_REGULARIZATION],
                 'LR': [LR],
                 'Epoch': [EPOCH]}


mcl_names = ['prefix', 'dataset', 'classifier', 'encode_shape', 'projection_regularization', 'weight_regularization', 'exp', 'LR', 'Epoch', 'nb_neighbor']

mcl_conf = {'prefix': ['MCL'],
            'dataset': ['cifar10', 'cifar100', 'celebA32x32_100C', 'celebA32x32_200C', 'celebA32x32_500C',
                        'cifar10_subset', 'cifar100_subset', 'celebA32x32_500C_subset'],
            'classifier': ['allcnn'],
            'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
            'projection_regularization': [BASELINE_REGULARIZATION],
            'weight_regularization': [BASELINE_REGULARIZATION],
            'exp': [0, 1, 2],
            'LR': [LR],
            'Epoch': [EPOCH],
            'nb_neighbor': [(5, 20)]}

autoencoder_names = ['prefix', 'dataset', 'encode_shape', 'loss_type', 'complexity', 'use_unlabeled_data', 'regularization', 'LR', 'Epoch']

autoencoder_conf = {'prefix': ['autoencoder',],
                    'dataset': ['cifar10', 'cifar100', 'celebA32x32_100C', 'celebA32x32_200C', 'celebA32x32_500C'],
                    'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
                    'loss_type': ['mae'],
                    'complexity': ['low', 'high'],
                    'use_unlabeled_data': [False],
                    'regularization': [PRIOR_REGULARIZATION],
                    'LR': [LR],
                    'Epoch': [EPOCH]}

autoencoder_semi_supervised_conf = {'prefix': ['autoencoder',],
                                    'dataset': ['cifar10_subset', 'cifar100_subset', 'celebA32x32_500C_subset'],
                                    'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
                                    'loss_type': ['mae'],
                                    'complexity': ['low',],
                                    'use_unlabeled_data': [False, True],
                                    'regularization': [PRIOR_REGULARIZATION],
                                    'LR': [LR],
                                    'Epoch': [EPOCH]}

prior_names = ['prefix', 'dataset', 'classifier', 'encode_shape', 'loss_type', 'complexity', 'LR', 'Epoch']

prior_conf = {'prefix': ['prior'],
              'dataset': ['cifar10', 'cifar100', 'celebA32x32_100C', 'celebA32x32_200C', 'celebA32x32_500C',],
              'classifier': ['allcnn'],
              'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
              'loss_type': ['mae'],
              'complexity': ['low', 'high'],
              'LR': [LR],
              'Epoch': [EPOCH]}

prior_subset_conf = {'prefix': ['prior'],
              'dataset': ['cifar10_subset', 'cifar100_subset', 'celebA32x32_500C_subset',],
              'classifier': ['allcnn'],
              'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
              'loss_type': ['mae'],
              'complexity': ['low',],
              'LR': [LR],
              'Epoch': [EPOCH]}

mclwp_names = ['prefix', 'dataset', 'classifier', 'encode_shape', 'regularization', 'distillation_loss', 'distillation_coef',
               'D_LR', 'D_Epoch', 'prior_LR', 'prior_Epoch', 'LR', 'Epoch', 'exp', 'distill_encoder', 'distill_encoder_decoder', 
               'distill_prediction', 'complexity', 'nb_neighbor']

mclwp_conf = {'prefix': ['MCLwP'],
              'dataset': ['cifar10', 'cifar100', 'celebA32x32_100C', 'celebA32x32_200C', 'celebA32x32_500C'],
              'classifier': ['allcnn'],
              'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
              'regularization': [PRIOR_REGULARIZATION],
              'distillation_loss': ['mae'],
              'distillation_coef': [1.0],
              'D_LR': [DISTILLATION_LR],
              'D_Epoch': [DISTILLATION_EPOCH],
              'prior_LR': [LR],
              'prior_Epoch': [EPOCH],
              'LR': [LR],
              'Epoch': [EPOCH],
              'exp': [0, 1, 2],
              'distill_encoder': [True, False],
              'distill_encoder_decoder': [True, False],
              'distill_prediction': [True, False],
              'complexity': ['low'],
              'nb_neighbor': [(5, 20)]}

mclwp_complexity_conf = {'prefix': ['MCLwP'],
                         'dataset': ['cifar10', 'cifar100', 'celebA32x32_100C', 'celebA32x32_200C', 'celebA32x32_500C'],
                         'classifier': ['allcnn'],
                         'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
                         'regularization': [PRIOR_REGULARIZATION],
                         'distillation_loss': ['mae'],
                         'distillation_coef': [1.0],
                         'D_LR': [DISTILLATION_LR],
                         'D_Epoch': [DISTILLATION_EPOCH],
                         'prior_LR': [LR],
                         'prior_Epoch': [EPOCH],
                         'LR': [LR],
                         'Epoch': [EPOCH],
                         'exp': [0, 1, 2],
                         'distill_encoder': [True,],
                         'distill_encoder_decoder': [True,],
                         'distill_prediction': [True,],
                         'complexity': ['high'],
                         'nb_neighbor': [(),]}

mclwp_semi_supervised_conf = {'prefix': ['MCLwP'],
                         'dataset': ['cifar10_subset', 'cifar100_subset', 'celebA32x32_500C_subset'],
                         'classifier': ['allcnn'],
                         'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
                         'regularization': [PRIOR_REGULARIZATION],
                         'distillation_loss': ['mae'],
                         'distillation_coef': [1.0],
                         'D_LR': [DISTILLATION_LR],
                         'D_Epoch': [DISTILLATION_EPOCH],
                         'prior_LR': [LR],
                         'prior_Epoch': [EPOCH],
                         'LR': [LR],
                         'Epoch': [EPOCH],
                         'exp': [0, 1, 2],
                         'distill_encoder': [True,],
                         'distill_encoder_decoder': [True,],
                         'distill_prediction': [True,],
                         'complexity': ['low'],
                         'nb_neighbor': [(),]}


mclwop_names = ['prefix', 'dataset', 'classifier', 'encode_shape', 'regularization', 'exp', 'LR', 'Epoch']
mclwop_conf = {'prefix': ['MCLwoP'],
               'dataset': ['cifar10', 'cifar100', 'celebA32x32_100C', 'celebA32x32_200C', 'celebA32x32_500C'],
               'classifier': ['allcnn'],
               'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
               'regularization': [PRIOR_REGULARIZATION],
               'exp': [0, 1, 2],
               'LR': [LR],
               'Epoch': [EPOCH]}


prior_semi_supervised_names = ['prefix', 'dataset', 'classifier', 'encode_shape', 'distillation_loss', 'LR', 'Epoch', 'confidence_score']
prior_semi_supervised_conf = {'prefix': ['priorS'],
                              'dataset': ['cifar10_subset', 'cifar100_subset', 'celebA32x32_500C_subset'],
                              'classifier': ['allcnn'],
                              'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
                              'distillation_loss': ['mae'],
                              'LR': [LR],
                              'Epoch': [EPOCH],
                              'confidence_score': [0.7, 0.8, 0.9]}


mclwps_names = ['prefix', 'dataset', 'classifier', 'encode_shape', 'regularization', 'distillation_loss', 'distillation_coef',
                'D_LR', 'D_Epoch', 'prior_LR', 'prior_Epoch', 'LR', 'Epoch', 'exp', 'confidence_score']

mclwps_conf = {'prefix': ['MCLwPS'],
               'dataset': ['cifar10_subset', 'cifar100_subset', 'celebA32x32_500C_subset'],
               'classifier': ['allcnn'],
               'encode_shape': [(6, 6, 1), (9, 7, 1), (13, 12, 1), (14, 11, 2)],
               'regularization': [PRIOR_REGULARIZATION],
               'distillation_loss': ['mae'],
               'distillation_coef': [1.0],
               'D_LR': [DISTILLATION_LR],
               'D_Epoch': [DISTILLATION_EPOCH],
               'prior_LR': [LR],
               'prior_Epoch': [EPOCH],
               'LR': [LR],
               'Epoch': [EPOCH],
               'exp': [0, 1, 2],
               'confidence_score': [0.7, 0.8, 0.9]}


