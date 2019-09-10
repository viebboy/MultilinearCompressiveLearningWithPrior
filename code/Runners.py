#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

import Models
import Utility, numpy as np, os
import pickle
import exp_configurations as Configuration
from sklearn.neighbors import KNeighborsClassifier

OUTPUT_DIR = Configuration.output_dir
BASELINE_DIR = Configuration.baseline_dir
LOG_DIR = Configuration.log_dir
DATA_DIR = Configuration.data_dir
BATCH_SIZE = Configuration.BATCH_SIZE

def train_baseline(args):
    
    result_filename = '_'.join([str(v) for v in args]) + '.pickle'
    result_filename = os.path.join(BASELINE_DIR, result_filename)
    
    if os.path.exists(result_filename):
        fid = open(result_filename, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result
    
    dataset, classifier, regularizer, LR, Epoch = args
    
    weight_decay, constraint = regularizer
    
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    
    train_gen, train_steps = Utility.get_baseline_data_generator(x_train, y_train, shuffle=True, augmentation=True, batch_size=BATCH_SIZE)
    val_gen, val_steps = Utility.get_baseline_data_generator(x_val, y_val, shuffle=False, augmentation=False, batch_size=BATCH_SIZE)
    test_gen, test_steps = Utility.get_baseline_data_generator(x_test, y_test, shuffle=False, augmentation=False, batch_size=BATCH_SIZE)
    
    input_shape = x_train.shape[1:]
    n_class = y_train.shape[-1]
    
    current_weights = None
    optimal_weights = None
    optimal_acc = 0.0
    
    history = {'train_categorical_crossentropy':[], 'train_acc':[], 'val_acc': [], 'val_categorical_crossentropy':[]}
    
    get_model = Models.get_baseline_allcnn
    
    for lr, epoch in zip(LR, Epoch):
        model = get_model(input_shape=input_shape,
                          n_class=n_class,
                          regularizer=weight_decay,
                          constraint=constraint)
        
        model.compile('adam', 'categorical_crossentropy', ['acc',])
        
        if current_weights is not None:
            model.set_weights(current_weights)
        
        model.optimizer.lr = lr
        
        for iteration in range(epoch):
            h = model.fit_generator(train_gen, train_steps, epochs=1, verbose=0)
            
            val_performance = model.evaluate_generator(val_gen, val_steps)
            acc_index = model.metrics_names.index('acc')
            loss_index = model.metrics_names.index('loss')
            
            if val_performance[acc_index] > optimal_acc:
                optimal_acc = val_performance[acc_index]
                optimal_weights = model.get_weights()
                
            history['train_categorical_crossentropy'] += h.history['loss']
            history['train_acc'] += h.history['acc']
            history['val_categorical_crossentropy'].append(val_performance[loss_index])
            history['val_acc'].append(val_performance[acc_index])
            
        
        current_weights = model.get_weights()
    
    model.set_weights(optimal_weights)
    
    train_p = model.evaluate_generator(train_gen, train_steps)
    val_p = model.evaluate_generator(val_gen, val_steps)
    test_p = model.evaluate_generator(test_gen, test_steps)
    
    train_performance = {}
    val_performance = {}
    test_performance = {}
    
    for index, metric in enumerate(model.metrics_names):
        train_performance[metric] = train_p[index]
        val_performance[metric] = val_p[index]
        test_performance[metric] = test_p[index]
        
    weights = {}
    for layer in model.layers:
        if layer.name.startswith(classifier):
            weights[layer.name] = layer.get_weights()
    
    
    data = {'weights': weights,
            'train_acc': train_performance['acc'],
            'val_acc': val_performance['acc'],
            'test_acc': test_performance['acc'],
            'train_categorical_crossentropy': train_performance['loss'],
            'val_categorical_crossentropy': val_performance['loss'],
            'test_categorical_crossentropy': test_performance['loss'],
            'history': history}
    
    return data

def train_MCL(args):
    prefix = args[0]
    dataset = args[1]
    classifier = args[2]
    target_shape = args[3]
    height, width, depth = target_shape
    projection_regularizer = args[4]
    weight_regularizer = args[5]
    projection_decay, projection_constraint = projection_regularizer
    weight_decay, weight_constraint = weight_regularizer
    exp = args[6]
    LR = args[7]
    Epoch = args[8]
    nb_neighbor = args[9]
    
    # if exists result, load and return
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Configuration.output_dir, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        outputs = pickle.load(fid)
        fid.close()
        return outputs
    
    # load baseline model
    baseline_filename = '_'.join([str(dataset), 
                                  str(classifier), 
                                  str(Configuration.BASELINE_REGULARIZATION), 
                                  str(Configuration.LR), 
                                  str(Configuration.EPOCH)]) + '.pickle'
    
    baseline_filename = os.path.join(BASELINE_DIR, baseline_filename)
    
    if not os.path.exists(baseline_filename):
        raise RuntimeError('Baseline model doesnt exist!')
    
    with open(baseline_filename, 'rb') as fid:
        baseline_model = pickle.load(fid)
    
    baseline_weights = baseline_model['weights']
    
    # load dataset
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    input_shape = x_train.shape[1:]
    n_class = y_train.shape[-1]
    
    train_gen, train_steps = Utility.get_baseline_data_generator(x_train,
                                                        y_train,
                                                        shuffle=True,
                                                        augmentation=True,
                                                        batch_size=BATCH_SIZE)
    train_gen_fix, train_steps_fix = Utility.get_baseline_data_generator(x_train,
                                                                y_train,
                                                                shuffle=False,
                                                                augmentation=False,
                                                                batch_size=BATCH_SIZE)
    val_gen, val_steps = Utility.get_baseline_data_generator(x_val,
                                                    y_val,
                                                    shuffle=False,
                                                    augmentation=False,
                                                    batch_size=BATCH_SIZE)
    
    test_gen, test_steps = Utility.get_baseline_data_generator(x_test,
                                                               y_test,
                                                               shuffle=False,
                                                               augmentation=False,
                                                               batch_size=BATCH_SIZE)
    

    # handle sensing matrix
    W1, W2, W3, data_mean = Utility.load_HOSVD_matrix(dataset, height, width, depth, False)
        
    # training
    
    get_model = Models.get_MCL_allcnn
        
    model = get_model(input_shape=input_shape, 
                     n_class=n_class, 
                     h=height,
                     w=width,
                     d=depth,
                     projection_regularizer=projection_decay, 
                     projection_constraint=projection_constraint, 
                     regularizer=weight_decay, 
                     constraint=weight_constraint)

    model.compile('adam', 'categorical_crossentropy', ['acc'])
    metrics = model.metrics_names
    
    log_file = '_'.join([str(v) for v in args]) + '.pickle'
    log_file = os.path.join(LOG_DIR, log_file)
    
    # if not exist log file, start from beginning, otherwise load the log file
    if not os.path.exists(log_file):    
        # set the pretrained weights
        for layer_name in baseline_weights.keys():
            model.get_layer(layer_name).set_weights(baseline_weights[layer_name])
        
        # set the sensing matrix
        model.get_layer('sensing').set_weights([W1, W2, W3, W1.T, W2.T, W3.T])    
        
        current_weights = model.get_weights()        
        optimal_weights = model.get_weights() 
        train_acc_list = []
        val_acc_list = []
        test_acc_list = []
        
        train_p = model.evaluate_generator(train_gen_fix, train_steps_fix)
        val_p = model.evaluate_generator(val_gen, val_steps)
        test_p = model.evaluate_generator(test_gen, test_steps)
        
        train_acc_list.append(train_p[metrics.index('acc')])
        val_acc_list.append(val_p[metrics.index('acc')])
        test_acc_list.append(test_p[metrics.index('acc')])
        
        validation_measure = val_acc_list[0]
        
        history = {'train_acc':[], 'train_categorical_crossentropy':[], 'val_acc':[], 'val_categorical_crossentropy':[]}
        last_index = 0
    else:
        fid = open(log_file, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        validation_measure = log_data['validation_measure']
        train_acc_list = log_data['train_acc_list']
        val_acc_list = log_data['val_acc_list']
        test_acc_list = log_data['test_acc_list']
        history = log_data['history']
        last_index = log_data['last_index']
        
    learning_rates = []
    for lr, sc in zip(LR, Epoch):
        learning_rates += [lr,]*sc
        
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam', 'categorical_crossentropy', ['acc', ])
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
            
        h = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)

        val_p = model.evaluate_generator(val_gen, val_steps)
        
        if val_p[metrics.index('acc')] > validation_measure:
            validation_measure = val_p[metrics.index('acc')]
            optimal_weights = model.get_weights()
            
        history['train_acc'] += h.history['acc']
        history['train_categorical_crossentropy'] += h.history['loss']
        history['val_acc'].append(val_p[metrics.index('acc')])
        history['val_categorical_crossentropy'].append(val_p[metrics.index('loss')])
        
        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'validation_measure': validation_measure,
                        'history': history,
                        'train_acc_list': train_acc_list,
                        'val_acc_list': val_acc_list,
                        'test_acc_list': test_acc_list,
                        'last_index': epoch_index +1}
            
            fid = open(log_file, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
            
    model.set_weights(optimal_weights)
    
    train_p = model.evaluate_generator(train_gen_fix, train_steps_fix)
    val_p = model.evaluate_generator(val_gen, val_steps)
    test_p = model.evaluate_generator(test_gen, test_steps)
    
    train_acc_list.append(train_p[metrics.index('acc')])
    val_acc_list.append(val_p[metrics.index('acc')])
    test_acc_list.append(test_p[metrics.index('acc')])
    
                    
    train_acc = train_acc_list[-1]
    val_acc = val_acc_list[-1]
    test_acc = test_acc_list[-1]
    train_categorical_crossentropy = train_p[metrics.index('loss')]
    val_categorical_crossentropy = val_p[metrics.index('loss')]
    test_categorical_crossentropy = test_p[metrics.index('loss')]
    
    if len(nb_neighbor) > 0:
        sensing_component = Models.get_sensing_model(input_shape, target_shape)
        sensing_component.compile('adam', 'mse', ['mse'])
        sensing_weights = model.get_layer('sensing').get_weights()[:3]
        sensing_component.get_layer('linear_encoder').set_weights(sensing_weights)
        
        x_train_encoded = sensing_component.predict(x_train, BATCH_SIZE)
        n_train = x_train.shape[0]
        x_train_encoded = np.reshape(x_train_encoded, (n_train, -1))
        
        x_val_encoded = sensing_component.predict(x_val, BATCH_SIZE)
        n_val = x_val.shape[0]
        x_val_encoded = np.reshape(x_val_encoded, (n_val, -1))
        
        x_test_encoded = sensing_component.predict(x_test, BATCH_SIZE)
        n_test = x_test.shape[0]
        x_test_encoded = np.reshape(x_test_encoded, (n_test, -1))
        
        knn_val_acc = []
        knn_test_acc = []
        
        for KN in nb_neighbor:
            classifier = KNeighborsClassifier(n_neighbors=KN, algorithm='brute')
            classifier.fit(x_train_encoded, np.argmax(y_train, axis=-1))
            
            knn_val_acc.append(classifier.score(x_val_encoded, np.argmax(y_val, axis=-1)))
            knn_test_acc.append(classifier.score(x_test_encoded, np.argmax(y_test, axis=-1)))
            
    else:
        knn_val_acc = []
        knn_test_acc = []
        
    weights = {}
    for layer in model.layers:
        weights[layer.name] = layer.get_weights()
        
    results = {'history': history,
               'train_acc_list': train_acc_list,
               'val_acc_list': val_acc_list,
               'test_acc_list': test_acc_list,
               'train_acc': train_acc,
               'val_acc': val_acc,
               'test_acc': test_acc,
               'knn_val_acc': knn_val_acc,
               'knn_test_acc': knn_test_acc,
               'train_categorical_crossentropy': train_categorical_crossentropy,
               'val_categorical_crossentropy': val_categorical_crossentropy,
               'test_categorical_crossentropy': test_categorical_crossentropy,
               'weights': weights}
    
    return results

def train_autoencoder(args):
    
    prefix, dataset, encode_shape, loss_type, complexity, use_unlabeled_data, regularization, LR, Epoch = args
    
    result_filename = '_'.join([str(v) for v in args]) + '.pickle'
    result_filename = os.path.join(OUTPUT_DIR, result_filename)
    
    if os.path.exists(result_filename):
        fid = open(result_filename, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result
    
    x_train, _, x_val, _, x_test, _ = Utility.load_data(dataset)
    if use_unlabeled_data:
        x_extra = np.load(os.path.join(Configuration.data_dir, dataset + '_x_no_label.npy'))
        x_train = np.concatenate((x_train, x_extra), axis=0)
        
    input_shape = x_train.shape[1:]
    
    conv_regularizer, conv_constraint, bl_regularizer, bl_constraint = regularization
    
    # get model
    model = Models.get_autoencoder(input_shape, encode_shape, complexity, conv_regularizer, conv_constraint, bl_regularizer, bl_constraint)
    model.compile('adam', loss_type, [loss_type,])
    
    train_gen, train_steps = Utility.get_autoencoder_data_generator(x_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_autoencoder_data_generator(x_val, BATCH_SIZE, shuffle=False, augmentation=False)
    
    metrics = model.metrics_names

    log_file = '_'.join([str(v) for v in args]) + '.pickle'
    log_file = os.path.join(LOG_DIR, log_file)

    # if exist log file, load log file, otherwise initialize from scratch    
    if not os.path.exists(log_file):
        current_weights = model.get_weights()
        optimal_weights = model.get_weights()
        train_loss = [model.evaluate_generator(train_gen, train_steps)[metrics.index('loss')]]
        val_loss = [model.evaluate_generator(val_gen, val_steps)[metrics.index('loss')]]
        optimal_performance = val_loss[0]
        last_index = 0
        
    else:
        fid = open(log_file, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        last_index = log_data['last_index']
        optimal_performance = log_data['optimal_performance']
        train_loss = log_data['train_loss']
        val_loss = log_data['val_loss']
        
    learning_rates = []
    
    for lr, sc in zip(LR, Epoch):
        learning_rates += [lr,]*sc
    
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam', loss_type, [loss_type, ])
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
            
        model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)
        
        if epoch_index % Configuration.VAL_MEASURE_INTERVAL == 0:
            train_loss_ = model.evaluate_generator(train_gen, train_steps)[metrics.index('loss')]
            val_loss_ = model.evaluate_generator(val_gen, val_steps)[metrics.index('loss')]
            train_loss.append(train_loss_)
            val_loss.append(val_loss_)
                
            if val_loss_ < optimal_performance:
                optimal_weights = model.get_weights()
                optimal_performance = val_loss_
                    

        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'optimal_performance': optimal_performance,
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                        'last_index': epoch_index +1}
            
            fid = open(log_file, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
            
    
    model.set_weights(optimal_weights)
        
    weights = {}
    for layer in model.layers:
        if layer.name.startswith('nonlinear_encoder') or layer.name.startswith('nonlinear_decoder'):
            weights[layer.name] = layer.get_weights()
    
    result = {'weights': weights,
              'val_loss': val_loss,
              'train_loss': train_loss,
              'optimal_performance': optimal_performance}
    
    return result

def train_prior(args):
    
    prefix, dataset, classifier, encode_shape, ae_loss, complexity, LR, Epoch = args
    regularization = Configuration.PRIOR_REGULARIZATION
    
    """ if result exists, load and return """
    result_filename = '_'.join([str(v) for v in args]) + '.pickle'
    result_filename = os.path.join(OUTPUT_DIR, result_filename)
    
    if os.path.exists(result_filename):
        fid = open(result_filename, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result
    
    """ load baseline classifier """
    baseline_filename = '_'.join([str(dataset), 
                                  str(classifier), 
                                  str(Configuration.BASELINE_REGULARIZATION), 
                                  str(Configuration.LR), 
                                  str(Configuration.EPOCH)]) + '.pickle'
    
    baseline_filename = os.path.join(BASELINE_DIR, baseline_filename)
    
    if not os.path.exists(baseline_filename):
        raise RuntimeError('Baseline model doesnt exist!')
        
    with open(baseline_filename, 'rb') as fid:
        baseline_classifier = pickle.load(fid)
        
    """ load autoencoder """
    autoencoder_params = ['autoencoder', dataset, encode_shape, ae_loss, complexity, 'False', regularization, LR, Epoch]
    autoencoder_filename = '_'.join([str(v) for v in autoencoder_params]) + '.pickle'
    autoencoder_filename = os.path.join(OUTPUT_DIR, autoencoder_filename)
    
    if not os.path.exists(autoencoder_filename):
        raise RuntimeError('Autoencoder model doesnt exist!')
        
    with open(autoencoder_filename, 'rb') as fid:
        autoencoder_model = pickle.load(fid)
        
    
    # load dataset, note data tiny imagenet is in uint8 from 0 to 255 and other datasets in float from 0 to 1
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    
    input_shape = x_train.shape[1:]
    n_class = y_train.shape[-1]
    
    conv_regularizer, conv_constraint, bl_regularizer, bl_constraint = regularization
    
    if classifier == 'allcnn':
        get_model = Models.get_prior_model_allcnn
    
    model = get_model(input_shape, n_class, encode_shape, complexity, conv_regularizer, conv_constraint, bl_regularizer, bl_constraint)
    model.compile('adam', 'categorical_crossentropy', ['acc',])
    
    for layer_name in baseline_classifier['weights'].keys():
        model.get_layer(layer_name).set_weights(baseline_classifier['weights'][layer_name])
    for layer_name in autoencoder_model['weights'].keys():
        model.get_layer(layer_name).set_weights(autoencoder_model['weights'][layer_name])
    
    
    train_gen, train_steps = Utility.get_sensing_data_generator(x_train, y_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_sensing_data_generator(x_val, y_val, BATCH_SIZE, shuffle=False, augmentation=False)
    test_gen, test_steps = Utility.get_sensing_data_generator(x_test, y_test, BATCH_SIZE, shuffle=False, augmentation=False)
    
    metrics = model.metrics_names

    
    log_file = '_'.join([str(v) for v in args]) + '.pickle'
    log_file = os.path.join(LOG_DIR, log_file)
    
    if not os.path.exists(log_file):
        
        current_weights = model.get_weights()
        optimal_weights = model.get_weights()
        history = {}
        train_p = model.evaluate(x_train, y_train, verbose=0)
        val_p = model.evaluate(x_val, y_val, verbose=0)
        test_p = model.evaluate(x_test, y_test, verbose=0)
        
        history['train_acc'] = [train_p[metrics.index('acc')]]
        history['val_acc'] = [val_p[metrics.index('acc')]]
        history['test_acc'] = [test_p[metrics.index('acc')]]
        
        val_measure = val_p[metrics.index('acc')]
        last_index = 0
        
    else:
        fid = open(log_file, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        last_index = log_data['last_index']
        val_measure = log_data['val_measure']
        history = log_data['history']
        
    learning_rates = []
    
    for lr, sc in zip(LR, Epoch):
        learning_rates += [lr,]*sc
    
    
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam', 'categorical_crossentropy', ['acc', ])
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
        
        print('epoch: %s' % str(epoch_index))
        model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)
        
        if epoch_index % Configuration.VAL_MEASURE_INTERVAL == 0:
            train_p = model.evaluate(x_train, y_train, verbose=0)
            val_p = model.evaluate(x_val, y_val, verbose=0)
            test_p = model.evaluate(x_test, y_test, verbose=0)
            
            history['train_acc'].append(train_p[metrics.index('acc')])
            history['val_acc'].append(val_p[metrics.index('acc')])
            history['test_acc'].append(test_p[metrics.index('acc')])
            
                
            if history['val_acc'][-1] > val_measure:
                optimal_weights = model.get_weights()
                val_measure = history['val_acc'][-1]

        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'val_measure': val_measure,
                        'history': history,
                        'last_index': epoch_index +1}
            
            fid = open(log_file, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
            
    
    model.set_weights(optimal_weights)
    train_p = model.evaluate(x_train, y_train, verbose=0)
    val_p = model.evaluate(x_val, y_val, verbose=0)
    test_p = model.evaluate(x_test, y_test, verbose=0)
        
    weights = {}
    for layer in model.layers:
        if layer.name.startswith('nonlinear_encoder') or layer.name.startswith('nonlinear_decoder') or layer.name.startswith(classifier):
            weights[layer.name] = layer.get_weights()
    
    result = {'weights': weights,
              'history': history,
              'train_acc': train_p[metrics.index('acc')],
              'val_acc': val_p[metrics.index('acc')],
              'test_acc': test_p[metrics.index('acc')]}
    
    return result

def distill_encoder(logfile,
                    dataset, 
                    classifier, 
                    encode_shape, 
                    loss_type, 
                    complexity,
                    conv_regularizer, 
                    conv_constraint, 
                    D_LR, D_Epoch,
                    prior_LR, prior_Epoch):
    
    """ load prior model """
    prior_params = ['prior', dataset, classifier, encode_shape, loss_type, complexity, prior_LR, prior_Epoch]
    prior_filename = '_'.join([str(v) for v in prior_params]) + '.pickle'
    prior_filename = os.path.join(OUTPUT_DIR, prior_filename)
    
    if not os.path.exists(prior_filename):
        raise RuntimeError('Prior model doesnt exist!')
        
    with open(prior_filename, 'rb') as fid:
        prior_model = pickle.load(fid)
        
    x_train, _, x_val, _, x_test, _ = Utility.load_data(dataset)
    
    input_shape = x_train.shape[1:]
        
    model = Models.get_distillation_encoder(input_shape, encode_shape, loss_type, complexity, conv_regularizer, conv_constraint)
    model.compile('adam')
    
    """ set weights for the fixed nonlinear encoding part """
    for layer in model.layers:
        if layer.name.startswith('fix_'):
            name = layer.name[4:]
            layer.set_weights(prior_model['weights'][name])
            layer.trainable = False
    
    train_gen, train_steps = Utility.get_distillation_data_generator(x_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_distillation_data_generator(x_val, BATCH_SIZE, shuffle=False, augmentation=False)
    test_gen, test_steps = Utility.get_distillation_data_generator(x_test, BATCH_SIZE, shuffle=False, augmentation=False)
    
    learning_rates = []
    for lr, sc in zip(D_LR, D_Epoch):
        learning_rates += [lr,]*sc
        
    if not os.path.exists(logfile):
        train_loss = [model.evaluate_generator(train_gen, train_steps)]
        val_loss = [model.evaluate_generator(val_gen, val_steps)]
        test_loss = [model.evaluate_generator(test_gen, test_steps)]
        val_measure = val_loss[-1]
        current_weights = model.get_weights()
        optimal_weights = model.get_weights()
        last_index = 0
    else:
        fid = open(logfile, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        train_loss = log_data['train_loss']
        val_loss = log_data['val_loss']
        test_loss = log_data['test_loss']
        val_measure = log_data['val_measure']
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        last_index = log_data['last_index']
    
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam')
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
            for layer in model.layers:
                if layer.name.startswith('fix_'):
                    layer.trainable = False
                
        h = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)
        val_loss.append(model.evaluate_generator(val_gen, val_steps))
        test_loss.append(model.evaluate_generator(test_gen, test_steps))
        train_loss.append(h.history['loss'])
                
        if val_loss[-1] < val_measure:
            optimal_weights = model.get_weights()
            val_measure = val_loss[-1]
                    
        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'val_measure': val_measure,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                        'last_index': epoch_index +1}
            
            fid = open(logfile, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
    
    model.set_weights(optimal_weights)
    
    linear_encoder_weights = model.get_layer('linear_encoder').get_weights()
    return linear_encoder_weights, train_loss, val_loss, test_loss

def distill_encoder_decoder(logfile,
                            dataset, 
                            classifier, 
                            encode_shape, 
                            linear_encoder_weights,
                            loss_type, 
                            complexity,
                            conv_regularizer, 
                            conv_constraint, 
                            bl_regularizer,
                            bl_constraint,
                            D_LR, D_Epoch,
                            prior_LR, prior_Epoch):
    
    """ load nonlinear sensing model """
    prior_params = ['prior', dataset, classifier, encode_shape, loss_type, complexity, prior_LR, prior_Epoch]
    prior_filename = '_'.join([str(v) for v in prior_params]) + '.pickle'
    prior_filename = os.path.join(OUTPUT_DIR, prior_filename)
    
    if not os.path.exists(prior_filename):
        raise RuntimeError('prior model doesnt exist!')
        
    with open(prior_filename, 'rb') as fid:
        prior_model = pickle.load(fid)
        
    x_train, _, x_val, _, x_test, _ = Utility.load_data(dataset)
    
    input_shape = x_train.shape[1:]
        
    model = Models.get_distillation_encoder_decoder(input_shape, 
                                                    encode_shape, 
                                                    loss_type, 
                                                    complexity,
                                                    conv_regularizer, 
                                                    conv_constraint,
                                                    bl_regularizer,
                                                    bl_constraint)
    model.compile('adam')
    
    """ set weights for the fixed nonlinear encoding + decoding part """
    for layer in model.layers:
        if layer.name.startswith('fix_'):
            name = layer.name[4:]
            layer.set_weights(prior_model['weights'][name])
            layer.trainable = False

    """ set weights for the trainable linear encoder part """ 
    if linear_encoder_weights is not None:
        model.get_layer('linear_encoder').set_weights(linear_encoder_weights)
        
    """ set weights for the trainable nonlinear decoder part """
    for layer in model.layers:
        if layer.name.startswith('nonlinear_decoder'):
            layer.set_weights(prior_model['weights'][layer.name])
    
    train_gen, train_steps = Utility.get_distillation_data_generator(x_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_distillation_data_generator(x_val, BATCH_SIZE, shuffle=False, augmentation=False)
    test_gen, test_steps = Utility.get_distillation_data_generator(x_test, BATCH_SIZE, shuffle=False, augmentation=False)
    
    learning_rates = []
    for lr, sc in zip(D_LR, D_Epoch):
        learning_rates += [lr,]*sc
    
    if not os.path.exists(logfile):
        train_loss = [model.evaluate_generator(train_gen, train_steps)]
        val_loss = [model.evaluate_generator(val_gen, val_steps)]
        test_loss = [model.evaluate_generator(test_gen, test_steps)]
        val_measure = val_loss[-1]
        current_weights = model.get_weights()
        optimal_weights = model.get_weights()
        last_index = 0
    else:
        fid = open(logfile, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        train_loss = log_data['train_loss']
        val_loss = log_data['val_loss']
        test_loss = log_data['test_loss']
        val_measure = log_data['val_measure']
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        last_index = log_data['last_index']
    
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam')
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
            for layer in model.layers:
                if layer.name.startswith('fix_'):
                    layer.trainable = False
                
        h = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)
        val_loss.append(model.evaluate_generator(val_gen, val_steps))
        test_loss.append(model.evaluate_generator(test_gen, test_steps))
        train_loss.append(h.history['loss'])
                
        if val_loss[-1] < val_measure:
            optimal_weights = model.get_weights()
            val_measure = val_loss[-1]
                    
        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'val_measure': val_measure,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                        'last_index': epoch_index +1}
            
            fid = open(logfile, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
    
    model.set_weights(optimal_weights)
    
    linear_encoder_weights = model.get_layer('linear_encoder').get_weights()
    nonlinear_decoder_weights = {}
    for layer in model.layers:
        if layer.name.startswith('nonlinear_decoder'):
            nonlinear_decoder_weights[layer.name] = layer.get_weights()
    
    return linear_encoder_weights, nonlinear_decoder_weights, train_loss, val_loss, test_loss

def train_MCLwP(args):
    
    prefix = args[0]
    dataset = args[1]
    classifier = args[2]
    encode_shape = args[3]
    regularization = args[4]
    distillation_loss = args[5]
    distillation_coef = args[6]
    D_LR = args[7]
    D_Epoch = args[8]
    prior_LR = args[9]
    prior_Epoch = args[10]
    LR = args[11]
    Epoch = args[12]
    exp = args[13]
    distil_encoder = args[14]
    distil_encoder_decoder = args[15]
    distil_prediction = args[16]
    complexity = args[17]
    nb_neighbor = args[18]
    
    conv_regularizer, conv_constraint, bl_regularizer, bl_constraint = regularization
    
    """ if result exists, load and return """
    result_filename = '_'.join([str(v) for v in args]) + '.pickle'
    result_filename = os.path.join(OUTPUT_DIR, result_filename)
    
    if os.path.exists(result_filename):
        fid = open(result_filename, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result
    
    """ distill encoder """
    
    if distil_encoder:
        distillation_encoder_logfile = 'disen_' + '_'.join([str(v) for v in args[1:]]) + '.pickle'
        distillation_encoder_logfile = os.path.join(LOG_DIR, distillation_encoder_logfile)
        distillation_e_outputs = distill_encoder(distillation_encoder_logfile,
                                                 dataset, 
                                                 classifier, 
                                                 encode_shape, 
                                                 distillation_loss, 
                                                 complexity,
                                                 conv_regularizer, 
                                                 conv_constraint, 
                                                 D_LR, D_Epoch,
                                                 prior_LR, prior_Epoch)
        
    else:
        distillation_e_outputs = None
    
    
    """ train distillation to get linear encoder + nonlinear decoder weights """
    if distil_encoder_decoder:
        print('encoder decoder distillation')
        distillation_ed_logfile = 'disde_' + '_'.join([str(v) for v in args[1:]]) + '.pickle'
        distillation_ed_logfile = os.path.join(LOG_DIR, distillation_ed_logfile)
        
        distillation_ed_outputs = distill_encoder_decoder(distillation_ed_logfile,
                                                          dataset, 
                                                          classifier, 
                                                          encode_shape, 
                                                          distillation_e_outputs[0] if distillation_e_outputs is not None else None,
                                                          distillation_loss, 
                                                          complexity,
                                                          conv_regularizer, 
                                                          conv_constraint, 
                                                          bl_regularizer,
                                                          bl_constraint,
                                                          D_LR, D_Epoch,
                                                          prior_LR, prior_Epoch)
        
    else:
        distillation_ed_outputs = None
    
    
    
    """ load prior model """
    prior_params = ['prior', dataset, classifier, encode_shape, distillation_loss, complexity, prior_LR, prior_Epoch]
    prior_filename = '_'.join([str(v) for v in prior_params]) + '.pickle'
    prior_filename = os.path.join(OUTPUT_DIR, prior_filename)
    
    if not os.path.exists(prior_filename):
        raise RuntimeError('prior model doesnt exist!')
        
    with open(prior_filename, 'rb') as fid:
        prior_model = pickle.load(fid)
    
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    
    input_shape = x_train.shape[1:]
    n_class = y_train.shape[-1]
    
    
    
    get_model = Models.get_MCLwP_allcnn
    
    model = get_model(input_shape,
                     n_class, 
                     encode_shape, 
                     distil_prediction,
                     distillation_coef,
                     complexity,
                     conv_regularizer, 
                     conv_constraint,
                     bl_regularizer,
                     bl_constraint)
    
    model.compile('adam', 
                  {classifier + '_prediction': 'categorical_crossentropy', 'fix_' + classifier + '_prediction': None},
                  {classifier + '_prediction': ['acc']})
    
    """ set weights """
    # set weights for the classifier part from nonlinear sensing model
    for layer in model.layers:
        if layer.name.startswith(classifier):
            layer.set_weights(prior_model['weights'][layer.name])

    # set weights for the trainable linear encoder part
    if distillation_e_outputs is not None:
        if distillation_ed_outputs is not None:
            model.get_layer('linear_encoder').set_weights(distillation_ed_outputs[0])
        
            # set weights for the trainable nonlinear decoder part
            for layer_name in distillation_ed_outputs[1].keys():
                if layer_name.startswith('nonlinear_decoder'):
                    model.get_layer(layer_name).set_weights(distillation_ed_outputs[1][layer_name])
        else:
            model.get_layer('linear_encoder').set_weights(distillation_e_outputs[0])
    else:
        if distillation_ed_outputs is not None:
            model.get_layer('linear_encoder').set_weights(distillation_ed_outputs[0])
        
            # set weights for the trainable nonlinear decoder part
            for layer_name in distillation_ed_outputs[1].keys():
                if layer_name.startswith('nonlinear_decoder'):
                    model.get_layer(layer_name).set_weights(distillation_ed_outputs[1][layer_name])
   
    
    # set the weights for fix branch of nonlinear sensing
    for layer_name in prior_model['weights'].keys():
        model.get_layer('fix_' + layer_name).set_weights(prior_model['weights'][layer_name])
        model.get_layer('fix_' + layer_name).trainable = False
    
    
    train_gen, train_steps = Utility.get_sensing_data_generator(x_train, y_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_sensing_data_generator(x_val, y_val, BATCH_SIZE, shuffle=False, augmentation=False)
    test_gen, test_steps = Utility.get_sensing_data_generator(x_test, y_test, BATCH_SIZE, shuffle=False, augmentation=False)
    
    metrics = model.metrics_names

    
    log_file = '_'.join([str(v) for v in args]) + '.pickle'
    log_file = os.path.join(LOG_DIR, log_file)
    
    acc_key = classifier + '_prediction_acc'
    
    if not os.path.exists(log_file):
        
        current_weights = model.get_weights()
        optimal_weights = model.get_weights()
        history = {}
        
        train_p = model.evaluate(x_train, y_train, BATCH_SIZE, verbose=0)
        val_p = model.evaluate(x_val, y_val, BATCH_SIZE, verbose=0)
        test_p = model.evaluate(x_test, y_test, BATCH_SIZE, verbose=0)
        
        history['train_acc'] = [train_p[metrics.index(acc_key)]]
        history['val_acc'] = [val_p[metrics.index(acc_key)]]
        history['test_acc'] = [test_p[metrics.index(acc_key)]]
        
        val_measure = history['val_acc'][-1]
        last_index = 0
        
    else:
        fid = open(log_file, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        last_index = log_data['last_index']
        val_measure = log_data['val_measure']
        history = log_data['history']
        
    learning_rates = []
    
    for lr, sc in zip(LR, Epoch):
        learning_rates += [lr,]*sc
    
    
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam',
                  {classifier + '_prediction': 'categorical_crossentropy', 'fix_' + classifier + '_prediction': None},
                  {classifier + '_prediction': ['acc']})

            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
            
            for layer in model.layers:
                if layer.name.startswith('fix_'):
                    layer.trainable = False
        
        print('epoch: %s' % str(epoch_index))
        model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)
        
        if epoch_index % Configuration.VAL_MEASURE_INTERVAL == 0:
            
            train_p = model.evaluate(x_train, y_train, BATCH_SIZE, verbose=0)
            val_p = model.evaluate(x_val, y_val, BATCH_SIZE, verbose=0)
            test_p = model.evaluate(x_test, y_test, BATCH_SIZE, verbose=0)
            
            history['train_acc'].append(train_p[metrics.index(acc_key)])
            history['val_acc'].append(val_p[metrics.index(acc_key)])
            history['test_acc'].append(test_p[metrics.index(acc_key)])
            
                
            if history['val_acc'][-1] > val_measure:
                optimal_weights = model.get_weights()
                val_measure = history['val_acc'][-1]

        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'val_measure': val_measure,
                        'history': history,
                        'last_index': epoch_index +1}
            
            fid = open(log_file, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
            
    
    model.set_weights(optimal_weights)
    train_p = model.evaluate(x_train, y_train, BATCH_SIZE, verbose=0)
    val_p = model.evaluate(x_val, y_val, BATCH_SIZE, verbose=0)
    test_p = model.evaluate(x_test, y_test, BATCH_SIZE, verbose=0)
    
    # compute K Nearest Neighbor
    if len(nb_neighbor) > 0:
        sensing_component = Models.Model(inputs=model.input, outputs=model.get_layer('linear_encoder').output)
        
        x_train_encoded = sensing_component.predict(x_train, BATCH_SIZE)
        n_train = x_train.shape[0]
        x_train_encoded = np.reshape(x_train_encoded, (n_train, -1))
        
        x_val_encoded = sensing_component.predict(x_val, BATCH_SIZE)
        n_val = x_val.shape[0]
        x_val_encoded = np.reshape(x_val_encoded, (n_val, -1))
        
        x_test_encoded = sensing_component.predict(x_test, BATCH_SIZE)
        n_test = x_test.shape[0]
        x_test_encoded = np.reshape(x_test_encoded, (n_test, -1))
        
        knn_val_acc = []
        knn_test_acc = []
        
        for KN in nb_neighbor:
            classifier = KNeighborsClassifier(n_neighbors=KN, algorithm='brute')
            classifier.fit(x_train_encoded, np.argmax(y_train, axis=-1))
            
            knn_val_acc.append(classifier.score(x_val_encoded, np.argmax(y_val, axis=-1)))
            knn_test_acc.append(classifier.score(x_test_encoded, np.argmax(y_test, axis=-1)))
    else:
        knn_val_acc = []
        knn_test_acc = []
        
    weights = {}
    for layer in model.layers:
        weights[layer.name] = layer.get_weights()
    
    result = {'weights': weights,
              'history': history,
              'distillation_encoder': distillation_e_outputs,
              'distillation_encoder_decoder': distillation_ed_outputs,
              'train_acc': train_p[metrics.index(acc_key)],
              'val_acc': val_p[metrics.index(acc_key)],
              'test_acc': test_p[metrics.index(acc_key)],
              'knn_val_acc': knn_val_acc,
              'knn_test_acc': knn_test_acc}
    
    return result

def train_MCLwoP(args):
    prefix = args[0]
    dataset = args[1]
    classifier = args[2]
    target_shape = args[3]
    height, width, depth = target_shape
    regularization = args[4]
    projection_decay, projection_constraint, weight_decay, weight_constraint = regularization
    exp = args[5]
    LR = args[6]
    Epoch = args[7]
    
    # if exists result, load and return
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Configuration.output_dir, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        outputs = pickle.load(fid)
        fid.close()
        return outputs
    
    # load baseline model
    baseline_filename = '_'.join([str(dataset), 
                                  str(classifier), 
                                  str(Configuration.BASELINE_REGULARIZATION), 
                                  str(Configuration.LR), 
                                  str(Configuration.EPOCH)]) + '.pickle'
    
    baseline_filename = os.path.join(BASELINE_DIR, baseline_filename)
    
    if not os.path.exists(baseline_filename):
        raise RuntimeError('Baseline model doesnt exist!')
    
    with open(baseline_filename, 'rb') as fid:
        baseline_model = pickle.load(fid)
    
    baseline_weights = baseline_model['weights']
    
    # load dataset
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    input_shape = x_train.shape[1:]
    n_class = y_train.shape[-1]
    
    # initialize the encoder + decoder part first
    train_gen, train_steps = Utility.get_autoencoder_data_generator(x_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_autoencoder_data_generator(x_val, BATCH_SIZE, shuffle=False, augmentation=False)

    hybrid_ae = Models.get_hybrid_autoencoder(input_shape, 
                                              target_shape, 
                                              weight_decay, 
                                              weight_constraint, 
                                              projection_decay, 
                                              projection_constraint)
    
    hybrid_ae.compile('adam', 'mae', ['mae'])
    
    ae_metrics = hybrid_ae.metrics_names
    
    current_ae_weights = hybrid_ae.get_weights()
    
    train_ae_loss = [hybrid_ae.evaluate_generator(train_gen, train_steps)[ae_metrics.index('loss')]]
    val_ae_loss = [hybrid_ae.evaluate_generator(val_gen, val_steps)[ae_metrics.index('loss')]]
    
    optimal_ae_weights = hybrid_ae.get_weights()
    optimal_ae_loss = val_ae_loss[-1]
    

    for lr, epoch in zip(LR, Epoch):
        hybrid_ae.compile('adam', 'mae', ['mae'])
        hybrid_ae.set_weights(current_ae_weights)
        hybrid_ae.optimizer.lr = lr
        
        hybrid_ae.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epoch, verbose=0)
        current_ae_weights = hybrid_ae.get_weights()
        
        train_ae_loss.append(hybrid_ae.evaluate_generator(train_gen, train_steps)[ae_metrics.index('loss')])
        val_ae_loss.append(hybrid_ae.evaluate_generator(val_gen, val_steps)[ae_metrics.index('loss')])
        
        if val_ae_loss[-1] < optimal_ae_loss:
            optimal_ae_loss = val_ae_loss[-1]
            optimal_ae_weights = hybrid_ae.get_weights()
            
    hybrid_ae.set_weights(optimal_ae_weights)
    hybrid_ae_weights = {}
    for layer in hybrid_ae.layers:
        hybrid_ae_weights[layer.name] = layer.get_weights()
    
    Models.K.clear_session()
    
    # train MCLwoP model
    
    train_gen, train_steps = Utility.get_baseline_data_generator(x_train,
                                                        y_train,
                                                        shuffle=True,
                                                        augmentation=True,
                                                        batch_size=BATCH_SIZE)
    train_gen_fix, train_steps_fix = Utility.get_baseline_data_generator(x_train,
                                                                y_train,
                                                                shuffle=False,
                                                                augmentation=False,
                                                                batch_size=BATCH_SIZE)
    val_gen, val_steps = Utility.get_baseline_data_generator(x_val,
                                                    y_val,
                                                    shuffle=False,
                                                    augmentation=False,
                                                    batch_size=BATCH_SIZE)
    
    test_gen, test_steps = Utility.get_baseline_data_generator(x_test,
                                                               y_test,
                                                               shuffle=False,
                                                               augmentation=False,
                                                               batch_size=BATCH_SIZE)
    
    

    
    
    get_model = Models.get_MCLwoP_allcnn
        
    model = get_model(input_shape=input_shape, 
                      n_class=n_class, 
                      encode_shape=target_shape,
                      conv_regularizer=weight_decay, 
                      conv_constraint=weight_constraint, 
                      bl_regularizer=projection_decay, 
                      bl_constraint=projection_constraint)

    model.compile('adam', 'categorical_crossentropy', ['acc'])
    metrics = model.metrics_names
    
    
    log_file = '_'.join([str(v) for v in args]) + '.pickle'
    log_file = os.path.join(LOG_DIR, log_file)
    
    # if not exist log file, start from beginning, otherwise load the log file
    if not os.path.exists(log_file):    
        # set the pretrained weights
        for layer_name in baseline_weights.keys():
            model.get_layer(layer_name).set_weights(baseline_weights[layer_name])
        
        # set the sensing and feature synthesis component
        for layer in model.layers:
            if layer.name in hybrid_ae_weights.keys():
                layer.set_weights(hybrid_ae_weights[layer.name])
        
        current_weights = model.get_weights()        
        optimal_weights = model.get_weights() 
        train_acc_list = []
        val_acc_list = []
        test_acc_list = []
        
        train_p = model.evaluate_generator(train_gen_fix, train_steps_fix)
        val_p = model.evaluate_generator(val_gen, val_steps)
        test_p = model.evaluate_generator(test_gen, test_steps)
        
        train_acc_list.append(train_p[metrics.index('acc')])
        val_acc_list.append(val_p[metrics.index('acc')])
        test_acc_list.append(test_p[metrics.index('acc')])
        
        validation_measure = val_acc_list[0]
        
        history = {'train_acc':[], 'train_categorical_crossentropy':[], 'val_acc':[], 'val_categorical_crossentropy':[]}
        last_index = 0
    else:
        fid = open(log_file, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        validation_measure = log_data['validation_measure']
        train_acc_list = log_data['train_acc_list']
        val_acc_list = log_data['val_acc_list']
        test_acc_list = log_data['test_acc_list']
        history = log_data['history']
        last_index = log_data['last_index']
        
    learning_rates = []
    for lr, sc in zip(LR, Epoch):
        learning_rates += [lr,]*sc
        
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam', 'categorical_crossentropy', ['acc', ])
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
            
        h = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)

        val_p = model.evaluate_generator(val_gen, val_steps)
        
        if val_p[metrics.index('acc')] > validation_measure:
            validation_measure = val_p[metrics.index('acc')]
            optimal_weights = model.get_weights()
            
        history['train_acc'] += h.history['acc']
        history['train_categorical_crossentropy'] += h.history['loss']
        history['val_acc'].append(val_p[metrics.index('acc')])
        history['val_categorical_crossentropy'].append(val_p[metrics.index('loss')])
        
        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'validation_measure': validation_measure,
                        'history': history,
                        'train_acc_list': train_acc_list,
                        'val_acc_list': val_acc_list,
                        'test_acc_list': test_acc_list,
                        'last_index': epoch_index +1}
            
            fid = open(log_file, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
            
    model.set_weights(optimal_weights)
    
    train_p = model.evaluate_generator(train_gen_fix, train_steps_fix)
    val_p = model.evaluate_generator(val_gen, val_steps)
    test_p = model.evaluate_generator(test_gen, test_steps)
    
    train_acc_list.append(train_p[metrics.index('acc')])
    val_acc_list.append(val_p[metrics.index('acc')])
    test_acc_list.append(test_p[metrics.index('acc')])
    
                    
    train_acc = train_acc_list[-1]
    val_acc = val_acc_list[-1]
    test_acc = test_acc_list[-1]
    train_categorical_crossentropy = train_p[metrics.index('loss')]
    val_categorical_crossentropy = val_p[metrics.index('loss')]
    test_categorical_crossentropy = test_p[metrics.index('loss')]
        
    weights = {}
    for layer in model.layers:
        weights[layer.name] = layer.get_weights()
        
    results = {'history': history,
               'train_acc_list': train_acc_list,
               'val_acc_list': val_acc_list,
               'test_acc_list': test_acc_list,
               'train_acc': train_acc,
               'val_acc': val_acc,
               'test_acc': test_acc,
               'ae_train_loss': train_ae_loss,
               'ae_val_loss': val_ae_loss,
               'train_categorical_crossentropy': train_categorical_crossentropy,
               'val_categorical_crossentropy': val_categorical_crossentropy,
               'test_categorical_crossentropy': test_categorical_crossentropy,
               'weights': weights}
    
    return results


def train_prior_semi_supervised_inner_loop(model, 
                                           dataset, 
                                           classifier, 
                                           x_augment, 
                                           y_augment, 
                                           LR, 
                                           Epoch, 
                                           log_file):
    
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    
    if x_augment is not None and y_augment is not None:
        x_train = np.concatenate((x_train, x_augment), axis=0)
        y_train = np.concatenate((y_train, y_augment), axis=0)
    
    train_gen, train_steps = Utility.get_sensing_data_generator(x_train, y_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_sensing_data_generator(x_val, y_val, BATCH_SIZE, shuffle=False, augmentation=False)
    test_gen, test_steps = Utility.get_sensing_data_generator(x_test, y_test, BATCH_SIZE, shuffle=False, augmentation=False)
    
    metrics = model.metrics_names
    
    if not os.path.exists(log_file):
        
        current_weights = model.get_weights()
        optimal_weights = model.get_weights()
        history = {}
        train_p = model.evaluate(x_train, y_train, verbose=0)
        val_p = model.evaluate(x_val, y_val, verbose=0)
        test_p = model.evaluate(x_test, y_test, verbose=0)
        
        history['train_acc'] = [train_p[metrics.index('acc')]]
        history['val_acc'] = [val_p[metrics.index('acc')]]
        history['test_acc'] = [test_p[metrics.index('acc')]]
        
        val_measure = val_p[metrics.index('acc')]
        last_index = 0
        
    else:
        fid = open(log_file, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        last_index = log_data['last_index']
        val_measure = log_data['val_measure']
        history = log_data['history']
        
    learning_rates = []
    
    for lr, sc in zip(LR, Epoch):
        learning_rates += [lr,]*sc
    
    
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam', 'categorical_crossentropy', ['acc', ])
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
        
        model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)
        
        if epoch_index % Configuration.VAL_MEASURE_INTERVAL == 0:
            train_p = model.evaluate(x_train, y_train, verbose=0)
            val_p = model.evaluate(x_val, y_val, verbose=0)
            test_p = model.evaluate(x_test, y_test, verbose=0)
            
            history['train_acc'].append(train_p[metrics.index('acc')])
            history['val_acc'].append(val_p[metrics.index('acc')])
            history['test_acc'].append(test_p[metrics.index('acc')])
            
                
            if history['val_acc'][-1] > val_measure:
                optimal_weights = model.get_weights()
                val_measure = history['val_acc'][-1]

        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'val_measure': val_measure,
                        'history': history,
                        'last_index': epoch_index +1}
            
            fid = open(log_file, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
        
    
    model.set_weights(optimal_weights)
    train_p = model.evaluate(x_train, y_train)
    val_p = model.evaluate(x_val, y_val)
    test_p = model.evaluate(x_test, y_test)
        
    weights = {}
    for layer in model.layers:
        if layer.name.startswith('nonlinear_encoder') or layer.name.startswith('nonlinear_decoder') or layer.name.startswith(classifier):
            weights[layer.name] = layer.get_weights()
    
    result = {'weights': weights,
              'history': history,
              'train_acc': train_p[metrics.index('acc')],
              'val_acc': val_p[metrics.index('acc')],
              'test_acc': test_p[metrics.index('acc')]}
    
    return result, model

def get_pseudo_label(model, x, min_confidence_score):
    prob = model.predict(x)
    y_pred = np.argmax(prob, axis=-1)
    y_prob = np.max(prob, axis=-1)
    indices = np.where(y_prob >= min_confidence_score)[0]
    
    if indices.size == 0:
        return None, None
    
    x_augment = x[indices]
    y_augment = Utility.to_categorical(y_pred[indices], prob.shape[-1])
    
    return x_augment, y_augment


def train_prior_semi_supervised(args):
    prefix, dataset, classifier, encode_shape, ae_loss, LR, Epoch, confidence_score = args
    regularization = Configuration.PRIOR_REGULARIZATION
    
    """ if result exists, load and return """
    result_filename = '_'.join([str(v) for v in args]) + '.pickle'
    result_filename = os.path.join(OUTPUT_DIR, result_filename)
    
    if os.path.exists(result_filename):
        fid = open(result_filename, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result
    
    """ load baseline classifier """
    baseline_filename = '_'.join([str(dataset), str(classifier), str(Configuration.BASELINE_REGULARIZATION), str(Configuration.LR), str(Configuration.EPOCH)]) + '.pickle'
    baseline_filename = os.path.join(BASELINE_DIR, baseline_filename)
    
    if not os.path.exists(baseline_filename):
        raise RuntimeError('Baseline model doesnt exist!')
        
    with open(baseline_filename, 'rb') as fid:
        baseline_classifier = pickle.load(fid)
        
    """ load autoencoder """
    autoencoder_params = ['autoencoder', dataset, encode_shape, ae_loss, 'low', 'True', regularization, LR, Epoch]
    autoencoder_filename = '_'.join([str(v) for v in autoencoder_params]) + '.pickle'
    autoencoder_filename = os.path.join(OUTPUT_DIR, autoencoder_filename)
    
    if not os.path.exists(autoencoder_filename):
        raise RuntimeError('Autoencoder model doesnt exist!')
        
    with open(autoencoder_filename, 'rb') as fid:
        autoencoder_model = pickle.load(fid)
        
    # load dataset
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    
    x_extra = np.load(os.path.join(Configuration.data_dir, dataset + '_x_no_label.npy'))
    
    input_shape = x_train.shape[1:]
    n_class = y_train.shape[-1]
    
    conv_regularizer, conv_constraint, bl_regularizer, bl_constraint = regularization
    
    get_model = Models.get_prior_model_allcnn
    
    model = get_model(input_shape, n_class, encode_shape, 'low', conv_regularizer, conv_constraint, bl_regularizer, bl_constraint)
    model.compile('adam', 'categorical_crossentropy', ['acc',])
    
    
    for layer_name in baseline_classifier['weights'].keys():
        model.get_layer(layer_name).set_weights(baseline_classifier['weights'][layer_name])
    for layer_name in autoencoder_model['weights'].keys():
        model.get_layer(layer_name).set_weights(autoencoder_model['weights'][layer_name])
    
    self_learning_iter = 0
    
    x_augment = None
    y_augment = None
    
    results = []
    
    while self_learning_iter < 20:
        log_file = '_'.join([str(v) for v in args]) + '_' + str(self_learning_iter) + '.pickle'
        log_file = os.path.join(LOG_DIR, log_file)
        
        current_result, model = train_prior_semi_supervised_inner_loop(model, 
                                                                       dataset, 
                                                                       classifier, 
                                                                       x_augment, 
                                                                       y_augment, 
                                                                       LR, 
                                                                       Epoch, 
                                                                       log_file)
        
        results.append(current_result)
        
        x_augment_new, y_augment_new = get_pseudo_label(model, x_extra, confidence_score) 
        
        if x_augment_new is None:
            break
        
        if x_augment is not None and x_augment.shape[0] == x_augment_new.shape[0] and np.allclose(x_augment, x_augment_new):
            break
    
        x_augment = x_augment_new
        y_augment = y_augment_new
        
        self_learning_iter += 1
        
    outputs = {'weights': current_result['weights'],
               'history': current_result['history'],
               'train_acc': current_result['train_acc'],
               'val_acc': current_result['val_acc'],
               'test_acc': current_result['test_acc'],
               'result_list': results}
    
    return outputs


def distill_encoder_semi_supervised(logfile,
                                    dataset, 
                                    classifier, 
                                    encode_shape, 
                                    loss_type, 
                                    confidence_score,
                                    conv_regularizer, 
                                    conv_constraint, 
                                    D_LR, D_Epoch,
                                    prior_LR, prior_Epoch):
    
    """ load prior model """
    prior_params = ['priorS', dataset, classifier, encode_shape, loss_type, prior_LR, prior_Epoch, confidence_score]
    prior_filename = '_'.join([str(v) for v in prior_params]) + '.pickle'
    prior_filename = os.path.join(OUTPUT_DIR, prior_filename)
    
    if not os.path.exists(prior_filename):
        raise RuntimeError('Prior model doesnt exist!')
        
    with open(prior_filename, 'rb') as fid:
        prior_model = pickle.load(fid)
        
    x_train, _, x_val, _, x_test, _ = Utility.load_data(dataset)
    x_extra = np.load(os.path.join(Configuration.data_dir, dataset + '_x_no_label.npy'))
    x_train = np.concatenate((x_train, x_extra), axis=0)
    
    
    input_shape = x_train.shape[1:]
        
    model = Models.get_distillation_encoder(input_shape, encode_shape, loss_type, 'low', conv_regularizer, conv_constraint)
    model.compile('adam')
    
    """ set weights for the fixed nonlinear encoding part """
    for layer in model.layers:
        if layer.name.startswith('fix_'):
            name = layer.name[4:]
            layer.set_weights(prior_model['weights'][name])
            layer.trainable = False
    
    train_gen, train_steps = Utility.get_distillation_data_generator(x_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_distillation_data_generator(x_val, BATCH_SIZE, shuffle=False, augmentation=False)
    test_gen, test_steps = Utility.get_distillation_data_generator(x_test, BATCH_SIZE, shuffle=False, augmentation=False)
    
    learning_rates = []
    for lr, sc in zip(D_LR, D_Epoch):
        learning_rates += [lr,]*sc
        
    if not os.path.exists(logfile):
        train_loss = [model.evaluate_generator(train_gen, train_steps)]
        val_loss = [model.evaluate_generator(val_gen, val_steps)]
        test_loss = [model.evaluate_generator(test_gen, test_steps)]
        val_measure = val_loss[-1]
        current_weights = model.get_weights()
        optimal_weights = model.get_weights()
        last_index = 0
    else:
        fid = open(logfile, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        train_loss = log_data['train_loss']
        val_loss = log_data['val_loss']
        test_loss = log_data['test_loss']
        val_measure = log_data['val_measure']
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        last_index = log_data['last_index']
    
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam')
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
            for layer in model.layers:
                if layer.name.startswith('fix_'):
                    layer.trainable = False
                
        h = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)
        val_loss.append(model.evaluate_generator(val_gen, val_steps))
        test_loss.append(model.evaluate_generator(test_gen, test_steps))
        train_loss.append(h.history['loss'])
                
        if val_loss[-1] < val_measure:
            optimal_weights = model.get_weights()
            val_measure = val_loss[-1]
                    
        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'val_measure': val_measure,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                        'last_index': epoch_index +1}
            
            fid = open(logfile, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
    
    model.set_weights(optimal_weights)
    
    linear_encoder_weights = model.get_layer('linear_encoder').get_weights()
    return linear_encoder_weights, train_loss, val_loss, test_loss

def distill_encoder_decoder_semi_supervised(logfile,
                                            dataset, 
                                            classifier, 
                                            encode_shape, 
                                            linear_encoder_weights,
                                            loss_type, 
                                            confidence_score,
                                            conv_regularizer, 
                                            conv_constraint, 
                                            bl_regularizer,
                                            bl_constraint,
                                            D_LR, D_Epoch,
                                            prior_LR, prior_Epoch):
    
    """ load prior model """
    prior_params = ['priorS', dataset, classifier, encode_shape, loss_type, prior_LR, prior_Epoch, confidence_score]
    prior_filename = '_'.join([str(v) for v in prior_params]) + '.pickle'
    prior_filename = os.path.join(OUTPUT_DIR, prior_filename)
    
    if not os.path.exists(prior_filename):
        raise RuntimeError('prior model doesnt exist!')
        
    with open(prior_filename, 'rb') as fid:
        prior_model = pickle.load(fid)
        
    x_train, _, x_val, _, x_test, _ = Utility.load_data(dataset)
    x_extra = np.load(os.path.join(Configuration.data_dir, dataset + '_x_no_label.npy'))
    x_train = np.concatenate((x_train, x_extra), axis=0)
    
    input_shape = x_train.shape[1:]
        
    model = Models.get_distillation_encoder_decoder(input_shape, 
                                                    encode_shape, 
                                                    loss_type, 
                                                    'low',
                                                    conv_regularizer, 
                                                    conv_constraint,
                                                    bl_regularizer,
                                                    bl_constraint)
    model.compile('adam')
    
    """ set weights for the fixed nonlinear encoding + decoding part """
    for layer in model.layers:
        if layer.name.startswith('fix_'):
            name = layer.name[4:]
            layer.set_weights(prior_model['weights'][name])
            layer.trainable = False

    """ set weights for the trainable linear encoder part """ 
    if linear_encoder_weights is not None:
        model.get_layer('linear_encoder').set_weights(linear_encoder_weights)
        
    """ set weights for the trainable nonlinear decoder part """
    for layer in model.layers:
        if layer.name.startswith('nonlinear_decoder'):
            layer.set_weights(prior_model['weights'][layer.name])
    
    train_gen, train_steps = Utility.get_distillation_data_generator(x_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_distillation_data_generator(x_val, BATCH_SIZE, shuffle=False, augmentation=False)
    test_gen, test_steps = Utility.get_distillation_data_generator(x_test, BATCH_SIZE, shuffle=False, augmentation=False)
    
    learning_rates = []
    for lr, sc in zip(D_LR, D_Epoch):
        learning_rates += [lr,]*sc
    
    if not os.path.exists(logfile):
        train_loss = [model.evaluate_generator(train_gen, train_steps)]
        val_loss = [model.evaluate_generator(val_gen, val_steps)]
        test_loss = [model.evaluate_generator(test_gen, test_steps)]
        val_measure = val_loss[-1]
        current_weights = model.get_weights()
        optimal_weights = model.get_weights()
        last_index = 0
    else:
        fid = open(logfile, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        train_loss = log_data['train_loss']
        val_loss = log_data['val_loss']
        test_loss = log_data['test_loss']
        val_measure = log_data['val_measure']
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        last_index = log_data['last_index']
    
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam')
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
            for layer in model.layers:
                if layer.name.startswith('fix_'):
                    layer.trainable = False
                
        h = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)
        val_loss.append(model.evaluate_generator(val_gen, val_steps))
        test_loss.append(model.evaluate_generator(test_gen, test_steps))
        train_loss.append(h.history['loss'])
                
        if val_loss[-1] < val_measure:
            optimal_weights = model.get_weights()
            val_measure = val_loss[-1]
                    
        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'val_measure': val_measure,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'test_loss': test_loss,
                        'last_index': epoch_index +1}
            
            fid = open(logfile, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
    
    model.set_weights(optimal_weights)
    
    linear_encoder_weights = model.get_layer('linear_encoder').get_weights()
    nonlinear_decoder_weights = {}
    for layer in model.layers:
        if layer.name.startswith('nonlinear_decoder'):
            nonlinear_decoder_weights[layer.name] = layer.get_weights()
    
    return linear_encoder_weights, nonlinear_decoder_weights, train_loss, val_loss, test_loss


def train_MCLwPS(args):
    prefix = args[0]
    dataset = args[1]
    classifier = args[2]
    encode_shape = args[3]
    regularization = args[4]
    distillation_loss = args[5]
    distillation_coef = args[6]
    D_LR = args[7]
    D_Epoch = args[8]
    prior_LR = args[9]
    prior_Epoch = args[10]
    LR = args[11]
    Epoch = args[12]
    exp = args[13]
    confidence_score = args[14]
    
    conv_regularizer, conv_constraint, bl_regularizer, bl_constraint = regularization
    
    """ if result exists, load and return """
    result_filename = '_'.join([str(v) for v in args]) + '.pickle'
    result_filename = os.path.join(OUTPUT_DIR, result_filename)
    
    if os.path.exists(result_filename):
        fid = open(result_filename, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result
    
    """ distill encoder, encoder + decoder """
    
    distillation_encoder_logfile = 'disen_' + '_'.join([str(v) for v in args[1:]]) + '.pickle'
    distillation_encoder_logfile = os.path.join(LOG_DIR, distillation_encoder_logfile)
    distillation_e_outputs = distill_encoder_semi_supervised(distillation_encoder_logfile,
                                                             dataset, 
                                                             classifier, 
                                                             encode_shape, 
                                                             distillation_loss, 
                                                             confidence_score,
                                                             conv_regularizer, 
                                                             conv_constraint, 
                                                             D_LR, D_Epoch,
                                                             prior_LR, prior_Epoch)

    
    

    distillation_ed_logfile = 'disde_' + '_'.join([str(v) for v in args[1:]]) + '.pickle'
    distillation_ed_logfile = os.path.join(LOG_DIR, distillation_ed_logfile)
    distillation_ed_outputs = distill_encoder_decoder_semi_supervised(distillation_ed_logfile,
                                                                      dataset, 
                                                                      classifier, 
                                                                      encode_shape, 
                                                                      distillation_e_outputs[0],
                                                                      distillation_loss, 
                                                                      confidence_score,
                                                                      conv_regularizer, 
                                                                      conv_constraint, 
                                                                      bl_regularizer,
                                                                      bl_constraint,
                                                                      D_LR, D_Epoch,
                                                                      prior_LR, prior_Epoch)
    
    """ load prior model """
    prior_params = ['priorS', dataset, classifier, encode_shape, distillation_loss, prior_LR, prior_Epoch, confidence_score]
    prior_filename = '_'.join([str(v) for v in prior_params]) + '.pickle'
    prior_filename = os.path.join(OUTPUT_DIR, prior_filename)
    
    if not os.path.exists(prior_filename):
        raise RuntimeError('prior model doesnt exist!')
        
    with open(prior_filename, 'rb') as fid:
        prior_model_data = pickle.load(fid)
        
    # load dataset
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    

    x_extra = np.load(os.path.join(Configuration.data_dir, dataset + '_x_no_label.npy'))
    prior_model = Models.get_prior_model_allcnn(x_train.shape[1:], y_train.shape[-1], encode_shape, 'low', conv_regularizer, conv_constraint, bl_regularizer, bl_constraint)
    prior_model.compile('adam', 'categorical_crossentropy', ['acc',])
    for layer_name in prior_model_data['weights'].keys():
        prior_model.get_layer(layer_name).set_weights(prior_model_data['weights'][layer_name])
    
    x_augment, y_augment = get_pseudo_label(prior_model, x_extra, confidence_score)
    if x_augment is not None:
        x_train = np.concatenate((x_train, x_augment), axis=0)
        y_train = np.concatenate((y_train, y_augment), axis=0)
    
    input_shape = x_train.shape[1:]
    n_class = y_train.shape[-1]
    
    
    get_model = Models.get_MCLwP_allcnn
    
    model = get_model(input_shape,
                      n_class, 
                      encode_shape, 
                      True,
                      distillation_coef, 
                      'low',
                      conv_regularizer, 
                      conv_constraint,
                      bl_regularizer,
                      bl_constraint)
    
    model.compile('adam', 
                  {classifier + '_prediction': 'categorical_crossentropy', 'fix_' + classifier + '_prediction': None},
                  {classifier + '_prediction': ['acc']})
    

    
    """ set weights """
    # set weights for the classifier part from nonlinear sensing model
    for layer in model.layers:
        if layer.name.startswith(classifier):
            layer.set_weights(prior_model_data['weights'][layer.name])

    # set weights for the trainable linear encoder part
    model.get_layer('linear_encoder').set_weights(distillation_ed_outputs[0])

    # set weights for the trainable nonlinear decoder part
    for layer_name in distillation_ed_outputs[1].keys():
        if layer_name.startswith('nonlinear_decoder'):
            model.get_layer(layer_name).set_weights(distillation_ed_outputs[1][layer_name])
    
    # set the weights for fixed parts
    for layer_name in prior_model_data['weights'].keys():
        model.get_layer('fix_' + layer_name).set_weights(prior_model_data['weights'][layer_name])
        model.get_layer('fix_' + layer_name).trainable = False

    
    train_gen, train_steps = Utility.get_sensing_data_generator(x_train, y_train, BATCH_SIZE, shuffle=True, augmentation=True)
    val_gen, val_steps = Utility.get_sensing_data_generator(x_val, y_val, BATCH_SIZE, shuffle=False, augmentation=False)
    test_gen, test_steps = Utility.get_sensing_data_generator(x_test, y_test, BATCH_SIZE, shuffle=False, augmentation=False)
    
    metrics = model.metrics_names

    log_file = '_'.join([str(v) for v in args]) + '.pickle'
    log_file = os.path.join(LOG_DIR, log_file)
    
    acc_key = classifier + '_prediction_acc'
    
    if not os.path.exists(log_file):
        
        current_weights = model.get_weights()
        optimal_weights = model.get_weights()
        history = {}
        
        train_p = model.evaluate(x_train, y_train, BATCH_SIZE)
        val_p = model.evaluate(x_val, y_val, BATCH_SIZE)
        test_p = model.evaluate(x_test, y_test, BATCH_SIZE)
        
        history['train_acc'] = [train_p[metrics.index(acc_key)]]
        history['val_acc'] = [val_p[metrics.index(acc_key)]]
        history['test_acc'] = [test_p[metrics.index(acc_key)]]
        
        val_measure = history['val_acc'][-1]
        last_index = 0
        
    else:
        fid = open(log_file, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        last_index = log_data['last_index']
        val_measure = log_data['val_measure']
        history = log_data['history']
        
    learning_rates = []
    
    for lr, sc in zip(LR, Epoch):
        learning_rates += [lr,]*sc
    
    
    for epoch_index in range(last_index, len(learning_rates)):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam',
                  {classifier + '_prediction': 'categorical_crossentropy', 'fix_' + classifier + '_prediction': None},
                  {classifier + '_prediction': ['acc']})

            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)
            
            for layer in model.layers:
                if layer.name.startswith('fix_'):
                    layer.trainable = False
        
        model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)
        
        if epoch_index % Configuration.VAL_MEASURE_INTERVAL == 0:
            
            train_p = model.evaluate(x_train, y_train, BATCH_SIZE)
            val_p = model.evaluate(x_val, y_val, BATCH_SIZE)
            test_p = model.evaluate(x_test, y_test, BATCH_SIZE)
            
            history['train_acc'].append(train_p[metrics.index(acc_key)])
            history['val_acc'].append(val_p[metrics.index(acc_key)])
            history['test_acc'].append(test_p[metrics.index(acc_key)])
            
                
            if history['val_acc'][-1] > val_measure:
                optimal_weights = model.get_weights()
                val_measure = history['val_acc'][-1]

        current_weights = model.get_weights()
        
        if epoch_index % Configuration.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'val_measure': val_measure,
                        'history': history,
                        'last_index': epoch_index +1}
            
            fid = open(log_file, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
                        
    
    model.set_weights(optimal_weights)
    train_p = model.evaluate(x_train, y_train, BATCH_SIZE)
    val_p = model.evaluate(x_val, y_val, BATCH_SIZE)
    test_p = model.evaluate(x_test, y_test, BATCH_SIZE)
        
    weights = {}
    for layer in model.layers:
        weights[layer.name] = layer.get_weights()
    
    result = {'weights': weights,
              'history': history,
              'distillation_encoder': distillation_e_outputs,
              'distillation_encoder_decoder': distillation_ed_outputs,
              'train_acc': train_p[metrics.index(acc_key)],
              'val_acc': val_p[metrics.index(acc_key)],
              'test_acc': test_p[metrics.index(acc_key)]}
    
    return result
