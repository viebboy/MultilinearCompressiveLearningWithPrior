#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

import exp_configurations as Conf
import Utility
import sys, getopt
import pickle
import Runners
import os


def save_result(output, path):
    fid = open(path, 'wb')
    pickle.dump(output, fid)
    fid.close()

def main(argv):

    try:
      opts, args = getopt.getopt(argv,"h", ['index=' ])
    
    except getopt.GetoptError:
        sys.exit(2)
    
    for opt, arg in opts:             
        if opt == '--index':
            index = int(arg)
    
    
    parameters = [Conf.mclwop_names]
    values = [Conf.mclwop_conf]
    
    configurations = Utility.create_configuration(parameters, values)
    print('total configurations: %d' % len(configurations))
    
    assert index < len(configurations), 'The given configuration index is out of range, there are only %d configurations' % len(configurations)
    
    output = Runners.train_MCLwoP(configurations[index])

    filename = '_'.join([str(v) for v in configurations[index]]) + '.pickle'
    filename = os.path.join(Conf.output_dir, filename)
        
    save_result(output, filename)
    
if __name__ == "__main__":
    main(sys.argv[1:])