# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 12:11:01 2018

@author: inezk
"""

from yaml import load
import pickle
from predictor import *
import argparse

f = open("training.yaml", "r")
args = load(f)

temp = dict(args)
f = open(temp['input_file'], 'rb')
dataset = pickle.load(f)
del temp['input_file']

for key in temp:
    run = temp[key]
    out_file = run['output_file']
    del run['output_file']
    fun_args = dict(run)
    fun_args = argparse.Namespace(**fun_args)
    print("Current test: " + str(key))

    if 'mov_avg' in key:
        obj = moving_average.moving_average(dataset, out_file, fun_args.moving_window, fun_args.crime_types)
    elif 'kde' in key:
        obj = kde.KDE(dataset, out_file, fun_args.moving_window, fun_args.crime_types, fun_args.bandwidth,
                      fun_args.start_index, fun_args.end_index)
    elif 'log_reg' in key:
        obj = logistic_regression.logistic_regression(dataset, out_file, fun_args.penalty, fun_args.reg_constant,
                                                          fun_args.crime_types, fun_args.lag_window,
                                                          fun_args.start_index, fun_args.end_index, 
                                                          fun_args.time_window)
    obj.export()
    print("Outputted: " + str(out_file))
    
        
        
        