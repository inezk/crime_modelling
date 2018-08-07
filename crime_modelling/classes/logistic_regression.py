# -*- coding: utf-8 -*-
"""
Logistic Regression class
Created on Tue Jul  3 13:22:25 2018

@author: inezk
"""

from sklearn import linear_model
from predictor_class import Predictor
import numpy as np

class logistic_regression(Predictor):
    def __init__(self, data, filename, penalty = 'l2',reg_constant = 1.0, feature_cols = [],
                 lag_window = 8, start_index = 104, end_index = 260, moving_window = 52): #indices in terms of weeks, including week of end_index
        Predictor.__init__(self, data, filename)
        self.data = self.structure_data(data.counts, lag_window, feature_cols)
        self.model = self.train(penalty, reg_constant) 
        self.predictions = self.predict(moving_window, start_index, end_index, lag_window)
    
    #note: 1st column of data frame will be labels
    #restructure data so it will be lagged - lag_window weeks of data per row, with 0/1 indicating appearance of counts
    def structure_data(self, data, lag_window, feature_cols):
        if len(feature_cols) == 0:
            feature_cols = list(range(2,len(data[0,0])))
        new_frame = np.zeros((len(data) * (len(data[0]) - lag_window), (len(feature_cols) * lag_window) + 4), dtype = object)
        row = 0
        for j in range(lag_window, len(data[0])): #for each week
            for i in range(0, len(data)):
                 new_frame[row, 4:]= data[i,j - lag_window:j,feature_cols].reshape(1, lag_window * len(feature_cols))
                 total = np.sum(data[i,j,feature_cols])
                 new_frame[row, 0] = 1 if total > 0 else 0 #label
                 new_frame[row, 1] = data[i,j, 0] #spatial_id
                 new_frame[row, 2] = data[i,j, 1] #week
                 new_frame[row, 3] = total
                 row += 1
        return new_frame      
    
    #setting up logistic regression model
    def train(self, penalty, reg_constant):
        if penalty == 'l2':
            solver = 'sag'
        else: solver = 'saga' #these solvers are best for large datasets
        model = linear_model.LogisticRegression(C = reg_constant, penalty = penalty, 
                                                solver = solver)
        return model
    
    def predict(self, moving_window, start_index, end_index, lag_window):
        frame = self.data
        time_range = end_index - start_index
        n_space = len(self.SDS.counts)
        results = np.zeros((time_range * n_space, 5), dtype = object)
        row = 0
        for i in range((start_index- lag_window) * n_space, (end_index - lag_window - 1)* n_space, n_space):
            training_chunk = frame[i - (moving_window * n_space):i,:]
            self.model.fit(training_chunk[:,4:].astype(int), training_chunk[: ,0].astype(int))
            predictions = self.model.predict_proba(frame[i: i + n_space, 4:])[:,1] #prob of crime appearing
            results[row: row + n_space, 0] = str(self.outfile)
            results[row: row + n_space, 1] = frame[i: i + n_space, 2].astype(str) 
            results[row: row + n_space, 2] = frame[i: i + n_space, 1].astype(str)
            results[row: row + n_space, 3] = frame[i: i + n_space, 3].astype(str)
            results[row: row + n_space, 4] = predictions.astype(str)
            row += n_space
        return results
            
        
        
   
        
        
    
        