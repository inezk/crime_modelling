# -*- coding: utf-8 -*-
"""
Kernel Density Estimation (KDE) class
Created on Wed Jun 27 17:15:59 2018

@author: inezk
"""
from predictor_class import Predictor
from sklearn.neighbors.kde import KernelDensity
import numpy as np

#note kde requires bandwidth in US survey feet
class KDE(Predictor):
    def __init__(self, data, filename, moving_window = 52, crime_types = [2,3,4,5], bandwidth = 500, start_index = 104,
                 end_index = 260):
        Predictor.__init__(self, data, filename)
        self.model = self.train(bandwidth)
        self.predictions = self.predict(start_index, end_index, 
                                        moving_window, crime_types, bandwidth)
    
    #setting up KDE
    def train(self, bandwidth):
        return KernelDensity(bandwidth)
    
    #assuming data is sorted by date
    def predict(self, start_index, end_index, time_window, crime_types, bandwidth):
        frame = self.SDS.coords
        counts = self.data
        if len(crime_types) > 0:
            counts = counts[:,:, [0,1] + crime_types]
        row_num = 0
        n_space = len(self.data)
        time_range = end_index - start_index
        results = np.zeros((time_range * n_space, 5), dtype = object)
        for i in range(start_index, end_index):
            date_range = counts[:, i- time_window:i, :]
            instances = np.sum(date_range[:,:,2:], axis = (2,1), dtype = np.int64) #get instances 
            xtrain = np.repeat(frame, instances, axis = 0)
            self.model.fit(xtrain) #fitting model
            predictions = self.model.score_samples(frame).astype(float)     
            #save prediction to output frame
            results[row_num: row_num + n_space,0] = str(self.outfile)
            results[row_num: row_num + n_space,1] = counts[:,i,1].astype(str) 
            results[row_num: row_num + n_space,2] = counts[:,i,0].astype(str)
            results[row_num: row_num + n_space,3] = np.sum(counts[:,i, 2:], axis = 1).astype(str)
            results[row_num: row_num + n_space,4] = predictions.astype(str)
            row_num += 1
        return results
        