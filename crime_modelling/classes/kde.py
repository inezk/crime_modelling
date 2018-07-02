# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:15:59 2018

@author: inezk
"""

from predictor import Predictor
from sklearn.neighbors.kde import KernelDensity
import numpy as np

#note kde requires bandwidth in US survey feet
class KDE(Predictor):
    def __init__(self, data, time_window = 52, crime_types = "", bandwidth = 500):
        Predictor.__init__(self, data)
        self.predictions = self.predict(self.SDS.start_date, self.SDS.end_date, 
                                        time_window, crime_types, bandwidth)
    
    #assuming data is sorted by date
    def predict(self, start_window, end_window, time_window, crime_types, bandwidth):
        frame = self.SDS.coords
        counts = self.data
        if crime_types != "":
            counts = counts[:,:, crime_types]
        row_num = 0
        start_index = np.where([counts[0,i,1] == start_window for i in range(0, len(counts[0]))])
        start_index = np.asscalar(start_index[0])
        end_index = np.where([counts[0,i,1] == end_window for i in range(0, len(counts[0]))])
        end_index = np.asscalar(end_index[0])
        results = np.zeros((len(self.SDS.view_frame), 5), dtype = object)
        for i in range(0, len(counts)):
            for j in range(start_index + time_window, end_index + 1):
                actual_count = np.sum(counts[i,j,2:])
                if actual_count != 0:
                    end_date = counts[i,j,1]
                    date_range = counts[:, j-time_window:j, 2:]
                    instances = np.sum(date_range, axis = (1,2), dtype = np.int64)
                    xtrain = np.repeat(frame, instances, axis = 0)
                    kde = KernelDensity(bandwidth)
                    kde.fit(xtrain)
                    sample = frame[i,:].reshape(1, -1)
                    prediction = float(kde.score_samples(sample)[0])              
                    results[row_num,:] = ["KDE", str(end_date)[:10], str(counts[i,j,0]), str(actual_count), str(prediction)]
                    row_num += 1
        return results
        