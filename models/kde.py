# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:15:59 2018

@author: inezk
"""

from predictor import Predictor
from sklearn.neighbors.kde import KernelDensity
from numpy import np

#note kde requires bandwidth in US survey feet
class KDE(Predictor):
    def __init__(self, data, time_window = 42, crime_types = "", bandwidth = 500):
        Predictor.__init__(self, data)
        self.predictions = self.predict(self.SDS.start_date, self.SDS.end_date, 
                                        time_window, crime_types, bandwidth)
    
    def predict(self, start_window, end_window, time_window, crime_types, bandwidth):
        results = []
        frame = self.SDS.coords
        if crime_types != "":
            frame = frame[:,:,crime_types]
        time_col = np.array(self.data[:,:,1], dtype = "datetime64")
        for i in range(0, len(frame)):
            row = frame[i]
            end_date = time_col[i]
            num_weeks = (end_date - start_window)
            num_weeks = num_weeks/np.timedelta64(7, "D")
            if(num_weeks > time_window):
                start_delta = end_date - np.timedelta64(time_window, "w")
                mask = np.where((time_col < end_date) & (time_col >= start_delta))[0]
                xtrain = frame[mask,:] #get rows within a certain date
                kde = KernelDensity(bandwidth)
                kde.fit(xtrain)
                sample = np.asmatrix(np.array(row))
                prediction = float(kde.score_samples(sample)[0])
                results += row + [self.SDS.view_frame[0,i], time_col[i], prediction]
            if end_date == end_window: break
        return results
    
    def export(self, filename, colnames = ["XCOORD","YCOORD", "Grid ID", "Week", "Density"]):
        filename = "moving_average_predictions.csv"
        super(KDE, self).export(colnames, filename)
        