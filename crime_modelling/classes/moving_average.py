"""
Moving Average class
"""
from predictor_class import Predictor
import numpy as np

#assuming data is of type SpatialDataSet
#crime_types field needs to be an array of integers (indices)
#time_window in weeks

class moving_average(Predictor):
       
    def __init__(self, data, filename, time_window = 52, crime_types = []):
        Predictor.__init__(self, data, filename)
        self.predictions = self.predict(self.SDS.start_date, self.SDS.end_date, time_window, crime_types)
#note start window - end window includes whole data used in model, so actual data pts start @ start +time_window
#end_window is assuming including it
    def predict(self, start_window, end_window, time_window, crime_types):
        frame = self.data
        if(len(crime_types) > 0):
            frame = frame[:,:,[0,1] + crime_types]
        #getting start/end indices
        start_index = np.where([frame[0,i,1] == start_window for i in range(0, len(frame[0]))])
        start_index = np.asscalar(start_index[0])
        end_index = np.where([frame[0,i,1] == end_window for i in range(0, len(frame[0]))])
        end_index = np.asscalar(end_index[0])
        results = np.zeros((len(frame) * len(frame[0]), 5), dtype = object)
        row_num = 0
        for i in range(0, len(frame)):
            for j in range(start_index + time_window, end_index + 1):
                end_date = frame[i,j,1]
                data_chunk = frame[i, j-time_window:j, 2:]  
                #getting averages - predictions
                mean = np.sum(data_chunk)/time_window
                results[row_num,:] = [str(self.outfile), 
                       str(end_date)[:10], str(frame[i,j,0]),str(np.sum(frame[i,j,2:])),
                       str(mean)]
                row_num += 1
        return results



    

