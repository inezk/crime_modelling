from predictor import Predictor
import numpy as np

#assuming data is of type SpatialDataSet
#crime_types field needs to be an array of integers (indices)
#time_window in weeks
class moving_average(Predictor):

    def __init__(self, data, time_window = 52, crime_types = ""):
        Predictor.__init__(self, data)
        self.predictions = self.predict(self.SDS.start_date, self.SDS.end_date, time_window, crime_types)
#note start window - end window includes whole data used in model, so actual data pts start @ start +time_window
#end_window is assuming including it
    def predict(self, start_window, end_window, time_window, crime_types):
        frame = self.data
        if(crime_types != ""):
            frame = frame[:,:,crime_types]
        results = []
        print(type(frame[0,:,1]), start_window)
        start_index = np.where(frame[0,:,1] == start_window)
        end_index = np.where(frame[0,:,1] == end_window)
        for i in range(0, len(frame)):
            for j in range(start_index + time_window, end_index + 1):
                end_date = frame[i,j,1]
                data_chunk = frame[i][j-time_window:j, 2:]
                mean = np.sum(data_chunk)/time_window
                results += [[frame[i,j,0], end_date, mean]]
        return results

    def export(self, filename = "moving_average_predictions.csv", colnames = ["Grid ID", "Week", "# Crimes"]):
        super().export(colnames, filename)
