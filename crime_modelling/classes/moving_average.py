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
                mean = np.sum(data_chunk)/time_window
                results[row_num,:] = ["MOVING AVERAGE", str(end_date)[:10], str(frame[i,j,0]),str(np.sum(frame[i,j,2:])),
                       str(mean)]
                row_num += 1
        return results

# parser = argparse.ArgumentParser(
#            description = "Moving Average predictive model\n\n" + "Outputs a prediction set using a moving average." +
#            "This is done by averaging fields over a" +  
#            "time period for each spatial id. Output is a file of the following columns:\n\n" +
#            "Method (Moving Average), Week, Grid ID, Actual Counts, Predictions")
#    required = parser.add_argument_group("required arguments")
#    required.add_argument("-data", help = "Name of dataframe of SpatialDataSet type")
#    parser.add_argument("-time_window", help = "Integer period of time in weeks to average values over.",
#                        default = 52)
#    parser.add_argument("-crime_types", help = "Indices of crime types in data field to include in model in a list.",
#                        default = "")
#    parser.add_argument("-output_file", help = "Name of resulting predictions file.", default = "predictions.csv")
#    parser.add_argument("-output_columns", help ="Rename the output columns. Must be of length 5", default = 
#                        "Method, Week, Grid ID, Actual Counts, Predictions")
#    args = parser.parse_args()
#
#    if not args.data:
#        print("data argument must be provided.")
#        return
#    moving_average(args.data, args.time_window, args.crime_types)


    

