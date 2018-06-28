from dataset import SpatialDataSet
import pandas as pd
import numpy as np
from datetime import timedelta 

#make sure doc is csv

#converts to nearest monday for week by week data
def setup_time_frame(time_col):
    time_col = pd.to_datetime(time_col.T.squeeze()) #computationally expensive - inner functions needed  to convert from pd.Dataframe to pd.Series
    for i in range(0, len(time_col)):
        time_col[i] = time_col[i] - timedelta(days=(time_col[i].weekday())) #convert to nearest monday
    time_col = np.array(time_col).reshape(len(time_col), 1)
    return time_col

def preprocess(file, id_col, time_col, coords_cols, feature_cols, type_col, date_format = "%Y/%d/%m"):
    crime = pd.read_csv(file)
    id_frame = np.array(crime[id_col])
    time_frame = setup_time_frame(crime[time_col])
    coord_frame = np.array(crime[coords_cols])
    feature_frame = np.array(crime[feature_cols])
    type_frame = np.array(crime[type_col])
    dataset_obj = SpatialDataSet(id_frame, time_frame, coord_frame, feature_frame, 
        type_frame, date_format = "%m/%d/%Y")
    return dataset_obj


#preprocess("aprs_5years.csv", ["Grid500ID"], ["DATE_START"], ["Grid500X", "Grid500Y"],
#["ARREST_FLAG", "VICTIMS", "SUSPECTS"], ["HIERARCHY"])

