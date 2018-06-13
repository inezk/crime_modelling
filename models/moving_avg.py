import numpy as np
import datetime
import utility_functions as ut

#note - file is ordered by dates earliest to latest
#data frame 

#assuming column names are Grid500X, Grid500Y,DATE_START, Grid500ID, num_crimes

def frame_avg(frame):
    frame = frame.drop(columns = ["Grid500X", "Grid500Y"], axis = 1)
    results = []
    start_date = frame.iloc[0,0]
    for index, row in frame.iterrows():
        end_date = row.iloc[0]
        num_weeks = (end_date - start_date)
        num_weeks = num_weeks.days/7 #for one year moving avg cal starting 1 yr from dataset start date
        if(num_weeks > 52):
            start_delta = end_date - datetime.timedelta(weeks = 52)
            mask = (frame["DATE_START"] < end_date) & (frame["DATE_START"] >= start_delta) & (frame["Grid500ID"] == row.iloc[1])
            sub = frame.loc[mask] #filter for date range and cell_id
            mean = sub["num_crimes"].sum()/52 #get mean
            results += [[row.iloc[0],row.iloc[1], mean]] #week, cell, avg
    return results     

def mov_avg(dataset):
    frame = ut.file_to_data(dataset)
    results = frame_avg(frame)
    colnames = ["week", "cell", "predicted average"]
    ut.outFile(results, "moving_avg", colnames)

mov_avg("grouped_crime.csv")
