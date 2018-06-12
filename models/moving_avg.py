import numpy as np
import datetime
import time

#note - file is ordered by dates earliest to latest
#data frame colnames = week, cell, num_crimes

def frame_to_data(dataset):
    data = open(dataset,"r").readlines()[1:]
    all_data = np.full((len(data), len(data[0].strip().split(',')) - 2), 0, dtype = "object")
    for i in range(0, len(data)):
        all_data[i] = data[i].strip().split(",")[2:]
        all_data[i, 0] = datetime.datetime.strptime(all_data[i,0], '"%Y-%m-%d"')
        all_data[i, 2] = float(all_data[i,2]) 
    return(all_data)

def frame_avg(frame):
    results = []
    start_date = frame[0,0]
    for row in frame:
        end_date = row[0]
        num_weeks = (end_date - start_date)
        num_weeks = num_weeks.days/7 #for one year moving avg cal starting 1 yr from dataset start date
        if(num_weeks > 52):
            start_delta = end_date - datetime.timedelta(weeks = 52)
            mask = (frame[:,0] < end_date) & (frame[:,0] >= start_delta) & (frame[:,1] == row[1])
            sub = frame[np.where(mask)[0],:] #filter for date range and cell_id
            mean = sum(sub[:,2])/52 #get mean
            if(type(mean) != float): #account for when nrow(sub) == 0
                mean = 0
            results += [[row[0],row[1], mean]] #week, cell, avg
    return results     

def outFile(results):
    f = open("moving_avg_predictions.csv", "w")
    f.write("week,")
    f.write("cell,")
    f.write("predicted average")
    f.write("\n")
    for row in results:
        for elem in row:
            if type(elem) != int and type(elem) != str and type(elem) != float:
                f.write(elem.strftime('%m/%d/%Y'))
            else: f.write(str(elem))
            f.write(",")
        f.write("\n")

def mov_avg(dataset):
    frame = frame_to_data(dataset)
    results = frame_avg(frame)
    outFile(results)

mov_avg("grouped_crime.csv")
