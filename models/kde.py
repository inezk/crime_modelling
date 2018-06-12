import numpy as np
import datetime
from sklearn.neighbors.kde import KernelDensity

def frame_to_data(dataset):
    data = open(dataset,"r").readlines()[1:]
    all_data = np.full((len(data), len(data[0].strip().split(','))), 0, dtype = "object")
    for i in range(0, len(data)):
        all_data[i] = data[i].strip().split(",")
        all_data[i, 2] = datetime.datetime.strptime(all_data[i,2], '"%Y-%m-%d"')
        all_data[i, 4] = float(all_data[i,4]) 
    return(all_data)

def get_kde(frame):
    results = []
    start_date = frame[0,2]
    for row in frame:
        end_date = row[2]
        num_weeks = (end_date - start_date)
        num_weeks = num_weeks.days/7 #for one year moving avg cal starting 1 yr from dataset start date
        if(num_weeks > 52):
            start_delta = end_date - datetime.timedelta(weeks = 52)
            mask = (frame[:,2] < end_date) & (frame[:,2] >= start_delta)
            sub = frame[np.where(mask)[0],:] #filter for date range and cell_id
            xtrain = sub[:,0:2]
            kde = KernelDensity(bandwidth = 500)
            kde.fit(xtrain)
            sample = np.asmatrix(np.array(row[0:2]))
            prediction = float(kde.score_samples(sample)[0])
            results +=[[row[0], row[1], row[2], row[3], prediction]]
    return results

def outFile(results):
    f = open("kde_predictions.csv", "w")
    f.write("GRID500X,")
    f.write("GRID500Y,")
    f.write("week,")
    f.write("cell,")
    f.write("predicted density")
    f.write("\n")
    for row in results:
        for elem in row:
            if type(elem) != int and type(elem) != str and type(elem) != float:
                f.write(elem.strftime('%m/%d/%Y')) #convert dates to strings
            else: f.write(str(elem))
            f.write(",")
        f.write("\n")

def kde(dataset):
    data = frame_to_data(dataset)
    results = get_kde(data)
    outFile(results)

kde("grouped_crime.csv")