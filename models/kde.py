import numpy as np
import datetime
from sklearn.neighbors.kde import KernelDensity
import utility_functions as ut

#assuming column names are Grid500X, Grid500Y,DATE_START, Grid500ID, num_crimes

def get_kde(frame):
    results = []
    start_date = frame.iloc[0,2]
    for index, row in frame.iterrows():
        end_date = row.iloc[2]
        num_weeks = (end_date - start_date)
        num_weeks = num_weeks.days/7 #for one year moving avg cal starting 1 yr from dataset start date
        if(num_weeks > 52):
            start_delta = end_date - datetime.timedelta(weeks = 52)
            mask = (frame["DATE_START"] < end_date) & (frame["DATE_START"] >= start_delta)
            sub = frame.loc[mask] #filter for date range and cell_id
            xtrain = sub[["Grid500X", "Grid500Y"]]
            kde = KernelDensity(bandwidth = 500)
            kde.fit(xtrain)
            sample = np.asmatrix(np.array(row.iloc[0:2]))
            prediction = float(kde.score_samples(sample)[0])
            results +=[[row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], prediction]]
    return results

def kde(dataset):
    data = ut.file_to_data(dataset)
    results = get_kde(data)
    colnames = ["GRID500X", "GRID500Y", "week", "cell", "prediction density"]
    ut.outFile(results, "kde", colnames)

kde("grouped_crime.csv")