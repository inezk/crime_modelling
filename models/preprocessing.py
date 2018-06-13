import pandas as pd
from datetime import datetime

#make sure doc is csv

def preprocess(file):
    aprs = pd.read_csv(file)
    crime_data = aprs[["DATE_START", "HIERARCHY","Grid500ID", "Grid500X", "Grid500Y"]]
    crime_data["HIERARCHY"] = pd.to_numeric(crime_data["HIERARCHY"])
    crime_data["DATE_START"] = pd.to_datetime(crime_data["DATE_START"])
    aprs["DATE_START"] = aprs["DATE_START"].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M') - timedelta(days = datetime.strptime(x, '%m/%d/%Y %H:%M').weekday()))
    crime_data = crime_data.loc[(crime_data.HIERARCHY >= 1) & (crime_data.HIERARCHY <= 4)]
    crime_data = crime_data.groupby(["Grid500X", "Grid500Y", "DATE_START", "Grid500ID"])
    crime_data = crime_data.sum()
    crime_data = crime_data.sort_values(by = 'DATE_START')
    crime_data.to_csv("grouped_crime_python.csv")

preprocess("aprs_5years.csv")