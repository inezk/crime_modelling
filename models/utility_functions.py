import pandas as pd 
import datetime

#assuming column names are Grid500X, Grid500Y,DATE_START, Grid500ID, num_crimes
def file_to_data(dataset):
    frame = pd.read_csv(dataset)
    frame["DATE_START"] = pd.to_datetime(frame["DATE_START"])
    frame["num_crimes"] = pd.to_numeric(frame["num_crimes"])
    return(frame)

def outFile(results, model_name, colnames):
    final_file = pd.DataFrame(results, columns = colnames)
    file_name = str(model_name) + "_predictions.csv"
    final_file.to_csv(file_name, sep=',', encoding='utf-8',  float_format='%.10f')


