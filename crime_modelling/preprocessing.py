<<<<<<< HEAD
from dataset import SpatialDataSet
import pandas as pd
import numpy as np
from datetime import timedelta 
import argparse

#make sure doc is csv
"""
Preprocessing stage of data pipeline. More information is found in def main().
example usage in command:

python3 preprocessing.py  -input_file aprs_5years.csv -id_col 'Grid500ID' -time_col 'DATE_START' -coords_cols 'Grid500X', 'Grid500Y' -type_col 'HIERARCHY'
"""

#converts to nearest monday for week by week data
def setup_time_frame(time_col):
    time_col = pd.to_datetime(time_col.T.squeeze()) #computationally expensive - inner functions needed  to convert from pd.Dataframe to pd.Series
    for i in range(0, len(time_col)):
        time_col[i] = time_col[i] - timedelta(days=(time_col[i].weekday())) #convert to nearest monday
    time_col = np.array(time_col).reshape(len(time_col), 1)
    return time_col

def preprocess(file, out_file, id_col, time_col, coords_cols, feature_cols, type_col, date_format = "%Y/%d/%m",
               uniform_areas = True, spatial_unit_areas = 250000):
    crime = pd.read_csv(file)
    id_frame = crime[id_col].values
    if date_format == "%Y/%d/%m": time_frame = setup_time_frame(crime[time_col])
    else: time_frame = crime[time_col].values
    coord_frame = crime[coords_cols].values
    if feature_cols != "": feature_frame = crime[feature_cols].values
    else: feature_frame = ""
    type_frame = crime[type_col].values
    dataset_obj = SpatialDataSet(out_file, id_frame, time_frame, coord_frame, feature_frame, 
        type_frame, date_format, uniform_areas, spatial_unit_areas)
    return dataset_obj

def main():
    parser = argparse.ArgumentParser(description = "Mellon Grant preprocessing module\n\n" +
                                     "This module takes in a csv as an input and produces a data structure of" +
                                     "class SpatialDataSet to use in the prediction module. NOTE: The prediction" + 
                                     "module is designed to specifically work with the SpatialDataSet class, so this"+
                                     "step is necessary before moving on to later modules. The output of this module" +
                                     "will be an object file containing the SpatialDataSet object.\n\n"
                                     "Input file should be comma-delimited with the following columns:\n\n"+
                                     "id_col - Column that provides the spatial ID coordinates of each data point.\n\n"+
                                     "time_col - Column that provides date and time of data point.\n\n" + 
                                     "coords_cols - Column that provides coordinates of data point. Assumption is coordinates are in US survey feet.\n\n"+
                                     "feature_cols - Column(s) that are features concerning the data point ex. victims.\n\n" + 
                                     "type_col - Column that separates crimes by type. \n\n"+
                                     "date_format - Format of fields in time_col, and should be in standard string formatting (default %Y/%m/%d).\n\n"+
                                     "uniform_areas - Boolean indicating if the spatial areas are the same size.\n\n" +
                                     "spatial_unit_areas - If uniform_areas = True, then it is an integer indicating the grid area of each spatial unit. If"+
                                     "uniform_areas = False, then it is an array of integers indicating the grid area of each spatial unit. The index in the array" + 
                                     "should correspond to the rows of the input file.", formatter_class=argparse.RawTextHelpFormatter)
    required = parser.add_argument_group('required arguments')
    required.add_argument("-input_file", help = "Name of file containing data points")
    required.add_argument("-output_file", help = "Name of output file")
    required.add_argument("-id_col", nargs = 1, help = "Spatial ID Column")
    required.add_argument("-time_col",nargs = 1, help = "Time Events Column")
    required.add_argument("-coords_cols", nargs = 2, help = "Coordinate Columns: First column is X, second Y")
    
    parser.add_argument("-feature_cols", nargs = "*", help = "Any Additional Features Column", default = "")
    
    required.add_argument("-type_col", nargs = 1, help = "Crime Type Column")
    parser.add_argument("-date_format", nargs = 1, help = "Date format in Time Column", default = "%Y/%m/%d")
    
    parser.add_argument("-uniform_areas", help ="Boolean indicating if grid sizes are uniform", default = True)
    parser.add_argument("-spatial_unit_areas", nargs = "*", help = "Area of spatial units", default = 250000)
    
    # Parse arguments and check for required args.
    args = parser.parse_args()
    if not args.input_file:
        print("input-file argument must be provided.")
        return  
    if not args.id_col:
        print("id_col argument must be provided.")   
        return
    if not args.time_col:
        print("time_col argument must be provided.")
        return
    if not args.coords_cols:
        print("coords_cols argument must be provided.")
        return
    if not args.type_col:
        print("type_col argument must be provided.")
        return   
    if args.uniform_areas and type(args.spatial_unit_areas) != int:
        print("uniform_areas set to True but spatial_unit_areas is not an integer. Set uniform_areas = False or provide an integer.")
        return
    if not args.uniform_areas and type(args.spatial_unit_areas) == int:
        print("uniform_areas set to False but spatial_unit_areas is an integer. Set uniform_areas = True or provider an array of integers.")
        return
    
    file = preprocess(args.input_file,args.output_file,  args.id_col, args.time_col, args.coords_cols, args.feature_cols,
                      args.type_col, args.date_format, args.uniform_areas, args.spatial_unit_areas)
    return file.export()
    
if __name__ == "__main__":
    main()



=======
from dataset import SpatialDataSet
import pandas as pd
import numpy as np
from datetime import timedelta 
import argparse

#make sure doc is csv
"""
Preprocessing stage of data pipeline. More information is found in def main().
example usage in command:

python3 preprocessing.py  -input_file aprs_5years.csv -id_col 'Grid500ID' -time_col 'DATE_START' -coords_cols 'Grid500X', 'Grid500Y' -type_col 'HIERARCHY'
"""

#converts to nearest monday for week by week data
def setup_time_frame(time_col):
    time_col = pd.to_datetime(time_col.T.squeeze()) #computationally expensive - inner functions needed  to convert from pd.Dataframe to pd.Series
    for i in range(0, len(time_col)):
        time_col[i] = time_col[i] - timedelta(days=(time_col[i].weekday())) #convert to nearest monday
    time_col = np.array(time_col).reshape(len(time_col), 1)
    return time_col

def preprocess(file, out_file, id_col, time_col, coords_cols, feature_cols, type_col, date_format = "%Y/%d/%m",
               uniform_areas = True, spatial_unit_areas = 250000):
    crime = pd.read_csv(file)
    id_frame = crime[id_col].values
    if date_format == "%Y/%d/%m": time_frame = setup_time_frame(crime[time_col])
    else: time_frame = crime[time_col].values
    coord_frame = crime[coords_cols].values
    if feature_cols != "": feature_frame = crime[feature_cols].values
    else: feature_frame = ""
    type_frame = crime[type_col].values
    dataset_obj = SpatialDataSet(out_file, id_frame, time_frame, coord_frame, feature_frame, 
        type_frame, date_format, uniform_areas, spatial_unit_areas)
    return dataset_obj

def main():
    parser = argparse.ArgumentParser(description = "Mellon Grant preprocessing module\n\n" +
                                     "This module takes in a csv as an input and produces a data structure of" +
                                     "class SpatialDataSet to use in the prediction module. NOTE: The prediction" + 
                                     "module is designed to specifically work with the SpatialDataSet class, so this"+
                                     "step is necessary before moving on to later modules. The output of this module" +
                                     "will be an object file containing the SpatialDataSet object.\n\n"
                                     "Input file should be comma-delimited with the following columns:\n\n"+
                                     "id_col - Column that provides the spatial ID coordinates of each data point.\n\n"+
                                     "time_col - Column that provides date and time of data point.\n\n" + 
                                     "coords_cols - Column that provides coordinates of data point. Assumption is coordinates are in US survey feet.\n\n"+
                                     "feature_cols - Column(s) that are features concerning the data point ex. victims.\n\n" + 
                                     "type_col - Column that separates crimes by type. \n\n"+
                                     "date_format - Format of fields in time_col, and should be in standard string formatting (default %Y/%m/%d).\n\n"+
                                     "uniform_areas - Boolean indicating if the spatial areas are the same size.\n\n" +
                                     "spatial_unit_areas - If uniform_areas = True, then it is an integer indicating the grid area of each spatial unit. If"+
                                     "uniform_areas = False, then it is an array of integers indicating the grid area of each spatial unit. The index in the array" + 
                                     "should correspond to the rows of the input file.", formatter_class=argparse.RawTextHelpFormatter)
    required = parser.add_argument_group('required arguments')
    required.add_argument("-input_file", help = "Name of file containing data points")
    required.add_argument("-output_file", help = "Name of output file")
    required.add_argument("-id_col", nargs = 1, help = "Spatial ID Column")
    required.add_argument("-time_col",nargs = 1, help = "Time Events Column")
    required.add_argument("-coords_cols", nargs = 2, help = "Coordinate Columns: First column is X, second Y")
    
    parser.add_argument("-feature_cols", nargs = "*", help = "Any Additional Features Column", default = "")
    
    required.add_argument("-type_col", nargs = 1, help = "Crime Type Column")
    parser.add_argument("-date_format", nargs = 1, help = "Date format in Time Column", default = "%Y/%m/%d")
    
    parser.add_argument("-uniform_areas", help ="Boolean indicating if grid sizes are uniform", default = True)
    parser.add_argument("-spatial_unit_areas", nargs = "*", help = "Area of spatial units", default = 250000)
    
    # Parse arguments and check for required args.
    args = parser.parse_args()
    if not args.input_file:
        print("input-file argument must be provided.")
        return  
    if not args.id_col:
        print("id_col argument must be provided.")   
        return
    if not args.time_col:
        print("time_col argument must be provided.")
        return
    if not args.coords_cols:
        print("coords_cols argument must be provided.")
        return
    if not args.type_col:
        print("type_col argument must be provided.")
        return   
    if args.uniform_areas and type(args.spatial_unit_areas) != int:
        print("uniform_areas set to True but spatial_unit_areas is not an integer. Set uniform_areas = False or provide an integer.")
        return
    if not args.uniform_areas and type(args.spatial_unit_areas) == int:
        print("uniform_areas set to False but spatial_unit_areas is an integer. Set uniform_areas = True or provider an array of integers.")
        return
    
    file = preprocess(args.input_file,args.output_file,  args.id_col, args.time_col, args.coords_cols, args.feature_cols,
                      args.type_col, args.date_format, args.uniform_areas, args.spatial_unit_areas)
    return file.export()
    
if __name__ == "__main__":
    main()



>>>>>>> a83aa26439585374f000de4f40575c539f600084
