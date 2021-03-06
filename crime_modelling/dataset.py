<<<<<<< HEAD
import numpy as np
import pickle

#don't assume type_col included in feature_cols 
"""
This file contains the class for a SpatialDataSet object. The inputs are as follows:
    
out_file (required)- The name of the output file. The output file will be an .obj, to be managed with the 
pickle package.
id_col (required)- The column in the dataframe that contains the ids of each data entry.
time_col (required)- Timestamp column
coords_cols (required)- Two columns (x,y) that contains the coordinates of each data entry.
feature_cols - Variable number of columns that contain any features for each data entry. 
Each feature is a column.
type_col (required) - The column that determines the crime type of each data entry. 
date_format - String format of time_col
uniform_areas - Whether the grid cells of the data points are uniform area.
spatial_unit_areas - The area of the grid cells. If uniform_areas = True, then it will be an integer.
Otherwise, it will be an array of integers.
"""

class SpatialDataSet(object):
    def __init__(self, out_file, id_col, time_col, coords_cols, feature_cols, type_col, date_format = "%Y/%d/%m",
                 uniform_areas = True, spatial_unit_areas = 250000):
        self.features = feature_cols #will probably change later
        (self.counts, self.ID, self.coords) = self.make_feature_frame(id_col, time_col, type_col, coords_cols)
        (self.start_date, self.end_date) = self.get_dates(time_col) 
        self.period = date_format
        self.view_frame = self.view(id_col, time_col, coords_cols, type_col, feature_cols) 
        self.uniform_areas = uniform_areas
        self.spatial_unit_areas = spatial_unit_areas
        self.out_file = out_file
    
    def make_feature_frame(self, id_col, time_col, type_col, coords_cols): #output is 3-d numpy array
        id_dict = dict() 
        id_row = 0
        all_weeks = np.unique(time_col)
        unique_id = np.unique(id_col) #spatial IDs
        unique_types = np.unique(type_col)
        counts_frame = np.zeros((len(unique_id), len(all_weeks), len(unique_types) + 2), dtype = object)
        coords = np.zeros((len(unique_id), 2), dtype = float)
        for i in range(0, len(unique_id)):
            type_counts = np.where(id_col == unique_id[i])[0] #get row indices for id
            sub_type = type_col[type_counts,:].astype(object) #gets crime type rows matching certain id
            sub_time = time_col[type_counts,:].astype(object) #gets week rows matching ID
            coords[i,:] = coords_cols[type_counts,:][0]
            for j in range(0, len(all_weeks)): #getting counts by type for each week
                unique, counts_temp = np.unique(sub_type[np.where(sub_time == all_weeks[j])[0],:], return_counts=True)
                if len(unique) == 0: 
                    counts_frame[i,j,:] = [unique_id[i], all_weeks[j]] + ([0] * len(unique_types))
                else:
                    counts = [0] * len(unique_types)
                    for k in range(0, len(unique_types)): #saving unique counts to frame
                        if unique_types[k] not in unique:
                            counts[k] = 0
                        else: counts[k] = np.asscalar(counts_temp[np.where(unique == unique_types[k])])
                    counts_frame[i,j,:] = [unique_id[i], all_weeks[j]]  + counts    
            id_dict[unique_id[i]] = i #frame index : spatial ID
            id_row += 1
        return (counts_frame, id_dict, coords)

    def get_dates(self, time_col): #getting start, end dates
        time_col = time_col.astype("datetime64")
        time = np.sort(time_col)
        return(time[0, 0], time[len(time) - 1, 0])

    def view(self, id_cols, time_cols, coord_cols, type_col, feature_cols): #view in presentable format
        if feature_cols != "":
            whole_frame = np.concatenate((id_cols.astype(object), time_cols.astype(object), coord_cols.astype(object),
                type_col.astype(object), feature_cols), axis = 1)
        else:
             whole_frame = np.concatenate((id_cols.astype(object), time_cols.astype(object), coord_cols.astype(object),
                type_col.astype(object)), axis = 1)
        return whole_frame

    def export(self): #save to csv file
        name = self.out_file + str(".obj")
        file = open(name, "wb")
        pickle.dump(self, file)
        file.close()

=======
import numpy as np
import pickle

#don't assume type_col included in feature_cols 
"""
This file contains the class for a SpatialDataSet object. The inputs are as follows:
    
out_file (required)- The name of the output file. The output file will be an .obj, to be managed with the 
pickle package.
id_col (required)- The column in the dataframe that contains the ids of each data entry.
time_col (required)- Timestamp column
coords_cols (required)- Two columns (x,y) that contains the coordinates of each data entry.
feature_cols - Variable number of columns that contain any features for each data entry. 
Each feature is a column.
type_col (required) - The column that determines the crime type of each data entry. 
date_format - String format of time_col
uniform_areas - Whether the grid cells of the data points are uniform area.
spatial_unit_areas - The area of the grid cells. If uniform_areas = True, then it will be an integer.
Otherwise, it will be an array of integers.
"""

class SpatialDataSet(object):
    def __init__(self, out_file, id_col, time_col, coords_cols, feature_cols, type_col, date_format = "%Y/%d/%m",
                 uniform_areas = True, spatial_unit_areas = 250000):
        self.features = feature_cols #will probably change later
        (self.counts, self.ID, self.coords) = self.make_feature_frame(id_col, time_col, type_col, coords_cols)
        (self.start_date, self.end_date) = self.get_dates(time_col) 
        self.period = date_format
        self.view_frame = self.view(id_col, time_col, coords_cols, type_col, feature_cols) 
        self.uniform_areas = uniform_areas
        self.spatial_unit_areas = spatial_unit_areas
        self.out_file = out_file
    
    def make_feature_frame(self, id_col, time_col, type_col, coords_cols): #output is 3-d numpy array
        id_dict = dict() 
        id_row = 0
        all_weeks = np.unique(time_col)
        unique_id = np.unique(id_col) #spatial IDs
        unique_types = np.unique(type_col)
        counts_frame = np.zeros((len(unique_id), len(all_weeks), len(unique_types) + 2), dtype = object)
        coords = np.zeros((len(unique_id), 2), dtype = float)
        for i in range(0, len(unique_id)):
            type_counts = np.where(id_col == unique_id[i])[0] #get row indices for id
            sub_type = type_col[type_counts,:].astype(object) #gets crime type rows matching certain id
            sub_time = time_col[type_counts,:].astype(object) #gets week rows matching ID
            coords[i,:] = coords_cols[type_counts,:][0]
            for j in range(0, len(all_weeks)): #getting counts by type for each week
                unique, counts_temp = np.unique(sub_type[np.where(sub_time == all_weeks[j])[0],:], return_counts=True)
                if len(unique) == 0: 
                    counts_frame[i,j,:] = [unique_id[i], all_weeks[j]] + ([0] * len(unique_types))
                else:
                    counts = [0] * len(unique_types)
                    for k in range(0, len(unique_types)): #saving unique counts to frame
                        if unique_types[k] not in unique:
                            counts[k] = 0
                        else: counts[k] = np.asscalar(counts_temp[np.where(unique == unique_types[k])])
                    counts_frame[i,j,:] = [unique_id[i], all_weeks[j]]  + counts    
            id_dict[unique_id[i]] = i #frame index : spatial ID
            id_row += 1
        return (counts_frame, id_dict, coords)

    def get_dates(self, time_col): #getting start, end dates
        time_col = time_col.astype("datetime64")
        time = np.sort(time_col)
        return(time[0, 0], time[len(time) - 1, 0])

    def view(self, id_cols, time_cols, coord_cols, type_col, feature_cols): #view in presentable format
        if feature_cols != "":
            whole_frame = np.concatenate((id_cols.astype(object), time_cols.astype(object), coord_cols.astype(object),
                type_col.astype(object), feature_cols), axis = 1)
        else:
             whole_frame = np.concatenate((id_cols.astype(object), time_cols.astype(object), coord_cols.astype(object),
                type_col.astype(object)), axis = 1)
        return whole_frame

    def export(self): #save to csv file
        name = self.out_file + str(".obj")
        file = open(name, "wb")
        pickle.dump(self, file)
        file.close()

>>>>>>> a83aa26439585374f000de4f40575c539f600084
