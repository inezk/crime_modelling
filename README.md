# crime_modelling
CONTENTS OF THIS README
  Pipeline Details
  Repo Hierarchy
  Required Packages

PIPELINE DETAILS

There are 3 components to this pipline, each of which has a corresponding python file which takes command line inputs.

1) Preprocessing the data: 

preprocessing.py - Takes in a .csv file and outputs an .obj file that can be managed using the pickle package in python. The .obj file contains a SpatialDataSet object containing the data points in the .csv file in terms of counts. The SpatialDataSet contains the following components:
    - .counts: A 3 -D numpy dataframe with dimensions (# spatial IDs, # unique timestamps, #crime types + 2), where the 3rd dimension        is [spatial ID, timstamp, crime type #1, crime type# 2, ..., crime type #n].
    - .ID: A dictionary that contains entries {index in .counts frame: spatial ID in dataframe}.
    - .coords: A dataframe with dimensions (#spatial IDs, 2) that holds the coordinates for each spatial ID. The columns are (x,y).
    - .start_date: The earliest timestamp in the data.
    - .end_date: The latest timestamp in the data.
    - .period: String format of the timestamps.
    - .view_frame: Viewing all of the column inputs together in one concatenated frame.
    - .uniform_areas: A boolean determining if the grid cells corresponding to the coordinates are uniform.
    - .spatial_unit_areas: If uniform_areas = True, then it is an integer determining the area of the grid cells. If uniform_areas = 
     False, then it is an array of integers containing the area of all grid cells.
    - .out_file: The name of the exported .obj file.

Example command usage:

python3 preprocessing.py  -input_file aprs_5years.csv -id_col 'Grid500ID' -time_col 'DATE_START' -coords_cols 'Grid500X', 'Grid500Y' -type_col 'HIERARCHY' -uniform_areas True -spatial_unit_areas 250000

Input explanations:

input_file (required) - The name of the .csv file to be inputted.
out_file (required)- The name of the output file. The output file will be an .obj, to be managed with the 
pickle package.
id_col (required)- The column in the dataframe that contains the spatial ids of each data entry.
time_col (required)- Timestamp column
coords_cols (required)- Two columns (x,y) that contains the coordinates of each data entry.
feature_cols - Variable number of columns that contain any features for each data entry. 
Each feature is a column.
type_col (required) - The column that determines the crime type of each data entry. 
date_format - String format of time_col
uniform_areas - Whether the grid cells of the data points are uniform area.
spatial_unit_areas - The area of the grid cells. If uniform_areas = True, then it will be an integer.
Otherwise, it will be an array of integers.

2) Prediction:

predictor.py - Takes in an .obj file that was outputted from preprocessing.py and exports a .csv of predictions. Currently there are 4 models to choose from with the keywords in () to call in the command line:
  - Moving Average (mov_avg)
  - Kernel Density Estimate (kde) - Building off of scikit-learn's KernelDensity function. For further information, look at: 
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
  - Logistic Regression (log_reg) - Building off of scikit-learn's LogisticRegression function. For further information, look at:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  - Hawkes Process as per Mohler et al. [2011] (mohler)
The output is a .csv with the columns [Prediction Method, Week, Spatial ID, Actual Counts, Prediction]

Example command usage:

python3 predictor.py -input_file sample_small.obj -output_file results.csv -time_window 817 -start_index 817 -end_index 818 mohler -fixed_bandwidth True -bandwidth 1

skeleton command usage: python3 predictoy.py -input_file -output_file [universal args] [model name] [model specific args]

Input explanations:

- Universal inputs regardless of model
    -input_file (required): .obj file containing SpatialDataSet.
    -output_file (required): name of .csv file to be outputted
    -start_index: Index by week # (ie 0th index corresponds to the 0th week in the data) to start making predictions on. (Default = 104)
    -end_index: Index by week # to top making predictions on.
    -time_window: Time in weeks to predict over for each week, ie # of weeks to having in moving window. (Default = 260)
    -crime_types: Array of integers indicating which crime type indices to include in model. (Default = [] [all crime_types])

- Moving Average has no additional inputs
- KDE specific inputs:
    -bandwidth: Bandwidth to apply on KDE model. (Default = 500)
- Logistic Regression inputs:
    -penalty: Define which regularization type, if any, to use. (Default = 'l2')
    -reg_constant: Regularization constant to use. (Default = 1.0)
    -lag_window: This particular logistic regression model compounds the weeks such that instead of one row corresponding to one week 
     of data, one row instead corresponds to law_window weeks of data. The counts are converted to boolean counts, 0 indicating no 
     crime and 1 indicating at least one. The columns then become the features of the model. (Default = 8)
- Hawkes Process inputs:
    -fixed_bandwidth: Boolean indicating if fixed or variable bandwidth KDE will be used. (Default = True)
    -bandwidth: If fixed_bandwidth = True, then the bandwidth supplied to the KDE. (Default = 500)
    -u_k: If fixed_bandwidth = False, then it represents Du_k, the u_kth nearest neighbor distance to data point i. A column of              coefficients used when calculating u(x,y) and g(t,x,y) in the model of form v(t)u(x,y) sum_(t_k<t)(t - t_k, x - x_k, y - y_k). 
     (Default = 15)
    -v_k: If fixed_bandwidth = False, then it represents Dv_k, the v_kth nearest neighbor distance to data point i. A column of              coefficients used when calculating v(t) in the model of form v(t)u(x,y) + sum_(t_k<t)(t - t_k, x - x_k, y - y_k). (Default = 100)
    
3) Evaluating the data:

eval.py - Takes in the output from predictor.py and outputs a tradeoff curve of: Average % of crimes predicted vs. % of area exposed to prevention, with % exposed fixed at each prediction period.

Example command usage:

python3 eval.py -input_file predictions.csv -spatial_unit_area 250000 -output_dir graphics

Input explanations:

-input_file (required): Name of .csv file containing predictions (outputted from predictor.py)
-output_dir: Directory for saving evaluation graphics.
-no_header: Flag indicating no header row in the input file.
-spatial_unit_area: Area of spatial units (e.g., for uniform grid 
-area_file: Name of file containing areas of spatial units.
-histogram: Produces histogram showing freq. of spatial units in top 1 percent of area exposed.

REPO HIERARCHY

/crime_modelling: containing all code files for data pipeline
  -dataset.py: file containing SpatialDataSet class
  -preprocessing.py: preprocessing module that takes in command line inputs and calls dataset.py
  -predictor_class.py: file containing parent Predictor class for all models. Imports classes folder for predictor.py
  -predictor.py: prediction module that takes in command line inputs 
  -eval.py: evaluation module that takes in command line inputs
  /classes: containing all code files for models
    -__init__.py: file exporting all model files
    -moving_average.py: file containing Moving Average model class
    -kde.py: file containing KDE model class
    -logistic_regression.py: Logistic Regression model class
    -hawkes_process.py: Hawkes Process model class
/data: scripts and .csv files for data collection/prediction
  -experiments.py: Runs all experiments set up in training.yaml. Example command usage: python3 experiments.py
  -training.yaml: File detailing experiments (can be customized)
   Input arguments:
   
   input_file: (SpatialDataSet object to take in)
   
   experiment_key (name must contain model keyword, which are the same as the command line inputs for the model name in predictor.py ex.      Moving Average's is mov_avg):
      -command_line inputs
      (example)
      -crime_types: [2,3,4,5]
  -aprs_5years.csv: Pittsburgh crime data for the past 5 years.
  -processed_data.obj: SpatialDataSet object of aprs_5years.csv
  -sample_data.py: inhomogeneous Poisson process to produce simulated data.
      function is generate_points(highest x coordinate, highest y coordinate, max time, output file)
      command line usage: python3 sample_data.py
  -sample_test_small.csv: Toy data set output from sample_data.py
  -sample_small.obj: SpatialDataSet object for sample_test_small.csv
  
REQUIRED PACKAGES

/crime_modelling
  argparse
  numpy 
  pandas
  pickle
  scikit-learn
  scipy
/data
  yaml
  
  
