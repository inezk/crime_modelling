
### Evaluating computational experiments for predicting crime hotspots ### 

# This script produces a prediction tradeoff curve 
# for the Mellon Grant predictive policing project.

# The input file should contain actual crime counts
# and a prediction for every time period and spatial unit
# combination. 

# The script produces and saves a prediction tradeoff curves,
# showing % of crimes predicted vs. % of area exposed to prevention,
# with % exposed fixed at each prediction period.
import matplotlib
matplotlib.use('Agg')

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import math
from itertools import cycle

def main():

    # Read arguments from command line

    parser = argparse.ArgumentParser(
            description="Mellon Grant evaluation module\n\n" +
            "Input file should be comma-delimited with the following " + 
            "columns:\n\n" + 
            "method,date,spatial_id,actual_count,prediction\n\n"+
            "  method - Prediction method identifier included for comparison " +
            "of predictions from multiple methods. If evaluating a single " +
            "prediction method, column should contain a single unique " + 
            "value.\n\n" + 
            "  date - Date of prediction. Dates should be in ISO 8601 format: " +
            "YYYY-MM-DD. For weekly/monthly predictions, date should " +
            "indicate the first day of each time period over which " +
            "prediction is made (e.g., '2016-01-01' for January 2016).\n\n" +
            "  spatial_id - Unique identifier for spatial units " + 
            "(e.g., block ID or grid cell ID).\n\n" +
            "  actual_count - Actual count of part 1 violent crimes " +
            "for each time period and spatial unit.\n\n" + 
            "  prediction - Prediction field used to rank spatial units " +
            "within each time period (e.g., predicted number of crimes). " +
            "Spatial units ranked in descending order.", 
            formatter_class=argparse.RawTextHelpFormatter)

    required = parser.add_argument_group('required arguments')
    required.add_argument('-input_file', help="Name of file containing " +
                                              "predictions")
    parser.add_argument('-output_file', help="Name of tradeoff curve " +
                                "output file", default='crimes predicted vs. area exposed.png')
    parser.add_argument('-no_header', action='store_true', help="Flag " +
                                "indicating no header row in the input file.")
    parser.add_argument('-spatial_unit_area', help="Area of spatial units " +
                                "(e.g., for uniform grid cells).")
    parser.add_argument('-area_file', help="Name of file containing " +
                                "areas of spatial units.")
    parser.add_argument('-histogram', action='store_true', help="Produces histogram showing " +
                                "freq. of spatial units in top 1 percent of area exposed.")
    
    # Parse arguments and check for required args.
    args = parser.parse_args()
    if not args.input_file:
        print("input-file argument must be provided.")
        return

    if not (args.spatial_unit_area or args.area_file):
        print("Either spatial_unit_area or area_file argument " + \
                                "must be provided.")
        return

    if args.spatial_unit_area and args.area_file:
        print("Cannot provide both spatial_unit_area and " + \
                                "area_file arguments.")
        return

    # Read input file into dataframe 
    if args.no_header:
        df = pd.read_csv(args.input_file, header=None)
    else:
        df = pd.read_csv(args.input_file)

    # Check for correct number of fields
    cols = ['method', 'date', 'spatial_id', 'actual_count', 'prediction']
    if len(df.columns) != len(cols):
        print("Incorrect number of columns in input file.")
        print(df.columns)
        return

    df.columns = cols

    # Create area column
    if args.spatial_unit_area:
        df['area'] = float(args.spatial_unit_area)
    else:
        df_area = pd.read_csv(args.area_file)
        df = pd.merge(df, df_area, on='spatial_id', how='left')
        if df['area'].isnull().sum() > 0:
            print("Some spatial ids not included in area file.")
            return
        df['area']=df['area'].astype(float)


    # Order rows by prediction field
    df = df.sort_values(['prediction'], ascending=False)

    # Set resolution for area bins in evaluation graphics
    area_resolution = 0.001
    num_digits = int(round(-math.log(area_resolution, 10)))


    # Create complete dataframe with all method, date, and area-resolution combinations
    methods = pd.DataFrame({'method': df['method'].unique()})
    dates = pd.DataFrame({'date': df['date'].unique()})
    bins = pd.DataFrame({'area_bin': np.arange(0, 1+area_resolution, area_resolution)})
    methods['key'] = 0
    dates['key'] = 0
    bins['key'] = 0
    df_complete = pd.merge(methods, dates, on='key')
    df_complete = pd.merge(df_complete, bins, on='key')
    df_complete.drop('key',1, inplace=True)
    df_complete['area_join'] = (df_complete['area_bin']*math.pow(10,num_digits)).round().astype(int)


    ## Calculate percent of crimes predicted versus area exposed for each method-date combination. ##

    # Group dataframe by method and date (grouped variables denoted with "MD")
    dfg = df.groupby(by=['method','date'])

    # Calculate num_predicted_MD field as cumulative sum of crimes predicted 
    # (moving from high-priority to low-priority sectors) for each date

    df['num_predicted_MD']= dfg['actual_count'].cumsum()

    # Calculate area_exposed field as cumulative sum of sector areas 
    # (moving from high-priority to low-priority sectors) divided by total area.
    area_cumsum = dfg['area'].cumsum()
    area_total = dfg['area'].transform(sum)
    df['area_exposed_MD'] = area_cumsum / area_total

    # Bin areas according to resolution parameter
    df['area_bin_MD']=(df['area_exposed_MD']+(area_resolution/2)).round(num_digits)

    # For each method, calculate average percent predicted for every area bin
    # Find max percent predicted for each bin by date and method
    df_binned = df.groupby(by=['method','date','area_bin_MD'])['num_predicted_MD'].max().reset_index()
    df_binned['area_join'] = (df_binned['area_bin_MD']*math.pow(10,num_digits)).round().astype(int)

    # Join bins to complete dataframe and forward fill any empty bins
    df_binned = pd.merge(df_complete, df_binned, on=['method', 'area_join', 'date'],how='left')
    df_binned['num_predicted_MD'] = df_binned.groupby(by=['method', 'date'])['num_predicted_MD'].fillna(method='ffill').fillna(0)

    # Sum across time periods for each bin
    df_total = df_binned.groupby(by=['method','area_bin']).sum().reset_index()

    df_total['percent_predicted'] = df_total['num_predicted_MD'] / df_total.groupby('method')['num_predicted_MD'].transform(max)


    ## Plot average percent of target crimes predicted against percent of total area exposed ##

    fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)

    for key, grp in df_total.groupby('method'):
        x = pd.Series(0).append(grp['area_bin'])
        y = pd.Series(0).append(grp['percent_predicted'])
        plt.plot(x,y, label=key, ls=next(linecycler))

    ax = plt.gca()
    ax.set_xlim([0,0.05])
    ax.set_ylim([0,0.5])
    ax.set_xlabel('Percent of area exposed to prevention', fontsize=18)
    ax.set_ylabel('Percent of crimes predicted', fontsize=18)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.1f}%'.format(x*100) for x in vals], fontsize=12)
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:3.1f}%'.format(x*100) for x in vals], fontsize=12)
    #plt.axvline(x=.02, color='0.75')
    plt.minorticks_on()
    plt.grid()

    plt.legend(loc='best')
    fig_name = args.output_file
    plt.savefig(fig_name)

    if args.histogram:
        df['exposed'] = df['area_exposed_MD'].apply(lambda x: 1 if x <= 0.01 else 0)
        for method in df['method'].unique():
            df_method = df[df['method']==method]
            exposure_freq = df_method.groupby('spatial_id')['exposed'].sum().values
            fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            n, bins, patches = plt.hist(exposure_freq, 156, normed=1, facecolor='green', alpha=0.75)
            plt.axis([0, 160, 0, 0.03])
            ax = plt.gca()
            ax.set_xlabel('Number of weeks in top 1%', fontsize=18)
            ax.set_ylabel('Percent of spatial units', fontsize=18)
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals], fontsize=12)
            plt.savefig(method + '_spatial_unit_histogram.png')

if __name__ == "__main__":
    main()
