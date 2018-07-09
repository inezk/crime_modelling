import argparse
import pickle
from classes import *

def main():
    parser = argparse.ArgumentParser(
            description = "Mellon Grant prediction module\n\n" + 
            "This module is used to train and predict various models on crime. It is designed to " 
            + "work with an object of type SpatialDataSet. If the input is not such a type, please "+
            "run preprocessing.py first to obtain a SpatialDataSet object. "
            "The following are" + 
            " the prediction methods provided in this packages, with the command to call them in parenthesis:\n\n"+
            "Moving average (mov_avg)\n\n" + "Kernel Density Estimation (kde)\n\n" +
            "Logistic Regression (log_reg)\n\n")
    
    required = parser.add_argument_group('required arguments')
    required.add_argument("-input_file", help = "file containing SpatialDataSet")
    required.add_argument("-output_file", help = "name of output file")
    
    subparsers = parser.add_subparsers(help = "Prediction method", dest = "models")
    
    #moving avg args
    parser_mov_avg = subparsers.add_parser("mov_avg", help = "Moving average prediction model")
    parser_mov_avg.add_argument("-time_window", help = "Time in weeks to average counts over",
                                default = 52)
    parser_mov_avg.add_argument("-crime_types", help = "Array of indices indicating which crime types"+
                                "to average over", default = [])
    
    #kde args
    parser_kde = subparsers.add_parser("kde", help = "KDE prediction model")
    parser.add_argument("-time_window", help = "Time in weeks to predict over for each point",
                            default = 52)
    parser_kde.add_argument("-crime_types", help = "Array of indices indicating which crime types " + 
                            "to include in model", default = [])
    parser_kde.add_argument("-bandwidth", help = "bandwidth for kde model", default = 500)
    
    #log_reg args
    parser_log_reg = subparsers.add_parser("log_reg", help = "Log Regression prediction model")
    parser_log_reg.add_argument("-time_window", help = "Time in weeks to predict over for each point",
                                default = 52)
    parser_log_reg.add_argument("-crime_types", help = "Array of indices indicating which crime types "+
                                "to include in model", default = [])
    parser_log_reg.add_argument("-penalty", help ="Define which regularization type, if any, to use",
                                default = 'l2')
    parser_log_reg.add_argument("-reg_constant", help = "Regularization constant to use", default = 1.0)
    parser_log_reg.add_argument("-lag_window", help = "# of weeks to combine as 1 feature set", default = 8)
    
    #general arguments
    parser.add_argument("start_index", help = "index to start evaluation from, corresponding in weeks",
                        default = 104)
    parser.add_argument("end_index", help = "index to end evaluation, corresponding in weeks",
                        default = 260)
    ###############parse args and check for required args###########################
    args = parser.parse_args()
    if not args.input_file:
        print("input-file argument must be provided.")
        return
    
    if args.models == "mov_avg" or args.models == "kde" or args.models == "log_reg":
        if type(args.crime_types) != list:
            print("crime_types must be provided as an array of ints")
            return
    
    #########################parsing args for to input into model#######################  
    file = open(args.input_file, "rb")
    dataset = pickle.load(file)
    file.close()
    if args.models == "mov_avg":
        mov_avg = moving_average.moving_average(dataset, args.time_window, args.crime_types)
        mov_avg.export(filename = args.output_file)
    elif args.models == "kde":
        kde = kde.KDE(dataset, args.time_window, args.crime_types, args.bandwidth, args.start_index,
                      args.end_index)
        kde.export(filename = args.output_file)
    elif args.models == "log_reg":
        log_reg = logistic_regression.logistic_regression(dataset, args.penalty, args.reg_constant,
                                                          args.crime_types, args.lag_window,
                                                          args.start_index, args.end_index, 
                                                          args.time_window)
        log_reg.export(filename = args.output_file)
    
if __name__ == "__main__":
    main()  
            
    
    