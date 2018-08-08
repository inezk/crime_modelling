<<<<<<< HEAD
import argparse
import pickle
from classes import *

""""
Prediction stage of data pipeline.
Example command usage:
    
python3 predictor.py -input_file processed_data.obj -output_file results.csv mov_avg -time_window 52 -crime_types [] 
"""

def main():
    parser = argparse.ArgumentParser(
            description = "Mellon Grant prediction module\n\n" + 
            "This module is used to train and predict various models on crime. It is designed to " 
            + "work with an object of type SpatialDataSet. If the input is not such a type, please "+
            "run preprocessing.py first to obtain a SpatialDataSet object. "
            "The following are" + 
            " the prediction methods provided in this packages, with the command to call them in parenthesis:\n\n"+
            "Moving average (mov_avg)\n\n" + "Kernel Density Estimation (kde)\n\n" +
            "Logistic Regression (log_reg)\n\n" + "Hawkes Process as per Mohler et al. (2011) paper (mohler)\n\n")
    
    required = parser.add_argument_group('required arguments')
    required.add_argument("-input_file", help = "file containing SpatialDataSet")
    required.add_argument("-output_file", help = "name of output file")
    
    subparsers = parser.add_subparsers(help = "Prediction method", dest = "models")
    
    #moving avg args
    parser_mov_avg = subparsers.add_parser("mov_avg", help = "Moving average prediction model")
    
    #kde args
    parser_kde = subparsers.add_parser("kde", help = "KDE prediction model")
    parser_kde.add_argument("-bandwidth", help = "bandwidth for kde model", default = 500)
    
    #log_reg args
    parser_log_reg = subparsers.add_parser("log_reg", help = "Log Regression prediction model")
    parser_log_reg.add_argument("-penalty", help ="Define which regularization type, if any, to use",
                                default = 'l2')
    parser_log_reg.add_argument("-reg_constant", help = "Regularization constant to use", default = 1.0)
    parser_log_reg.add_argument("-lag_window", help = "# of weeks to combine as 1 feature set", default = 8)
    
    #hawkes process args
    parser_mohler = subparsers.add_parser("mohler", help = "Hawkes Process as per Mohler et al. (2011) paper")
    parser_mohler.add_argument("-fixed_bandwidth", help = "The kernel used is gaussian. Boolean field indicated whether to use fixed or variable bandwidth",
                               default = True)
    parser_mohler.add_argument("-bandwidth", help = "If fixed_bandwidth = True, then the bandwidth supplied to the KDE", default = 500)
    parser_mohler.add_argument("-u_k", help = "If fixed_bandwidth = False, then it represents Du_k, the u_kth nearest neighbor distance to data point i." +
                               "A column of coefficients used when calculating u(x,y) and g(t,x,y) in the model of form v(t)u(x,y) + sum_(t_k<t)(t - t_k, x - x_k, y - y_k)",
                               default = 15)
    parser_mohler.add_argument("-v_k", help = "If fixed_bandwidth = False, then it represents Dv_k, the v_kth nearest neighbor distance to data point i." +
                               "A column of coefficients used when calculating v(t) in the model of form v(t)u(x,y) + sum_(t_k<t)(t - t_k, x - x_k, y - y_k)",
                               default = 100)
    
    #general arguments
    parser.add_argument("-start_index", help = "index to start evaluation from, corresponding in weeks",
                        default = 104)
    parser.add_argument("-end_index", help = "index to end evaluation, corresponding in weeks",
                        default = 260)
    parser.add_argument("-time_window", help = "Time in weeks to predict over for each point",
                            default = 52)
    parser.add_argument("-crime_types", help = "Array of indices indicating which crime types " + 
                            "to include in model", default = [])
    ###############parse args and check for required args###########################
    args = parser.parse_args()
    if not args.input_file:
        print("input-file argument must be provided.")
        return
    
    if type(args.crime_types) != list:
        print("crime_types must be provided as an array of ints")
        return
    
    if args.models == "mohler":
        if args.fixed_bandwidth:
            if not args.bandwidth:
                print("Please supply a bandwidth or set fixed_bandwidth to False.")
                return
        else:
            if not args.u_k or args.v_k:
                print("If fixed_bandwidth is False, u_k and v_k must be supplied.")
                return

    #########################parsing args to input into model#######################  
    file = open(args.input_file, "rb")
    dataset = pickle.load(file)
    file.close()
    if args.models == "mov_avg":
        mov_avg = moving_average.moving_average(dataset, args.output_file, args.time_window, args.crime_types)
        mov_avg.export()
    elif args.models == "kde":
        kde = kde.KDE(dataset, args.output_file, args.time_window, args.crime_types, args.bandwidth, args.start_index,
                      args.end_index)
        kde.export()
    elif args.models == "log_reg":
        log_reg = logistic_regression.logistic_regression(dataset, args.output_file, args.penalty, args.reg_constant,
                                                          args.crime_types, args.lag_window,
                                                          args.start_index, args.end_index, 
                                                          args.time_window)
        log_reg.export()
    elif args.models == "mohler":
        mohler = hawkes_process.hawkes_process(dataset, args.output_file, args.time_window, args.fixed_bandwidth,
                                               args.bandwidth, args.u_k, args.v_k,
                                               args.start_index, args.end_index,
                                                args.crime_types)
        mohler.export()
    
if __name__ == "__main__":
    main()  
            
    
=======
import argparse
import pickle
from classes import *

""""
Prediction stage of data pipeline.
Example command usage:
    
python3 predictor.py -input_file processed_data.obj -output_file results.csv mov_avg -time_window 52 -crime_types [] 
"""

def main():
    parser = argparse.ArgumentParser(
            description = "Mellon Grant prediction module\n\n" + 
            "This module is used to train and predict various models on crime. It is designed to " 
            + "work with an object of type SpatialDataSet. If the input is not such a type, please "+
            "run preprocessing.py first to obtain a SpatialDataSet object. "
            "The following are" + 
            " the prediction methods provided in this packages, with the command to call them in parenthesis:\n\n"+
            "Moving average (mov_avg)\n\n" + "Kernel Density Estimation (kde)\n\n" +
            "Logistic Regression (log_reg)\n\n" + "Hawkes Process as per Mohler et al. (2011) paper (mohler)\n\n")
    
    required = parser.add_argument_group('required arguments')
    required.add_argument("-input_file", help = "file containing SpatialDataSet")
    required.add_argument("-output_file", help = "name of output file")
    
    subparsers = parser.add_subparsers(help = "Prediction method", dest = "models")
    
    #moving avg args
    parser_mov_avg = subparsers.add_parser("mov_avg", help = "Moving average prediction model")
    
    #kde args
    parser_kde = subparsers.add_parser("kde", help = "KDE prediction model")
    parser_kde.add_argument("-bandwidth", help = "bandwidth for kde model", default = 500)
    
    #log_reg args
    parser_log_reg = subparsers.add_parser("log_reg", help = "Log Regression prediction model")
    parser_log_reg.add_argument("-penalty", help ="Define which regularization type, if any, to use",
                                default = 'l2')
    parser_log_reg.add_argument("-reg_constant", help = "Regularization constant to use", default = 1.0)
    parser_log_reg.add_argument("-lag_window", help = "# of weeks to combine as 1 feature set", default = 8)
    
    #hawkes process args
    parser_mohler = subparsers.add_parser("mohler", help = "Hawkes Process as per Mohler et al. (2011) paper")
    parser_mohler.add_argument("-fixed_bandwidth", help = "The kernel used is gaussian. Boolean field indicated whether to use fixed or variable bandwidth",
                               default = True)
    parser_mohler.add_argument("-bandwidth", help = "If fixed_bandwidth = True, then the bandwidth supplied to the KDE", default = 500)
    parser_mohler.add_argument("-u_k", help = "If fixed_bandwidth = False, then it represents Du_k, the u_kth nearest neighbor distance to data point i." +
                               "A column of coefficients used when calculating u(x,y) and g(t,x,y) in the model of form v(t)u(x,y) + sum_(t_k<t)(t - t_k, x - x_k, y - y_k)",
                               default = 15)
    parser_mohler.add_argyment("-v_k", help = "If fixed_bandwidth = False, then it represents Dv_k, the v_kth nearest neighbor distance to data point i." +
                               "A column of coefficients used when calculating v(t) in the model of form v(t)u(x,y) + sum_(t_k<t)(t - t_k, x - x_k, y - y_k)",
                               default = 100)
    
    #general arguments
    parser.add_argument("-start_index", help = "index to start evaluation from, corresponding in weeks",
                        default = 104)
    parser.add_argument("-end_index", help = "index to end evaluation, corresponding in weeks",
                        default = 260)
    parser.add_argument("-time_window", help = "Time in weeks to predict over for each point",
                            default = 52)
    parser.add_argument("-crime_types", help = "Array of indices indicating which crime types " + 
                            "to include in model", default = [])
    ###############parse args and check for required args###########################
    args = parser.parse_args()
    if not args.input_file:
        print("input-file argument must be provided.")
        return
    
    if type(args.crime_types) != list:
        print("crime_types must be provided as an array of ints")
        return
    
    if args.models == "mohler":
        if args.fixed_bandwidth:
            if not args.bandwidth:
                print("Please supply a bandwidth or set fixed_bandwidth to False.")
                return
            if type(args.bandwidth) != int:
                print("Bandwidth must be an integer.")
                return
        else:
            if not args.u_k or args.v_k:
                print("If fixed_bandwidth is False, u_k and v_k must be supplied.")
                return
            if int(args.u_k) != int or int(args.v_k) != int:
                print("u_k and v_k must be integers.")
                return
    #########################parsing args to input into model#######################  
    file = open(args.input_file, "rb")
    dataset = pickle.load(file)
    file.close()
    if args.models == "mov_avg":
        mov_avg = moving_average.moving_average(dataset, args.output_file, args.time_window, args.crime_types)
        mov_avg.export()
    elif args.models == "kde":
        kde = kde.KDE(dataset, args.output_file, args.time_window, args.crime_types, args.bandwidth, args.start_index,
                      args.end_index)
        kde.export()
    elif args.models == "log_reg":
        log_reg = logistic_regression.logistic_regression(dataset, args.output_file, args.penalty, args.reg_constant,
                                                          args.crime_types, args.lag_window,
                                                          args.start_index, args.end_index, 
                                                          args.time_window)
        log_reg.export()
    elif args.models == "mohler":
        mohler = hawkes_process.hawkes_process(dataset, args.output_file, args.crime_types, args.time_window, 
                                               args.start_index, args.end_index, args.fixed_bandwidth, 
                                               args.bandwidth, args.v_k, args.u_k)
        mohler.export()
    
if __name__ == "__main__":
    main()  
            
    
>>>>>>> a83aa26439585374f000de4f40575c539f600084
    