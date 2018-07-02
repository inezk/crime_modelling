import numpy as np
import models as md
import argparse
import h5py

class Predictor(object):

    def __init__(self, data):
        self.SDS = data
        self.data = data.counts #after preprocessing for model specific data
        self.model = self.train
        self.predictions = self.predict

    def train(self, **kwargs): #returns a model
        return self.data

    def predict(self, **kwargs): #returns output predictions
        return self.data

    def export(self, filename = "predictions.csv", colnames = "Method, Week, Grid ID, Actual Counts, Predictions"):
        filename = open(filename, "wb")
        np.savetxt(filename, self.predictions, header = colnames, fmt = "%5s",delimiter = ",")

def main():
    parser = argparse.ArgumentParser(
            description = "Mellon Grant prediction module\n\n" + "The following are" + 
            " the prediction methods provided in this packages, with the command to call them in parenthesis:\n\n"+
            "Moving average (mov_avg)\n\n" + "Kernel Density Estimation (kde)\n\n" )
    
    parser = parser.add_argument("-input_file", help = "file containing SpatialDataSet")
    
    models = parser.add_subparsers(help = "Prediction method", dest = "models")
    
    #moving avg args
    models_mov_avg = models.add_parser("mov_avg", help = "Moving average prediction model")
    models_mov_avg.add_argument("-time_window", help = "Time in weeks to average counts over",
                                default = 52)
    models_mov_avg.add_argument("-crime_types", help = "Array of indices indicating which crime types"+
                                "to average over", default = "")
    
    #kde args
    models_kde = models.add_parser("kde", help = "KDE prediction model")
    models_kde.add_argument("-time_window", help = "Time in weeks to predict over for each point",
                            default = 52)
    models_kde.add_argument("-crime_types", help = "Array of indices indicating which crime types" + 
                            "to include in model", default = "")
    models_kde.add_argument("-bandwidth", help = "bandwidth for kde model", default = 500)
    
    ###############parse args and check for required args###########################
    args = parser.parse_args()
    if not args.input_file:
        print("input-file argument must be provided.")
        return
    
    if args.models == "mov_avg" or args.models == "kde":
        if type(args.crime_types) != "list":
            print("crime_types must be provided as an array of ints")
            return
    
    #########################parsing args for to input into model#######################
    #MISSING: NEED TO PARSE HDF5 into SDS
    hf = h5py.File(args.input_file, "r")
    dataset = hf.get("SDS")
    if args.models == "mov_avg":
        mov_avg = md.moving_average(dataset, args.time_window, args.crime_types)
        mov_avg.export(filename = "moving_average_predictions.csv")
    elif args.models == "kde":
        kde = md.KDE(dataset, args.time_window, args.crime_types, args.bandwidth)
        kde.export(filename = "kde_predictions.csv")
    
if __name__ == "__main__":
    main()  
            
    
    