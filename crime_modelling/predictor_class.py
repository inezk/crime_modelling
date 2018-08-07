# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:00:22 2018

@author: inezk

Sets up Predictor class for predictive models. This class is a skeleton for all models.
"""
import numpy as np

class Predictor(object):

    def __init__(self, data, filename):
        self.SDS = data #SpatialDataSet object
        self.data = data.counts #after preprocessing for model specific data
        self.model = self.train
        self.predictions = self.predict
        self.outfile = filename

    def train(self, **kwargs): #returns a model
        return self.data

    def predict(self, **kwargs): #returns output predictions
        return self.data

    def export(self, colnames = "Method, Week, Grid ID, Actual Counts, Predictions"):
        filename = open(self.outfile, "wb")
        np.savetxt(filename, self.predictions, header = colnames, fmt = "%5s",delimiter = ",")