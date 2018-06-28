
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

    def export(self, colnames, filename):
        exp = open(filename, "w")
        exp.write(colnames)
        exp.write("\n")
        for row in range(0, len(self.predictions)):
            for col in range(0, len(self.predictions[0])):
                exp.write(self.predictions[row, col])
                exp.write(",")
            exp.write("\n")
        exp.close()