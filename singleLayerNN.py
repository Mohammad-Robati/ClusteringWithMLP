import numpy as np


class SingleLayerNeuralNetwork:

    def __init__(self, data, learningRate, numberOfIterations):
        self.data = data
        self.learningRate = learningRate
        self.numberOfIterations = numberOfIterations

    def normalizeMatrices(self, dataSet):
        xArray = []
        yArray = []
        for data in dataSet:
            xArray.append([1.0, data[0], data[1]])
            yArray.append(data[2])
        x = np.mat(xArray)
        y = np.mat(yArray).transpose()
        return x, y

    def learn(self):
        trainingSet = self.data.trainingSet
        lr = self.learningRate
        x, y = self.normalizeMatrices(trainingSet)
        w = np.random.rand(np.shape(x)[1], 1)
        for i in range(0, self.numberOfIterations):
            w = w - lr * self.calculateDerivative(x, y, w)
        return w

    def calculateY(self, z):
        return 1.0 / (1 + np.exp(-z))

    def calculateDerivative(self, x, y, w):
        return x.transpose() * (self.calculateY(x * w) - y)

    def calculatePrecision(self, w, dataSet):
        true = 0
        x, y = self.normalizeMatrices(dataSet)
        results = x * w
        for i in range(0, len(results)):
            if (results[i] <= 0 and y[i] == 0) or (results[i] > 0 and y[i] == 1):
                true += 1
        return true / len(y) * 100

    def run(self):
        weights = self.learn()
        precision = self.calculatePrecision(weights, self.data.testSet)
        return weights, precision

