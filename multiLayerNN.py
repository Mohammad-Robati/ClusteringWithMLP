import numpy as np


class MultiLayerNeuralNetwork:

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
        x = np.array(xArray)
        y = np.array([yArray]).T
        return x, y

    def learn(self):
        trainingSet = self.data.trainingSet
        lr = self.learningRate
        x, y = self.normalizeMatrices(trainingSet)
        np.random.seed(1)
        w1 = np.random.random((3, 2))
        w2 = np.random.random((2, 1))
        for i in range(0, self.numberOfIterations):
            input = x
            hidden = self.calculateSig(np.dot(input, w1))
            output = self.calculateSig(np.dot(hidden, w2))
            outputError = y - output
            outputGrad = outputError * self.calculateDerivative(output)
            hiddenError = outputGrad.dot(w2.T)
            hiddenGrad = hiddenError * self.calculateDerivative(hidden)
            w2 += lr * hidden.T.dot(outputGrad)
            w1 += lr * input.T.dot(hiddenGrad)
        return w1, w2

    def calculateSig(self, z):
        return 1.0 / (1 + np.exp(-z))

    def calculateDerivative(self, z):
        return z * (1-z)

    def calculatePrecision(self, weights, dataSet):
        true = 0
        x, y = self.normalizeMatrices(dataSet)
        w1, w2 = weights
        output = self.calculateSig(np.dot(self.calculateSig(np.dot(x, w1)), w2))
        predicted = self.calculateSig(output)
        predicted = predicted - predicted.mean()
        for i in range(0, len(predicted)):
            if (predicted[i] <= 0 and y[i] == 0) or (predicted[i] > 0 and y[i] == 1):
                true += 1
        return true / len(y) * 100

    def run(self):
        weights = self.learn()
        precision = self.calculatePrecision(weights, self.data.testSet)
        return weights, precision
