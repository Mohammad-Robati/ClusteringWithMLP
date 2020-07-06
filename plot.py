import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    def __init__(self, data):
        self.data = data

    def calculateSig(self, z):
        return 1.0 / (1 + np.exp(-z))

    def getDecisionBoundaryPoints(self, weights, minX, minY, maxX, maxY):
        w1, w2 = weights
        minBasis = minX
        maxBasis = maxX
        minWindow = minY
        maxWindow = maxY
        stepNumberBasis = 200
        stepSizeBasis = (maxBasis - minBasis) / stepNumberBasis
        stepNumberWindow = 200
        stepSizeWindow = (maxWindow - minWindow) / stepNumberWindow
        boundary = []
        for i in range(0, stepNumberBasis):
            for j in range(0, stepNumberWindow):
                x = [1, minBasis + i * stepSizeBasis, minWindow + j * stepSizeWindow]
                output = self.calculateSig(np.dot(self.calculateSig(np.dot(x, w1)), w2))
                if 0.48 < output < 0.52:
                    boundary.append([minBasis + i * stepSizeBasis, minWindow + j * stepSizeWindow])
        return boundary

    def plot(self, weightsSLNN, weightsMLNN):
        data = self.data
        trainingXG1, trainingYG1, trainingXG2, trainingYG2 = [], [], [], []
        testXG1, testYG1, testXG2, testYG2 = [], [], [], []
        for sample in data.trainingSet:
            if sample[2] == 1:
                trainingXG1.append(sample[0])
                trainingYG1.append(sample[1])
            else:
                trainingXG2.append(sample[0])
                trainingYG2.append(sample[1])
        for sample in data.testSet:
            if sample[2] == 1:
                testXG1.append(sample[0])
                testYG1.append(sample[1])
            else:
                testXG2.append(sample[0])
                testYG2.append(sample[1])
        minX = min(min(trainingXG1), min(trainingXG2), min(testXG1), min(testXG2))
        minY = min(min(trainingYG1), min(trainingYG2), min(testYG1), min(testYG2))
        maxX = max(max(trainingXG1), max(trainingXG2), max(testXG1), max(testXG2))
        maxY = max(max(trainingYG1), max(trainingYG2), max(testYG1), max(testYG2))
        g1 = (trainingXG1, trainingYG1)
        g2 = (trainingXG2, trainingYG2)
        g3 = (testXG1, testYG1)
        g4 = (testXG2, testYG2)
        data = (g1, g2, g3, g4)
        colors = ("red", "darkred", "blue", "darkblue")
        groups = ("Training G1", "Training G2", "Test G1", "Test G2")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for data, color, group in zip(data, colors, groups):
            x, y = data
            ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
        x1, y1 = [minX, maxX], [(weightsSLNN[0] + weightsSLNN[1] * minX) / -(weightsSLNN[2]), (weightsSLNN[0] + weightsSLNN[1] * maxX) / -(weightsSLNN[2])]
        y1 = [list(y1)[0].tolist()[0][0], list(y1)[1].tolist()[0][0]]
        boundary = np.array(self.getDecisionBoundaryPoints(weightsMLNN, minX, minY, maxX, maxY))
        ax.scatter(boundary[:, 0], boundary[:, 1], alpha=0.8, c='green', edgecolors='none', s=5)
        plt.plot(x1, y1, marker='o')
        plt.title('Training and Test Points')
        plt.legend(loc=2)
        plt.show()
