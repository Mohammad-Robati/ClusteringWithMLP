from plot import Plotter
from data import Data
from singleLayerNN import SingleLayerNeuralNetwork
from multiLayerNN import MultiLayerNeuralNetwork
data = Data('./dataset.csv')
plotter = Plotter(data)
slnn = SingleLayerNeuralNetwork(data, 0.01, 1000)
weightsSLNN, precisionSLNN = slnn.run()
mlnn = MultiLayerNeuralNetwork(data, 0.1, 10000)
weightsMLNN, precisionMLNN = mlnn.run()
print("\nSingle Layer Neural Net Precision:\t", precisionSLNN, "%")
print("Multi Layer Neural Net Precision: \t", precisionMLNN, "%")
plotter.plot(weightsSLNN, weightsMLNN)



