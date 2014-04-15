from __future__ import division
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.networks.network.Network import FeedForwardNetwork
from pybrain.structure.modules.neuronlayer.NeuronLayer import LinearLayer, SigmoidLayer
from pybrain.structure.modules.module.Module import addInputModule, addModule, addOutputModule, addConnection, sortModules
from pybrain.structure import FullConnection

def convertDataNeuralNetwork(x, y):
	data = SupervisedDataSet(x.shape[1], 1)
	for xIns, yIns in zip(x, y):
    	data.addSample(xIns, yIns)    
	return data

def NN(xTrain, yTrain, xTest, yTest):
	trainData = convertDataNeuralNetwork(xTrain, yTrain)
	testData = convertDataNeuralNetwork(xTest, yTest)
	fnn = FeedForwardNetwork()
	inLayer = SigmoidLayer(trainData.indim)
	hiddenLayer = SigmoidLayer(5)
	outLayer = LinearLayer(trainData.outdim)
	fnn.addInputModule(inLayer)
	fnn.addModule(hiddenLayer)
	fnn.addOutputModule(outLayer)
	in_to_hidden = FullConnection(inLayer, hiddenLayer)
	hidden_to_out = FullConnection(hiddenLayer, outLayer)
	fnn.addConnection(in_to_hidden)
	fnn.addConnection(hidden_to_out)
	fnn.sortModules()
	trainer = BackpropTrainer(fnn, dataset = trainData, momentum = 0.1, verbose = True, weightdecay = 0.01)

	for i in xrange(10):
	    trainer.trainEpochs(500)
	    
	rmse = percentError(trainer.testOnClassData(dataset = testData), yTest)
	return rmse/100

def main():
	rmse = NN(xTrain, yTrain, xTest, yTest)
	print rmse

if __name__=="__main__":
	main()