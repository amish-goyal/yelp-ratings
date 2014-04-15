from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import operator

def RandomForest(xTrain, yTrain, xTest, yTest, treeNum):
	rms = dict()
	for trees in treeNum:
		rf = RandomForestClassifier(n_estimators = trees)
		rf.fit(xTrain, yTrain)
		yPred = rf.predict(xTest)
		rms[trees] = sqrt(mean_squared_error(yTest, yPred))

	(bestClassifier, rmse) = sorted(rms.iteritems(), key = operator.itemgetter(1))[0]

	return bestClassifier, rmse

def AdaBoost(xTrain, yTrain, xTest, yTest, treeNum):
	rms = dict()
	for trees in treeNum:
		ab = AdaBoostClassifier(n_estimators = trees)
		ab.fit(xTrain, yTrain)
		yPred = ab.predict(xTest)
		rms[trees] = sqrt(mean_squared_error(yTest, yPred))

	(bestClassifier, rmse) = sorted(rms.iteritems(), key = operator.itemgetter(1))[0]

	return bestClassifier, rmse

def main():
	
	treeNum = list(xrange(1,502,10))
	bestClassifier, rmse = RandomForest(xTrain, yTrain, xTest, yTest, treeNum)
	print bestClassifier, rmse

	bestClassifier, rmse = AdaBoost(xTrain, yTrain, xTest, yTest, treeNum)
	print bestClassifier, rmse

if __name__=="__main__":
	main()
