from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import operator

def RandomForest(xTrain, yTrain, xTest, yTest, treeNum):
	rms = dict()
	for trees in treeNum:
		rf = RandomForestRegressor(n_estimators = trees)
		rf.fit(xTrain, yTrain)
		yPred = rf.predict(xTest)
		rms[trees] = sqrt(mean_squared_error(yTest, yPred))

	(bestRegressor, rmse) = sorted(rms.iteritems(), key = operator.itemgetter(1))[0]

	return bestRegressor, rmse

def AdaBoost(xTrain, yTrain, xTest, yTest, treeNum):
	rms = dict()
	for trees in treeNum:
		ab = AdaBoostRegressor(n_estimators = trees)
		ab.fit(xTrain, yTrain)
		yPred = ab.predict(xTest)
		rms[trees] = sqrt(mean_squared_error(yTest, yPred))

	(bestRegressor, rmse) = sorted(rms.iteritems(), key = operator.itemgetter(1))[0]

	return bestRegressor, rmse

def main():
	
	treeNum = list(xrange(1,502,10))
	bestRegressor, rmse = RandomForest(xTrain, yTrain, xTest, yTest, treeNum)
	print bestRegressor, rmse

	bestRegressor, rmse = AdaBoost(xTrain, yTrain, xTest, yTest, treeNum)
	print bestRegressor, rmse

if __name__=="__main__":
	main()
