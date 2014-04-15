from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import operator

def LinReg(xTrain, yTrain, xTest, yTest):
	reg = linear_model.LinearRegression()
	reg.fit(xTrain, yTrain)
	yPred = reg.predict(xTest)
	rms = sqrt(mean_squared_error(yTest, yPred))
	return rms

def LinRegRidge(xTrain, yTrain, xTest, yTest, alphaList):
	rms = dict()
	for alpha in alphas:
		reg = linear_model.Ridge(alphas = alpha, store_cv_values = True)
		reg.fit(xTrain, yTrain)
		yPred = reg.predict(xTest)
		rms[alpha] = sqrt(mean_squared_error(yTest, yPred))

	(bestClassifier, rmse) = sorted(rms.iteritems(), key = operator.itemgetter(1))[0]
	return bestClassifier, rmse

def LinRegLasso(xTrain, yTrain, xTest, yTest, alphaList):
	rms = dict()
	for alpha in alphas:
		reg = linear_model.Lasso(alphas = alphaList)
		reg.fit(xTrain, yTrain)
		yPred = reg.predict(xTest)
		rms[alpha] = sqrt(mean_squared_error(yTest, yPred))

	(bestClassifier, rmse) = sorted(rms.iteritems(), key = operator.itemgetter(1))[0]
	return bestClassifier, rmse

def main():

	rmse = LinReg(xTrain, yTrain, xTest, yTest)
	print rmse
	
	alphaList = [10**x for x in xrange(-10,10,1)]
	bestAlpha, rmse = LinRegRidge(xTrain, yTrain, xTest, yTest, alphaList)
	print bestAlpha, rmse

	alphaList = [10**x for x in xrange(-30,1,1)]
	bestAlpha, rmse = LinRegLasso(xTrain, yTrain, xTest, yTest, alphaList)
	print bestAlpha, rmse

if __name__=="__main__":
	main()
