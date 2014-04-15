from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import operator

def SVMReg(xTrain, yTrain, xTest, yTest, marginList):
	rms = dict()
	for margin in marginList:
		svm = SVR(C = margin)
		svm.fit(xTrain, yTrain)
		yPred = svm.predict(xTest)
		rms[margin] = sqrt(mean_squared_error(yTest, yPred))

	(bestClassifier, rmse) = sorted(rms.iteritems(), key = operator.itemgetter(1))[0]
	return bestClassifier, rmse


def main():
	marginList = [10**x for x in xrange(-10,10,1)]
	bestClassifier, rmse = SVMReg(xTrain, yTrain, xTest, yTest, marginList)
	print bestClassifier, rmse

if __name__=="__main__":
	main()