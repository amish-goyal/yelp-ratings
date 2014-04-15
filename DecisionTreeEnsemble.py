from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
#from sklearn import datasets
import numpy as np
import operator

def RandomForest(xTrain, yTrain, xTest, yTest, treeNum):
	accuracy = dict()
	for trees in treeNum:
		rf = RandomForestClassifier(n_estimators = trees)
		rf.fit(xTrain, yTrain)
		accuracy[trees] = rf.score(xTest, yTest)

	#print accuracy
	(bestClassifier, accuracy) = sorted(accuracy.iteritems(), key = operator.itemgetter(1), reverse = True)[0]

	return bestClassifier, accuracy

def AdaBoost(xTrain, yTrain, xTest, yTest, treeNum):
	accuracy = dict()
	for trees in treeNum:
		rf = AdaBoostClassifier(n_estimators = trees)
		rf.fit(xTrain, yTrain)
		accuracy[trees] = rf.score(xTest, yTest)

	#print accuracy
	(bestClassifier, accuracy) = sorted(accuracy.iteritems(), key = operator.itemgetter(1), reverse = True)[0]

	return bestClassifier, accuracy

def main():
	'''diabetes = datasets.load_diabetes()
	diabetes_X = diabetes.data[:, np.newaxis]
	diabetes_X_temp = diabetes_X[:, :, 2]
	xTrain = diabetes_X_temp[:-20]
	xTest = diabetes_X_temp[-20:]
	yTrain = diabetes.target[:-20]
	yTest = diabetes.target[-20:]'''

	treeNum = list(xrange(1,502,10))
	bestClassifier, accuracy = RandomForest(xTrain, yTrain, xTest, yTest, treeNum)
	print bestClassifier, accuracy

	bestClassifier, accuracy = AdaBoost(xTrain, yTrain, xTest, yTest, treeNum)
	print bestClassifier, accuracy

if __name__=="__main__":
	main()
