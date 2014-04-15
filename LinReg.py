from sklearn import linear_model
#from sklearn import datasets
import numpy as np

def LinReg(xTrain, yTrain, xTest, yTest):
	reg = linear_model.LinearRegression()
	reg.fit(xTrain, yTrain)
	accuracy = reg.score(xTest, yTest)
	return accuracy

def LinRegRidge(xTrain, yTrain, xTest, yTest, alphaList):
	reg = linear_model.RidgeCV(alphas = alphaList, store_cv_values = True)
	reg.fit(xTrain, yTrain)
	#print reg.get_params, reg.alpha_
	accuracy = reg.score(xTest, yTest)
	return reg.alpha_, accuracy

def LinRegLasso(xTrain, yTrain, xTest, yTest, alphaList):
	reg = linear_model.LassoCV(alphas = alphaList)
	reg.fit(xTrain, yTrain)
	#print reg.get_params, reg.alpha_
	accuracy = reg.score(xTest, yTest)
	return reg.alpha_, accuracy

def main():
	'''diabetes = datasets.load_diabetes()
	diabetes_X = diabetes.data[:, np.newaxis]
	diabetes_X_temp = diabetes_X[:, :, 2]
	xTrain = diabetes_X_temp[:-20]
	xTest = diabetes_X_temp[-20:]
	yTrain = diabetes.target[:-20]
	yTest = diabetes.target[-20:]'''
	
	accuracy = LinReg(xTrain, yTrain, xTest, yTest)
	print accuracy
	
	alphaList = [10**x for x in xrange(-10,10,1)]
	bestAlpha, accuracy = LinRegRidge(xTrain, yTrain, xTest, yTest, alphaList)
	print bestAlpha, accuracy

	alphaList = [10**x for x in xrange(-30,1,1)]
	bestAlpha, accuracy = LinRegLasso(xTrain, yTrain, xTest, yTest, alphaList)
	print bestAlpha, accuracy

if __name__=="__main__":
	main()
