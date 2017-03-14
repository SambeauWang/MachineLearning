from numpy import *


def adaClassify(dataToClass):
	dataMat, classLabels = loadSimpleData()
	weakClassArr = adaBoostTrainDs(dataMat, classLabels)
	dataMatrix = mat(dataToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m, 1)))
	for i in range(len(weakClassArr)):
		aggClassEst += weakClassArr[i]['alpha'] * stumpClassify(dataMatrix, weakClassArr[i]['dim'],
																weakClassArr[i]['thresh'], weakClassArr[i]['ineq'])
	return sign(aggClassEst)


def loadSimpleData():
	dataMat = matrix([
		[1., 2.1],
		[2., 1.1],
		[1.3, 1.],
		[1. , 1.],
		[2., 1.]
	])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	retArray = ones((shape(dataMatrix)[0], 1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:, dimen] > threshVal] = -1.0
	return retArray


def buildStump(dataArr, classLabels, D):
	dataMat = mat(dataArr)
	classMat = mat(classLabels).T
	m, n = shape(dataMat)
	numSteps = 10.0
	bestStump = {}
	bestClassEst = mat(zeros((m, 1)))
	minError = inf
	for i in range(n):
		rangeMin = dataMat[:, i].min()
		rangeMax = dataMat[:, i].max()
		stepSizes = (rangeMax - rangeMin) / numSteps
		for j in range(-1, int(numSteps) + 1):
			threshVal = rangeMin + float(j)* stepSizes
			for inequal in ['lt', 'gt']:
				predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
				errArr = mat(ones((m, 1)))
				errArr[predictedVals == classMat] = 0
				weightedError = D.T * errArr
				if weightedError < minError:
					minError = weightedError
					# print minError
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
					bestClassEst = predictedVals.copy()
	return bestStump, minError, bestClassEst


def adaBoostTrainDs(dataArr, classLabels, numIt=30):
	weakClassArr = []
	m = shape(dataArr)[0]
	D = mat(ones((m, 1)) / m)
	aggClassEst = mat(zeros((m, 1)))
	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataArr, classLabels, D)
		alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		# print alpha
		expon = multiply((-1 * alpha) * mat(classLabels).T, classEst)
		# print expon
		D = multiply(D, exp(expon))
		# print "D",D
		D = D / D.sum()
		# print D
		aggClassEst += alpha * classEst
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
		errorRate = float(aggErrors.sum() / m)
		# print "the errorRate : %f" %errorRate
		if errorRate == 0.0:
			break
	return weakClassArr
