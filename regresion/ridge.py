from numpy import *

def ridgeRegres(xMat,yMat,lam = 0.2):
	xTx = xMat.T * xMat
	denom = xTx + eye(shape(xMat)[1]) * lam
	if linalg.det(denom) == 0.0:
		print 'error'
		return
	ws = denom.I * (xMat.T * yMat)
	return ws

def ridgeTest(filename):
	xArr,yArr = loadDataSet(filename)
	xMat  = mat(xArr)
	yMat = mat(yArr).T
	xMean = mean(xMat,0)
	xMat = (xMat - xMean) / var(xMat,0)
	yMean = mean(yMat,0)
	yMat = yMat - yMean
	num = 30
	wMat = zeros((num, shape(xMat)[1]))
	for i in range(num):
		ws  = ridgeRegres(xMat, yMat, exp(i-10) )
		wMat[i,:] = ws.T
	return wMat

def loadDataSet(filename):
	num = len(open(filename).readline().split('\t')) - 1
	dataMat = []
	labelMat = []
	file = open(filename)
	for line in file.readlines():
		temp_line = line.split('\t')
		temp_data = []
		for i in range(num):
			temp_data.append(float(temp_line[i]))
		dataMat.append(temp_data)
		labelMat.append(float(temp_line[-1]))
	return dataMat,labelMat
