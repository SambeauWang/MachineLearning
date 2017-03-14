from numpy import *
import matplotlib.pyplot as  plt

def draw_result():
	yHat,dataMat,labelMat = lwlrTest('ex0.txt')
	dataMat = mat(dataMat)
	labelMat = mat(labelMat).T
	strInd = dataMat[:,1].argsort(0)
	xSort = dataMat[strInd][:,0,:]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataMat[:,1].flatten().A[0],labelMat[:,0].flatten().A[0],s=2,c = 'red')
	ax.plot(xSort[:,1],yHat[strInd])
	plt.show()
	plt.savefig('temp.png')

def lwlr(testPoint, xArr,yArr,k=0.01):
	dataMat = mat(xArr)
	labelMat = mat(yArr).T
	m = shape(dataMat)[0]
	weight = mat(eye(m))
	for i in range(m):
		temp = testPoint - dataMat[i,:]
		weight[i,i] = exp( (temp * temp.T) / ( -2 * k**2))
	xTx = dataMat.T * weight * dataMat
	if linalg.det(xTx) == 1.0:
		print "error"
		return
	ws = xTx.I * dataMat.T * weight * labelMat
	return testPoint * ws

def lwlrTest(filename):
	dataMat,labelMat = loadDataSet(filename)
	m = shape(dataMat)[0]
	yHat = zeros(m)
	for i in range(m):
		yHat[i] = lwlr(dataMat[i], dataMat, labelMat)
	return yHat,dataMat,labelMat

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