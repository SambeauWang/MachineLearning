from numpy import *
import matplotlib.pyplot as  plt

def stageWise(eps=0.01,numIt = 200):   #xArr, yArr, 
	xArr,yArr = loadDataSet('abalone.txt')
	xMat  = mat(xArr)
	yMat = mat(yArr).T
	yMean = mean(yMat,0)
	yMat = yMat - yMean
	xMat = regularize(xMat)
	m,n = shape(xMat)
	returnMat = zeros((numIt, n))
	ws = zeros((n,1))
	wsTest = ws.copy()
	wsMax = ws.copy()
	for i in range(numIt):
		lowerError = inf
		for j in range(n):
			for sign in [-1, 1]:
				wsTest = ws.copy()
				wsTest[j] += sign * eps
				yTest = xMat * wsTest
				rssE = rssError(yMat.A , yTest.A)
				if rssE <lowerError:
					lowerError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i , :] = ws.T
	return returnMat


def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def draw_result():
	ws,dataMat,labelMat = standRegres()
	xCopy = dataMat.copy()
	xCopy.sort(0)
	yHat = xCopy * ws
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataMat[:,1].flatten().A[0],labelMat[:,0].flatten().A[0])
	ax.plot(xCopy[:,1],yHat)
	plt.show()
	plt.savefig('temp.png')

def standRegres():
	xArr,yArr = loadDataSet('ex0.txt')
	dataMat = mat(xArr)
	labelMat = mat(yArr).T
	temp = dataMat.T * dataMat
	if linalg.det(temp) == 0.0:
		print "error"
		return
	ws = temp.I * dataMat.T * labelMat
	return ws,dataMat,labelMat

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