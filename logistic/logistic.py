from numpy import *
import random

def stocGradAscent():
	dataArr, labelArr = loadDataset('testSet.txt')
	dataArr = array(dataArr)
	m ,n = shape(dataArr)
	numIt = 150
	weights = ones(n)
	for j in range(numIt):
		dataIndex = range(m)
		for i in range(m):
			alpha = 3/(i+j+1.0) +0.01
			randIndex = int( random.uniform(0, len(dataIndex)) )
			h = sigmoid( sum(dataArr[ randIndex ] * weights ) )
			error = labelArr[ randIndex ] - h
			weights += alpha * error * dataArr[ randIndex ]
			del(dataIndex[randIndex])
	return weights

def plotbestFit():
	import matplotlib.pyplot as plt
	weights = stocGradAscent();
	dataArr, labelArr = loadDataset('testSet.txt')
	dataArr = array(dataArr)
	n = shape(dataArr)[0]
	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(n):
		if int(labelArr[i] )== 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30, c='red',marker = 's')
	ax.scatter(xcord2, ycord2,s=30,c='green')
	x = arange(-3.0,3.0,0.1)
	y = (-weights[0] - weights[1] * x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()
	plt.savefig('temp.png')


def loadDataset(filename):
	dataArr = []
	labelArr = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataArr.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelArr.append(int(lineArr[2]))
	return dataArr,labelArr

def sigmoid(z):
	return 1.0/(1+exp(-z))

def gradAscent():
	dataArr, labelArr = loadDataset('testSet.txt')
	dataMat = mat(dataArr)
	labelMat = mat(labelArr).T
	m,n = shape(dataMat)
	alpha = 0.001
	weights = ones((n,1))
	numIt = 500
	for i in range(numIt):
		h = sigmoid(dataMat * weights)
		error = labelMat -h
		weights += alpha * (dataMat.T * error)
	return weights