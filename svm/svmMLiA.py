from numpy import *
import random
import matplotlib.pyplot as plt


def loadDataSet(filename):
	dataMat = []
	labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat, labelMat


def selectJrand(i, m):
	j = i
	while j == i:
		j = int(random.uniform(0, m))
		return j


def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj


def smoSimple(filename, C =0.6 , toler=0.001, maxIter=50):
	dataMat, labelMat = loadDataSet(filename)
	dataMatrix = mat(dataMat)
	labelMatrix = mat(labelMat).transpose()
	b = 0
	(m, n) = shape(dataMatrix)

	alphas = mat(zeros((m, 1)))
	iter = 0

	while iter < maxIter:
		alphaPairsChanged = 0
		for i in range(m):
			fxi = float(multiply(alphas,labelMatrix).T*(dataMatrix*dataMatrix[i,:].T)) + b
			Ei = fxi - float(labelMatrix[i])
			if ((labelMatrix[i]*Ei > toler) and (alphas[i] > 0)) or \
					((labelMatrix[i]*Ei < -toler) and (alphas[i] < C)):
				j = selectJrand(i, m)
				fxj = float((multiply(alphas, labelMatrix).T)* \
							(dataMatrix * dataMatrix[j, :].T)) + b
				Ej = fxj - float(labelMatrix[j])
				alphasIold = alphas[i].copy()
				alphasJold = alphas[j].copy()
				if (labelMatrix[i] != labelMatrix[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0, alphas[j] + alphas[i] - C)
					H = min(C, alphas[j] + alphas[i])

				if L == H: print "L == H"; continue
				eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
					  dataMatrix[i, :] * dataMatrix[i, :].T - \
					  dataMatrix[j, :] * dataMatrix[j, :].T
				if eta >= 0: print "eta >= 0"; continue
				alphas[j] -= alphasIold + labelMatrix[j] * (Ei - Ej) / eta
				alphas[j] = clipAlpha(alphas[j], H, L)
				if (abs(alphas[j] - alphasIold) < 0.00001):
					print "j not moving enough"
					continue
				alphas[i] += labelMatrix[i] * labelMatrix[j]*(alphasJold - alphas[j])
				b1 = -Ei - labelMatrix[i] * (dataMatrix[i, :] * dataMatrix[i, :].T) * \
						   (alphas[i] - alphasIold) - labelMatrix[j] * (dataMatrix[i, :] * \
																		dataMatrix[j, :].T) * (
														  alphas[j] - alphasJold) + b
				b2 = -Ej - labelMatrix[i] * (dataMatrix[i, :] * dataMatrix[j, :].T) * \
						   (alphas[i] - alphasIold) - labelMatrix[j] * (dataMatrix[j, :] * \
																		dataMatrix[j, :].T) * (
														  alphas[j] - alphasJold) + b
				if (alphas[i] > 0) and (alphas[i] < C):
					b = b1
				elif (alphas[j] > 0) and (alphas[j] < C):
					b = b2
				else:
					b = (b1 + b2) / 2.0
				alphaPairsChanged += 1
				print "iter: %d i:%d, pairs changed %d" % \
					  (iter, i, alphaPairsChanged)
		if alphaPairsChanged == 0:
			iter += 1
		else:
			iter = 0
		print "iteration number: %d" % iter
	plotBestFit(dataMat, labelMat,b, alphas)
	return b, alphas

def plotBestFit(dataMat, labelMat,b, alphas):
	dataArr = array(dataMat)
	dataMatrix = mat(dataMat)
	labelMatrix = mat(labelMat).transpose()
	n = shape(dataArr)[0]
	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,0])
			ycord1.append(dataArr[i,1])
		else:
			xcord2.append(dataArr[i,0])
			ycord2.append(dataArr[i,1])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker ='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	x = arange(-3.0,3.0,0.1)
	y = []
	for j in x:
		y.append(float ((-multiply(alphas,labelMatrix).T*dataMatrix[:,0]*j-b) / (multiply(alphas,labelMatrix).T*dataMatrix[:,1]) ))
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.xlabel('X2')
	plt.show()
	plt.savefig('temp.png')
	return