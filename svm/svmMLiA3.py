from numpy import *
import matplotlib.pyplot as plt

def testRbf(k1=1.3):
	dataArr,labelArr = loadDataSet("testSetRBF.txt")
	b,alphas = smoP("testSetRBF.txt",200,0.0001,1000,('rbf',k1))
	datMat = mat(dataArr)
	labelMat = mat(labelArr).transpose()
	svInd = nonzero(alphas>0)[0]
	sVs = datMat[svInd]
	labelSV = labelMat[svInd]
	errorCount = 0
	m,n = shape(datMat)
	print "There is %d Support vectors." %shape(svInd)[0]
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',1))
		predict  = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict) != sign(labelArr[i]):
			errorCount += 1
	print "the training error rate is %f" % (float(errorCount)/m)

	dataArr,labelArr = loadDataSet("testSetRBF2.txt")
	datMat = mat(dataArr)
	labelMat = mat(labelArr).transpose()
	m,n = shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',1))
		predict  = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict) != sign(labelArr[i]):
			errorCount += 1
	print "the test error rate is %f" % (float(errorCount)/m)


def loadDataSet(filename):
	dataMat = []
	labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat, labelMat

def smoP(filename, C=0.6, toler=0.001, maxIter=50, kTup=('lin', 0)):
	dataMatln, classLabels = loadDataSet(filename)
	oS = optStruct(dataMatln, classLabels, C, toler ,kTup)
	iter = 0
	entireSet = True
	alphapairsChanged = 0
	while (iter < maxIter) and ((alphapairsChanged > 0) or entireSet):
		if entireSet:
			for i in range(oS.m):
				alphapairsChanged += innerL(i, oS)
			print "fullSet, iter:%d i:%d pairs changed %d" \
				  % (iter, i, alphapairsChanged)
			iter += 1
		else:
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			for i in nonBoundIs:
				alphapairsChanged += innerL(i, oS)
			print "non-bound, iter:%d i:%d pairs changed %d" \
				  % (iter, i, alphapairsChanged)
			iter += 1
		if entireSet:
			entireSet = False
		elif alphapairsChanged == 0:
			entireSet = True
		print "iteration number: %d" % iter
	# plotBestFit(dataMatln, classLabels,oS.b, oS.alphas)
	return oS.b, oS.alphas


def innerL(i, oS):
	Ei = calcEk(oS, i)
	if ((oS.labelMat[i] * Ei > oS.toler) and (oS.alphas[i] > 0)) or \
			((oS.labelMat[i] * Ei < - oS.toler) and (oS.alphas[i] < oS.C)):
		j, Ej = selectJ(i, oS, Ei)
		alphasIold = oS.alphas[i].copy()
		alphasJold = oS.alphas[j].copy()
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])

		if L == H:
			print "L == H"
			return 0
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
		if eta >= 0:
			print "eta >= 0"
			return 0
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
		updateEk(oS, j)

		if (abs(oS.alphas[j] - alphasIold) < 0.00001):
			print "j not moving enough"
			return 0
		oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * (alphasJold - oS.alphas[j])
		updateEk(oS,i)

		b1 = oS.b -Ei - oS.labelMat[i] * oS.K[i,i] * \
				   (oS.alphas[i] - alphasIold) - oS.labelMat[j] * oS.K[i,j] * (oS.alphas[j] - alphasJold)
		b2 = oS.b -Ej - oS.labelMat[i] * oS.K[i,j]* \
				   (oS.alphas[i] - alphasIold) - oS.labelMat[j] * oS.K[j,j] * (oS.alphas[j] - alphasJold)
		if (oS.alphas[i] > 0) and (oS.alphas[i] < oS.C):
			oS.b = b1
		elif (oS.alphas[j] > 0) and (oS.alphas[j] < oS.C):
			oS.b = b2
		else:
			oS.b = (b1 + b2) / 2.0
		return 1
	else:
		return 0

def kernelTrans(X,A,kTup):
	m,n = shape(X)
	k = mat(zeros((m,1)))
	if kTup[0] == 'lin':
		k = X*A.T
	elif kTup[0] == 'rbf':
		for i in range(m):
			deltaRow = X[i,:] - A
			k[i] = deltaRow * deltaRow.T
		k = exp(k / (-1*kTup[1]**2))
	else:
		raise NameError('Houston We Have a problem.')
	return k

class optStruct:
	def __init__(self, dataMatln, classLabels, C, toler,kTup):
		self.X = mat(dataMatln)
		self.labelMat = mat(classLabels).transpose()
		self.C = C
		self.toler = toler
		self.m = shape(dataMatln)[0]
		self.b = 0
		self.alphas = mat(zeros((self.m, 1)))
		self.eCache = mat(zeros((self.m, 2)))
		self.K = mat(zeros((self.m,self.m)))
		for i in range(self.m):
			self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)


def calcEk(oS, k):
	fx = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] +oS.b)
	ek = fx - float(oS.labelMat[k])
	return ek



def selectJrand(i, m):
	j = i
	while j == i:
		j = int(random.uniform(0, m))
		return j


def selectJ(i, oS, Ei):
	maxk = -1
	Ej = 0
	maxDeletaE = 0;
	oS.eCache[i] = [1, Ei]
	validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
	if len(validEcacheList) > 0:
		for j in validEcacheList:
			if i == j:
				continue
			Ek = calcEk(oS, j)
			deltaE = abs(Ek - Ei)
			if deltaE > maxDeletaE:
				maxDeletaE = deltaE
				maxk = j
				Ej = Ek
		return maxk, Ej

	else:
		j = selectJrand(i, oS.m)
		maxk = j
		Ej = calcEk(oS, j)
	return maxk, Ej


def updateEk(oS, k):
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1, Ek]


def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj

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