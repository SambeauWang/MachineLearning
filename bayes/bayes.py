from numpy import *


def spamTest():
	docList = []
	classList = []
	fullText = []
	for i in range(1, 26):
		wordList = textParse(open("email/spam/%d.txt" % i).read())
		docList.append(wordList)
		fullText.append(wordList)
		classList.append(1)
		wordList = textParse(open("email/ham/%d.txt" % i).read())
		docList.append(wordList)
		fullText.append(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = range(50)
	testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del (trainingSet[randIndex])
	trainMat = []
	trainClasses = []
	for i in trainingSet:
		trainMat.append(setOfwords2Vec(vocabList, docList[i]))
		trainClasses.append(classList[i])
	p0V, p1V, pSam = trainNB0(array(trainMat), array(trainClasses))
	# print p0V, p1V, pSam
	errorCount = 0
	for i in testSet:
		wordVector = setOfwords2Vec(vocabList, docList[i])
		if classifyNB(wordVector, p0V, p1V, pSam) != classList[i]:
			errorCount += 1
			print docList[i]
	print "the error rate is ", float(errorCount) / len(testSet)


def createVocabList(dataSet):
	vocabSet = set([])
	for data in dataSet:
		vocabSet = vocabSet | set(data)
	return list(vocabSet)


def setOfwords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for data in inputSet:
		if data in vocabList:
			returnVec[inputSet.index(data)] += 1
		# else:
		# 	print "The word %s is not in vocabList." %data
	return returnVec


def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory) / float(numTrainDocs)
	p0num = ones(numWords)
	p1num = ones(numWords)
	p0denom = 2.0
	p1denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 0:
			p0num = p0num + trainMatrix[i]
			p0denom = p0denom + sum(trainMatrix[i])
		else:
			p1num = p1num + trainMatrix[i]
			p1denom = p1denom + sum(trainMatrix[i])
	p0Vect = log(p0num / p0denom)
	p1Vect = log(p1num / p1denom)
	return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vect, p1Vect, pAbusive):
	p1 = sum(vec2Classify * p1Vect) + log(pAbusive)
	p0 = sum(vec2Classify * p0Vect) + log(1.0 - pAbusive)
	if p1 > p0:
		return 1
	else:
		return 0


def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]
