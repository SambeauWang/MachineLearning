from math import log
import operator

def createData():
    data = [[1,1,'yes'], [1,1,'yes'],[1,0,'no'], [0,1,'no'], [0,1,'no']]
    labels = ['no surfacing','flippers']
    return data,labels

def calcShannonEnt(dataSet):
    shannonEnt = 0.0
    num = len(dataSet)
    keyOfLabel = {}
    for data in dataSet:
        label = data[-1]
        if label not in keyOfLabel:
            keyOfLabel[label] = 0
        keyOfLabel[label] += 1
    for key in keyOfLabel.keys():
        p = float(keyOfLabel[key])/num
        shannonEnt -= p * log(p,2)
    return shannonEnt

def splitDataset(dataSet, axis, value):
    retDataset = []
    for data in dataSet:
        if data[axis] == value:
            reduceDataset = data[:axis]
            reduceDataset.extend(data[axis+1:])
            retDataset.append(reduceDataset)
    return retDataset

def choosebestFeatureToSplit(dataSet):
    bestFeature = -1
    baseEnt = calcShannonEnt(dataSet)
    numFeature = len(dataSet[0]) - 1
    infoGain = 0.0
    num = len(dataSet)
    for i in range(numFeature):
        currentEnt = 0.0
        temp = [example[i] for example in dataSet]
        # temp = []
        # for example in dataSet:
        #     temp.append(example[i])
        uniqueVals = set(list(temp))

        for val in uniqueVals:
            temp = splitDataset(dataSet, i, val)
            p = len(temp)/float(num)
            currentEnt += p * calcShannonEnt(temp)
        if baseEnt - currentEnt > infoGain:
            infoGain = baseEnt - currentEnt
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for key in classList:
        if key not in classCount.keys(): classCount[key] = 0
        classCount[key] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,label):
    dataList = [example[-1] for example in dataSet]
    if dataList.count(dataList[0]) == len(dataList):
        return dataList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(dataList)
    bestFeat = choosebestFeatureToSplit(dataSet)
    bestFeatlabel = label[bestFeat]
    myTree = {bestFeatlabel:{}}
    del(label[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for key in uniqueVals:
        subLabels = label[:]
        temp = splitDataset(dataSet,bestFeat, key)
        myTree[bestFeatlabel][key] = createTree(temp,subLabels)
    return myTree

def classifyTest():
    dataSet,label = createData()
    return createTree(dataSet,label)