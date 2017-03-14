from numpy import *
import operator

def dataClassify():
    hoRatio = 0.1
    dataSet,labelSet = file2matrix('datingTestSet2.txt')
    dataSet = autoNorm(dataSet)
    m = shape(dataSet)[0]
    numTest = int( m * hoRatio)
    errorCount = 0.0
    for i in range(numTest):
        classiferResult = classify0(dataSet[i,:],dataSet[numTest:m,:],labelSet[numTest:m],3)
        if (classiferResult != labelSet[i]):
            errorCount += 1.0
    print "the error rate is %f" %(errorCount / numTest)

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    temp = zeros(shape(dataSet))
    ranges = maxVals - minVals
    m = shape(dataSet)[0]
    temp = dataSet - tile(minVals,(m,1))
    temp = temp / tile(ranges , (m,1))
    return temp

def showData():
    import matplotlib.pyplot as plt
    dataSet,labelSet = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:,1],dataSet[:,2],15.0*array(labelSet),15.0*array(labelSet))

def file2matrix(filename):
    fr = open(filename)
    arrayline = fr.readlines()
    m = len(arrayline)
    dataSet = zeros( (m,3) )
    labelSet = []
    index = 0
    for line in arrayline:
        line = line.strip().split('\t')
        dataSet[index,:] = line[0:3]
        labelSet.append(int(line[-1]))
        index += 1
    return dataSet,labelSet

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k = 3):
    # k = 3
    # dataSet, labels = createDataSet()
    m = dataSet.shape[0]
    diffMat = tile(inX, (m, 1)) - dataSet
    diffMat = diffMat ** 2
    sqDistances = diffMat.sum(axis=1)
    distance = sqDistances ** 0.5
    distance_index = distance.argsort()
    classCount = {}
    for i in range(k):
        label = labels[distance_index[i]]
        classCount[label] = classCount.get(label, 0) + 1
    sorted_label = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_label[0][0]
