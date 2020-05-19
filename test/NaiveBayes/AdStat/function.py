import numpy as np

from Main.NaiveBayes.Polynomial import Polynomial


def bagOfWords2VecMN(vocList, inputSet):
    retVec = [0] * len(vocList)
    for word in inputSet:
        if word in vocList:
            retVec[vocList.index(word)] += 1
    return np.array(retVec)


def createVocList(dataSet):
    vocList = set([])
    for doc in dataSet:
        vocList = vocList | set(doc)
    return list(vocList)


def textParse(bigStr):
    import re
    listOfTokens = re.split(r'\W*', bigStr)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# RSS源分类器，高频词去除
def calcMostFreq(vocList, fullText):
    import operator
    freqDict = {}
    for token in vocList:
        freqDict[token] = fullText.count(token)
    sortedFreqDict = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreqDict[: 20]


def bayesTest(feed1, feed0):
    docList=[]
    classList=[]
    fullText=[]
    bayes = Polynomial()
    minLen=min(len(feed1['entries']),len(feed0['entries']))

    for i in range(minLen):
        # 每次访问一条RSS源
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocList=createVocList(docList)
    top30Words=calcMostFreq(vocList, fullText)

    # 去掉出现次数最高的词
    for pairW in top30Words:
        if pairW[0] in vocList:
            vocList.remove(pairW[0])

    trainingSet=[i for i in range(2 * minLen)]
    testSet=[]

    # 取20条作为测试
    for i in range(10):
        randIndex=int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    phix, phiy = bayes.train(np.array(trainMat), np.array(trainClasses))

    errorCount=0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocList,docList[docIndex])
        if bayes.predict(wordVector) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount) / len(testSet))
    return vocList, phix, phiy


def getTopWords(vocList, phix):
    topF0 = []
    topF1 = []
    for i in range(phix.shape[1]):
        if phix[0, i] > -6.:
            topF0.append((vocList[i], phix[0, i]))
        if phix[1, i] > -6.:
            topF1.append((vocList[i], phix[1, i]))

    sortedF0 = sorted(topF0, key=lambda pair: pair[1], reverse=True)
    print('---------Feed 0-------------')
    for i in sortedF0:
        print(i[0])
    sortedF1 = sorted(topF1, key=lambda pair: pair[1], reverse=True)
    print('---------Feed 1-------------')
    for i in sortedF1:
        print(i[0])
