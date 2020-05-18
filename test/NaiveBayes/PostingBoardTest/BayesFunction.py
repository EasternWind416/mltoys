

def loadDataSet():
    """
    自建数据集
    :return: 单词列表postingList，所属类别classVec
    """

    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please', 'my'],  # [0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocList(dataSet):
    vocSet = set([])
    for data in dataSet:
        vocSet = vocSet | set(data)
    return list(vocSet)


def words2vec(vocList, inputSet):
    retVec = [0] * len(vocList)

    for word in inputSet:
        if word in vocList:
            retVec[vocList.index(word)] = 1
        else:
            print('\nthe word {} is not in voc.\n'.format(word))
    return retVec