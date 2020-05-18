import os
import re
import numpy as np


def loadData(root, label, docList, labels):
    dirs = os.listdir(root)
    for dir in dirs:
        fileName = os.path.join(root, dir)
        with open(fileName, 'r') as file:
            wordList = re.split(r'\W*', file.read())
            docList.append(wordList)
            labels.append(label)


def createVocList(docList):
    vocList = set([])
    for doc in docList:
        vocList = vocList | set(doc)
    return list(vocList)


def word2vec(input, vocList):
    retVec = np.zeros((len(input), len(vocList)))
    for i in range(len(input)):
        for word in input[i]:
            if word in vocList:
                retVec[i, vocList.index(word)] += 1
            else:
                print('word {} is not in vocList'.format(word))
    return retVec