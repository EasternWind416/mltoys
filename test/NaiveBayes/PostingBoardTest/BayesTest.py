import numpy as np

from test.NaiveBayes.PostingBoardTest.BayesFunction import *
from ml.NaiveBayes import Polynomial


postingList, classVec = loadDataSet()
vocList = createVocList(postingList)

x = []
# for i in range(len(classVec)):
#     x.append(words2vec(vocList, postingList[i]))
# x = np.array(x)

# bayes = Bernoulli()
#
# phix, phiy = bayes.train(x, classVec)
#
# testEntry = ['stupid', 'love', 'garbage']
# testDoc = np.array(words2vec(vocList, testEntry))
#
# pred = bayes.predict(testDoc)
#
# if pred == 1:
#     print('this is abusive phase!')
# else:
#     print('this is not abusive phase.')
bayes = Polynomial()
for i in range(len(classVec)):
    x.append(words2vec(vocList, postingList[i]))
x = np.array(x)

phix, phiy = bayes.train(x, classVec)
testEntry = ['dog', 'love', 'my', 'love']
testDoc = np.array(words2vec(vocList, testEntry))

pred = bayes.predict(testDoc)

if pred == 1:
    print('this is abusive phase!')
else:
    print('this is not abusive phase.')