import numpy as np

from Main.NaiveBayes.Polynomial import Polynomial
from Main.NaiveBayes.Bernoulli import Bernoulli
from test.NaiveBayes.EmailTest.Function import *

hampath = '../email/ham/'
spampath = '../email/spam/'
labels = []
docList = []

loadData(hampath, 0, docList, labels)
loadData(spampath, 1, docList, labels)

vocList = createVocList(docList)
input = word2vec(docList, vocList)

bayes = Polynomial()
phix, phiy = bayes.train(input, labels)
pred = bayes.predict(input)
print(pred)
print(labels)
