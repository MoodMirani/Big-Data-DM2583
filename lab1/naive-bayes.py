from numpy import vectorize
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt

# read all data
trainData = pd.read_csv("train.csv", header=None, names=["sentiment", "data"])
testData = pd.read_csv("test.csv", header=None, names=["sentiment", "data"])
evaluationData = pd.read_csv("evaluation.csv", header=None, names=["sentiment", "data"])

"""
We need to clean the data from:
HTML tags, hashtags, mentions (@), emojis

"""
# remove headers
trainData = trainData[1:]
testData = testData[1:]
evaluationData = evaluationData[1:]

def preProcessing(data):
    processedData = []
    for line in data:
        line = re.sub("@", "", line) # remove @
        line = re.sub("#", "", line) # remove #
        # remove emojis: regex command from: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
        line = re.sub(re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])'), "", line)
        line = re.sub(re.compile('<.*?>'), "", line) # remove html tags 
        processedData.append(line)
    return processedData

#clean
trainData.data = preProcessing(trainData.data)
testData.data = preProcessing(testData.data)
evaluationData.data = preProcessing(evaluationData.data)


# feature extraction
vectorizer = CountVectorizer(stop_words="english")
trainText = vectorizer.fit_transform(trainData.data)
testText = vectorizer.transform(testData.data)
evalText = vectorizer.transform(preProcessing(evaluationData.data))

# Classify
classifier = MultinomialNB()
classifier.fit(trainText, trainData.sentiment)

# performance measure
print(classifier.score(testText, testData.sentiment)) # test score
print(classifier.score(evalText, evaluationData.sentiment)) # evaluation score
matrix = plot_confusion_matrix(classifier, evalText, evaluationData.sentiment, normalize="true") 
plot_roc_curve(classifier, testText, testData.sentiment)
plt.show()


