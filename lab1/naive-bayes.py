import pandas as pd
import re

# read all data
trainData = pd.read_csv("lab1/train.csv", header=None, names=["sentiment", "data"])
evaluationData = pd.read_csv("lab1/evaluation.csv", header=None, names=["sentiment", "data"])
testData = pd.read_csv("lab1/test.csv", header=None, names=["sentiment", "data"])

"""
We need to clean the data from:
HTML tags, hashtags, mentions (@), emojis

"""
# remove headers
trainData = trainData[1:]
evaluationData = evaluationData[1:]
testData = testData[1:]

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

# print(evaluationData[0:5])
print(preProcessing(testData.data)[127])
# print(testData)
