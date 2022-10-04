from xml.sax.handler import all_features
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from pathlib import Path

Twitter_Data = pd.read_csv("project/Twitter_Data.csv", names=["tweet", "sentiment"])
Twitter_Data = Twitter_Data[1:] # remove the first line (previous header)

print(Twitter_Data.shape)

def cleaning(data):
    cleaned_data = []
    counter=0
    print("size of sentiment columns: " + str(len(data)))
    for line in data:
        # TODO Remove the sentiment of deleted rows
        counter = counter + 1
        try:
            line = re.sub("@", "", line) # remove @
            line = re.sub("#", "", line) # remove #
            # remove emojis: regex command from: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
            line = re.sub(re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])'), "", line) # empojis
            line = re.sub(re.compile('<.*?>'), "", line) # remove html tags 
            line = re.sub('[^a-zA-Z\s]', '', line) # remove punctuations
            line = re.sub(r'\s+', ' ', line, flags=re.I) # remove unnecessary spaces
            # todo spellcheck()
            line = line.lower() # transform letters to lowercase
            cleaned_data.append(line)
        except:
            print("Deleting tweet nr: " + str(counter))
            cleaned_data.append("THIS TWEET WAS DELETED DURING CLEANUP")       
            
    print("number of added rows: " + str(counter))
    return cleaned_data

def clean_sentiments(sentiments):
    cleaned_sentiments = []
    counter = 0
    for row in sentiments:
        counter = counter + 1
        try:
            row = re.sub("@", "", row) # remove @
            cleaned_sentiments.append(row)

        except ValueError:
            print("Catched Value Error in sentiment at row: " + str(counter))
            cleaned_sentiments.append("0")

        except TypeError:
            print("Catched Type Error in sentiment at row: " + str(counter))
            cleaned_sentiments.append("0")
    return cleaned_sentiments

Twitter_Data.tweet = cleaning(Twitter_Data.tweet)
Twitter_Data.sentiment = clean_sentiments(Twitter_Data.sentiment)



filepath = Path('project/Cleaned_Twitter_Data.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)  
Twitter_Data.to_csv(filepath)


# vectorizing
vectorizer = CountVectorizer(stop_words="english")
all_features = vectorizer.fit_transform(Twitter_Data.tweet)


X_train, X_test, y_train, y_test = train_test_split(all_features, Twitter_Data.sentiment, test_size=0.30, random_state=42)

# Classify
classifier = svm.SVC() # the kernel: Gaussian Radial basis function
classifier.fit(X_train, y_train)


# performance measure
print(classifier.score(X_test, y_test)) # test score
ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, normalize="true") 
RocCurveDisplay.from_estimator(classifier, X_test, y_test)
plt.show()