import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


Twitter_Data = pd.read_csv("project/Twitter_Data.csv", names=["tweet", "sentiment"])
Twitter_Data = Twitter_Data[1:] # remove the first line (previous header)


def cleaning(data):
    cleaned_data = []
    counter=0
    for line in data:
        try:
            counter = counter + 1
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
        except TypeError:
            cleaned_data.append("THIS TWEET WAS DELETED DURING CLEANUP")
    return cleaned_data

Twitter_Data.tweet = cleaning(Twitter_Data.tweet)

X_train, X_test, y_train, y_test = train_test_split(Twitter_Data, Twitter_Data.sentiment, test_size=0.30, random_state=42)