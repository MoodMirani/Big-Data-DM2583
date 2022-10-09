from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # finds out the weight of the words, how determining the word is
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline # used to organise the flow from the vectorizer and MNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split
from Cleanup import cleanup


def Create_Training_Set(choice):
    if choice == "load_from_local":
        Training_Set = pd.read_csv("project/data/Cleaned_Training_Data.csv", names=["sentiment, tweet"])
        Training_Set = Training_Set[1:] # remove first row
        
    elif choice == "extract_new":
    # Gather data
        Training_Set = pd.read_csv("project/data/training.csv", names=["sentiment", "ID", "date", "query", "user", "tweet"])
        Training_Set = Training_Set.drop([ "ID", "date", "query", "user"], axis=1)
        Training_Set_Negative = Training_Set[1:7000] # Extract 7000 negative tweets
        Training_Set_Positive = Training_Set[900000:907000] # Extract 7000 positive tweets
        Training_Set = pd.concat([Training_Set_Negative, Training_Set_Positive]) # merge them into one file
    return Training_Set

def Save_To_File(filename, data):
    filepath = Path('project/data/' + filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    data.to_csv(filepath)
    print("Saved the tweets to the file: " + filename)

categories = ['positive', 'negative']
trainingData = Create_Training_Set()
trainingData.tweet = cleanup(trainingData.tweet)

X_train, X_test, y_train, y_test = train_test_split(trainingData.tweet, trainingData.sentiment, test_size=0.30, random_state=80)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

vectorizer = TfidfVectorizer()
# X_train = 
# y_train = 

model.fit(X_train, y_train)
labels = model.predict(["This is a very good wine"])

print(labels)

