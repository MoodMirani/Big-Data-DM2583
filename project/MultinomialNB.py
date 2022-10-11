from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # finds out the weight of the words, how determining the word is
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline # used to organise the flow from the vectorizer and MNB
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from functions import *

def train_multinomial_naive_bayes(training_set):
    categories = ['positive', 'negative']
    X_train, X_test, y_train, y_test = train_test_split(training_set.tweet, training_set.sentiment, test_size=0.30, random_state=80)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    
    """
    y_test_predictions = model.predict(X_test)

     
    # performance measure
    print("MNB Classifier score: " + str(model.score(X_test, y_test)))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize="true") 
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.show()

    
    # creating confusion matrix and heat map
    mat = confusion_matrix(y_test, y_test_predictions)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=["positive", "negative"], yticklabels=["positive", "negative"])
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    """

    return(model)