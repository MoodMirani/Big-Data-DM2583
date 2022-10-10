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
from functions import *

def train_multinomial_naive_bayes(training_set):
    categories = ['positive', 'negative']
    X_train, X_test, y_train, y_test = train_test_split(training_set.tweet, training_set.sentiment, test_size=0.30, random_state=80)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    """
    # performance measure
    print(model.M.score(y_train, y_test)) # test score
    matrix = plot_confusion_matrix(model, y_train, y_test, normalize="true") 
    plot_roc_curve(model, y_train, y_test)
    plt.show()
    """
    return(model)