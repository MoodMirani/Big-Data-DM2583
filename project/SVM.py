from xml.sax.handler import all_features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from pathlib import Path
from Cleanup import cleanup

def Train_SVM_Classifier(Training_Set):
    print("Training the SVM Classifier")

    # clean data
    Training_Set.tweet = cleanup(Training_Set.tweet)

    # vectorizing
    vectorizer = CountVectorizer(stop_words="english")
    all_features = vectorizer.fit_transform(Training_Set.tweet)

    # splitting
    X_train, X_test, y_train, y_test = train_test_split(all_features, Training_Set.sentiment, test_size=0.30, random_state=42)

    # create classifier
    classifier = svm.SVC() # the kernel: Gaussian Radial basis function
    classifier.fit(X_train, y_train)

    """ 
    # performance measure
    print("SVM Classifier score: " + str(classifier.score(X_test, y_test)))
    ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, normalize="true") 
    RocCurveDisplay.from_estimator(classifier, X_test, y_test)
    plt.show()
    """
    return classifier
