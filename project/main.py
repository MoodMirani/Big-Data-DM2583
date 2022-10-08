from pathlib import Path
from SVM import Train_SVM_Classifier
import pandas as pd
from TwitterAPI import Get_Tweets_From_API
from Cleanup import cleanup
from sklearn.feature_extraction.text import CountVectorizer


def main():
    Training_Set = Create_Training_Set()
    SVM_Classifier = Train_SVM_Classifier(Training_Set)
    Wine_Data_Set = Gather_Wine_Tweets("Load_From_Local")
    y = Classify_Data(Wine_Data_Set, SVM_Classifier)
    print("printing the shape of y: ")
    print(y.shape)


def Classify_Data(Wine_Data_Set, SVM_Classifier):
    print("Classifying the Wine Data")
    # vectorizing
    vectorizer = CountVectorizer(stop_words="english")
    all_features = vectorizer.fit_transform(Wine_Data_Set.tweet)
    y = SVM_Classifier.predict([all_features[0]])
    return y

def Save_To_File(filename, data):
    filepath = Path('project/data/' + filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    data.to_csv(filepath)
    print("Saved the tweets to the file: " + filename)

def Create_Training_Set():
    # Gather data
    Training_Set = pd.read_csv("project/data/training.csv", names=["sentiment", "ID", "date", "query", "user", "tweet"])
    Training_Set = Training_Set.drop([ "ID", "date", "query", "user"], axis=1)
    Training_Set_Negative = Training_Set[1:7000] # Extract 7000 negative tweets
    Training_Set_Positive = Training_Set[900000:907000] # Extract 7000 positive tweets
    Training_Set = pd.concat([Training_Set_Negative, Training_Set_Positive]) # merge them into one file
    return Training_Set

def Gather_Wine_Tweets(Choice):
    if Choice == "Load_From_Local":
        print("Loading tweets about wine from the local drive")
        Wine_Data_Set = pd.read_csv('project/data/Cleaned_Wine_Data_Set.csv', names=["tweet"])
        Wine_Data_Set = Wine_Data_Set[1:] # removing header
        return Wine_Data_Set

    elif Choice == "Load_From_Twitter_API":
        print("Gathering tweets about wine from the Twitter API")
        Wine_Data_Set = Get_Tweets_From_API()
        print("Cleaning the tweets")
        Wine_Data_Set.tweet = cleanup(Wine_Data_Set.tweet)
        Save_To_File("Cleaned_Wine_Data_Set.csv", Wine_Data_Set)
        return Wine_Data_Set

# start the programme
main()

