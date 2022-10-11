from pathlib import Path
import re
import pandas as pd
from TwitterAPI import Get_Tweets_From_API
from collections import Counter
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

def most_common_words(dataframe):
    text = ""
    for row in dataframe.tweet:
        text = text + " " + row
    
    print("tokenizing the words")
    words = word_tokenize(text)

    print("removing stopwords")
    words_without_stopwords = [word for word in words if not word in stopwords.words()]

    print("counting the words")
    counter = Counter(words_without_stopwords)

    print("printing the words")
    most_occur = counter.most_common(40)
    for word in most_occur:
        print(word)


def gather_wine_tweets(choice):
    if choice == "load_from_local":
        print("Loading tweets about wine from the local drive")
        wine_data_set = pd.read_csv('project/data/cleaned_wine_data_set.csv', names=["tweet"])
        wine_data_set = wine_data_set[1:] # removing header
        return wine_data_set

    elif choice == "Load_From_Twitter_API":
        print("Gathering tweets about wine from the Twitter API")
        wine_data_set = Get_Tweets_From_API()
        print("Cleaning the tweets")
        wine_data_set.tweet = cleanup(wine_data_set.tweet)
        save_to_file("cleaned_wine_data_set.csv", wine_data_set)
        return wine_data_set


def create_training_set(choice):
    if choice == "load_local_cleaned_Data":
        training_set = pd.read_csv("project/data/cleaned_training_data.csv", names=["sentiment", "tweet"])
        training_set = training_set[1:]
    
    elif choice == "extract_new_data":
        training_set = pd.read_csv("project/data/training.csv", names=["sentiment", "ID", "date", "query", "user", "tweet"])
        training_set = training_set.drop([ "ID", "date", "query", "user"], axis=1)
        training_set_negative = training_set[1:700000] # Extract 700000 negative tweets
        training_set_positive = training_set[810000:1510000] # Extract 700000 positive tweets
        training_set = pd.concat([training_set_negative, training_set_positive]) # merge them into one file
        training_set.tweet = cleanup(training_set.tweet)
        save_to_file("cleaned_training_data.csv", training_set)
    return training_set

def save_to_file(filename, data):
    filepath = Path('project/data/' + filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    data.to_csv(filepath)
    print("Saved the tweets to the file: " + filename)

def cleanup(data):
    cleaned_data = []
    counter=0
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
    return cleaned_data