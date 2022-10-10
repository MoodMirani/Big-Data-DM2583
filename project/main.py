from pathlib import Path
from SVM import Train_SVM_Classifier
import pandas as pd
from TwitterAPI import Get_Tweets_From_API
from sklearn.feature_extraction.text import CountVectorizer
from functions import *
from MultinomialNB import train_multinomial_naive_bayes
from collections import Counter
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize



def main():
    training_set = create_training_set("load_local_cleaned_Data")
    wine_data_set = gather_wine_tweets("load_from_local")
    print(wine_data_set.tail)

    model = train_multinomial_naive_bayes(training_set)

    sentiment = model.predict(wine_data_set.tweet)

    predicted_wine_tweets = pd.DataFrame({
        "sentiment": sentiment,
        "tweet": wine_data_set.tweet
    })

    save_to_file("predicted_wine_tweets.csv", predicted_wine_tweets)

    # find all tweets with positive sentiment
    positive_predicted_wine_tweets = predicted_wine_tweets[predicted_wine_tweets["sentiment"]=="4"]
    save_to_file("positive_predicted_wine_tweets.csv", positive_predicted_wine_tweets)

    text = ""
    for row in positive_predicted_wine_tweets.tweet:
        text = text + " " + row
    
    words = word_tokenize(text)
    words_without_stopwords = [word for word in words if not word in stopwords.words()]
      
    # split() returns list of all the words in the string
    #split_it = text.split()
    
    # Pass the split_it list to instance of Counter class.
    counter = Counter(words_without_stopwords)
    
    # most_common() produces k frequently encountered
    # input values and their respective counts.
    most_occur = counter.most_common(40)

    for word in most_occur:
        print(word)



# start the programme
main()

