from pathlib import Path
import pandas as pd
from TwitterAPI import Get_Tweets_From_API
from sklearn.feature_extraction.text import CountVectorizer
from functions import *
from MultinomialNB import train_multinomial_naive_bayes

def main():
    training_set = create_training_set("extract_new_data")
    wine_data_set = gather_wine_tweets("load_from_local")

    model = train_multinomial_naive_bayes(training_set)

    sentiment = model.predict(wine_data_set.tweet)

    predicted_wine_tweets = pd.DataFrame({
        "sentiment": sentiment,
        "tweet": wine_data_set.tweet
    })

    save_to_file("predicted_wine_tweets.csv", predicted_wine_tweets)

    # find all tweets with positive sentiment
    positive_predicted_wine_tweets = predicted_wine_tweets[predicted_wine_tweets["sentiment"]==4]
    save_to_file("positive_predicted_wine_tweets.csv", positive_predicted_wine_tweets)

    # find all tweets with negative sentiment
    negative_predicted_wine_tweets = predicted_wine_tweets[predicted_wine_tweets["sentiment"]==0]
    save_to_file("negative_predicted_wine_tweets.csv", negative_predicted_wine_tweets)

    print("Printing most common words in positive tweets")
    most_common_words(positive_predicted_wine_tweets)
    print("Printing most common words in negative tweets")
    most_common_words(negative_predicted_wine_tweets)

# start the programme
main()

