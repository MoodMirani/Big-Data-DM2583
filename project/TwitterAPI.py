import tweepy
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

def Get_Tweets_From_API():
    load_dotenv()

    API_KEY = os.getenv("API_KEY")
    API_SECRET = os.getenv("API_SECRET")
    BEARER_TOKEN = os.getenv('BEARER_TOKEN')
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")

    client = tweepy.Client(BEARER_TOKEN, API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    query = "#wine -is:retweet lang:en"
    timeline_tweets = tweepy.Paginator(client.search_recent_tweets, query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100).flatten(limit=10000)

    def create_list_of_tweets(tweet_objects):
        tweets = []
        for tweet in tweet_objects:
            tweets.append(tweet.text)
        return tweets

    Wine_Data_Set = pd.DataFrame(create_list_of_tweets(timeline_tweets), columns=["tweet"])
    filepath = Path('project/data/Wine_Data_Set.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    Wine_Data_Set.to_csv(filepath)

    return Wine_Data_Set