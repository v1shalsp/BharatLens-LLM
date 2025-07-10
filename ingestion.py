# ingestion.py
from newsapi import NewsApiClient
import tweepy
import config
from database import SessionLocal, NewsArticle, Tweet, MonitoredTopic
from datetime import datetime
from textblob import TextBlob
import trafilatura

# Initialize APIs
newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)

# --- FIX: Function signature corrected to accept the MonitoredTopic object ---
def fetch_and_store_news(topic: MonitoredTopic, db: SessionLocal):
    print(f"Fetching news for '{topic.name}'...")
    articles_raw = newsapi.get_everything(q=topic.name, language='en', sort_by='relevancy', page_size=25)
    
    for article_data in articles_raw.get("articles", []):
        exists = db.query(NewsArticle).filter(NewsArticle.url == article_data['url']).first()
        if not exists and article_data.get('title') != "[Removed]":
            # Use trafilatura to get clean article text
            downloaded = trafilatura.fetch_url(article_data['url'])
            main_text = trafilatura.extract(downloaded)
            
            if not main_text:
                main_text = article_data.get('description', '')

            new_article = NewsArticle(
                url=article_data['url'],
                title=article_data['title'],
                text=main_text,
                topic_id=topic.id,
                published_at=datetime.fromisoformat(article_data['publishedAt'].replace('Z', '')),
                # --- FIX: Sentiment is now calculated and stored ---
                sentiment=TextBlob(main_text).sentiment.polarity
            )
            db.add(new_article)
    db.commit()

# --- FIX: Function signature corrected to accept the MonitoredTopic object ---
def fetch_and_store_tweets(topic: MonitoredTopic, db: SessionLocal):
    print(f"Fetching tweets for keywords: '{topic.keywords}'...")
    client = tweepy.Client(bearer_token=config.X_BEARER_TOKEN)
    query = f"({topic.keywords}) -is:retweet lang:en"

    try:
        response = client.search_recent_tweets(query, max_results=25, tweet_fields=["created_at", "author_id"])
        if not response.data:
            print("No new tweets found.")
            return

        for tweet_data in response.data:
            exists = db.query(Tweet).filter(Tweet.tweet_id == tweet_data.id).first()
            if not exists:
                new_tweet = Tweet(
                    tweet_id=str(tweet_data.id),
                    text=tweet_data.text,
                    author_id=str(tweet_data.author_id),
                    topic_id=topic.id,
                    created_at=tweet_data.created_at,
                    # --- FIX: Sentiment is now calculated and stored ---
                    sentiment=TextBlob(tweet_data.text).sentiment.polarity
                )
                db.add(new_tweet)
        db.commit()
        print("Successfully stored new tweets.")

    except tweepy.errors.TooManyRequests:
        print("X API rate limit hit. Skipping this fetch cycle.")
    except Exception as e:
        print(f"An unexpected error occurred during tweet ingestion: {e}")