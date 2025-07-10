# anomaly_detector.py
from sqlalchemy import func
from database import SessionLocal, NewsArticle, Tweet, MonitoredTopic
from datetime import datetime, timedelta

DIVERGENCE_THRESHOLD = 0.4 # e.g., if MSM is 0.1 and X is -0.3, difference is 0.4

def check_narrative_divergence():
    """The core feedback loop logic."""
    print("Checking for narrative divergence...")
    with SessionLocal() as db:
        topics = db.query(MonitoredTopic).all()
        for topic in topics:
            # Get average sentiment from news in the last 24 hours
            avg_news_sentiment = db.query(func.avg(NewsArticle.sentiment))\
                .filter(NewsArticle.topic_id == topic.id)\
                .filter(NewsArticle.published_at >= datetime.now() - timedelta(hours=24))\
                .scalar() or 0.0

            # Get average sentiment from tweets in the last 3 hours
            avg_tweet_sentiment = db.query(func.avg(Tweet.sentiment))\
                .filter(Tweet.topic_id == topic.id)\
                .filter(Tweet.created_at >= datetime.now() - timedelta(hours=3))\
                .scalar() or 0.0
            
            divergence = abs(avg_news_sentiment - avg_tweet_sentiment)
            
            print(f"Topic: '{topic.name}' | News Sent: {avg_news_sentiment:.2f} | Tweet Sent: {avg_tweet_sentiment:.2f} | Divergence: {divergence:.2f}")

            # THE FEEDBACK LOOP TRIGGER
            if divergence > DIVERGENCE_THRESHOLD:
                handle_anomaly(topic, db)

def handle_anomaly(topic, db_session):
    """This is the 'ACT' part of the loop."""
    print(f"ALERT: High narrative divergence detected for topic '{topic.name}'!")
    # In a real system, this could trigger automated actions, like adjusting keywords.