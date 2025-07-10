# database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
from sqlalchemy import JSON

DATABASE_URL = "sqlite:///analysis_data.db"
# --- FIX: Added connect_args for safe multi-threading with SQLite ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MonitoredTopic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    keywords = Column(Text) # Comma-separated list of tracking keywords

    # --- FIX: Added relationship to link back to articles and results ---
    articles = relationship("NewsArticle", back_populates="topic")
    tweets = relationship("Tweet", back_populates="topic")
    results = relationship("AnalysisResult", back_populates="topic")


class NewsArticle(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True)
    title = Column(String)
    text = Column(Text)
    # --- FIX: Foreign key constraint added ---
    topic_id = Column(Integer, ForeignKey('topics.id'))
    published_at = Column(DateTime)
    sentiment = Column(Float, default=0.0)
    is_analyzed = Column(Boolean, default=False)
    # --- FIX: Relationship defined ---
    topic = relationship("MonitoredTopic", back_populates="articles")

class Tweet(Base):
    __tablename__ = "tweets"
    id = Column(Integer, primary_key=True)
    tweet_id = Column(String, unique=True)
    text = Column(Text)
    author_id = Column(String)
    # --- FIX: Foreign key constraint added ---
    topic_id = Column(Integer, ForeignKey('topics.id'))
    created_at = Column(DateTime)
    sentiment = Column(Float, default=0.0)
    is_analyzed = Column(Boolean, default=False)
    # --- FIX: Relationship defined ---
    topic = relationship("MonitoredTopic", back_populates="tweets")

class AnalysisResult(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True)
    # --- FIX: Foreign key constraint added ---
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Store complex data as JSON strings in the database
    executive_summary = Column(Text)
    perspectives = Column(JSON)
    summary_evaluation = Column(JSON)
    visualizations = Column(JSON)
    detailed_bias_report = Column(Text)
    # --- FIX: Relationship defined ---
    topic = relationship("MonitoredTopic", back_populates="results")

def init_db():
    # Create all tables in the database
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")