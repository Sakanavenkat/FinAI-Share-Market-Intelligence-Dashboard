# src/data/database.py
# ============================================================
# SQLAlchemy models + init function
# ============================================================

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, Text, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class StockPrice(Base):
    """Daily OHLCV data."""
    __tablename__ = "stock_prices"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), index=True, nullable=False)
    date = Column(DateTime, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adj_close = Column(Float)
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)


class TechnicalIndicator(Base):
    """Calculated indicators per ticker per date."""
    __tablename__ = "technical_indicators"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True)
    date = Column(DateTime, index=True)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_9 = Column(Float)
    ema_21 = Column(Float)
    vwap = Column(Float)
    atr = Column(Float)
    obv = Column(Float)
    support = Column(Float)
    resistance = Column(Float)
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_tech_ticker_date"),)


class Prediction(Base):
    """ML model predictions."""
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True)
    prediction_date = Column(DateTime, index=True)  # date prediction was made
    target_date = Column(DateTime)                  # date predicted for
    horizon_days = Column(Integer)                  # 1, 7, 30, 90
    prob_increase = Column(Float)
    prob_decrease = Column(Float)
    predicted_price = Column(Float)
    price_low = Column(Float)
    price_high = Column(Float)
    trend = Column(String(10))   # Bullish / Bearish / Neutral
    model_used = Column(String(30))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class Recommendation(Base):
    """BUY / SELL / HOLD recommendations."""
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True)
    date = Column(DateTime, index=True)
    action = Column(String(10))    # BUY | SELL | HOLD
    confidence = Column(Float)
    risk_level = Column(String(10))
    expected_trend = Column(String(10))
    explanation = Column(Text)
    factors = Column(Text)         # JSON list of supporting factors
    created_at = Column(DateTime, default=datetime.utcnow)


class SentimentScore(Base):
    """News sentiment per ticker."""
    __tablename__ = "sentiment_scores"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True)
    date = Column(DateTime, index=True)
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    compound_score = Column(Float)
    sentiment_label = Column(String(10))  # Bullish / Bearish / Neutral
    headline_summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class NewsArticle(Base):
    """Individual news articles."""
    __tablename__ = "news_articles"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True)
    title = Column(Text)
    url = Column(Text)
    source = Column(String(100))
    published_at = Column(DateTime)
    sentiment_score = Column(Float)
    sentiment_label = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)


class PortfolioHolding(Base):
    """User portfolio positions."""
    __tablename__ = "portfolio_holdings"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), index=True)
    company_name = Column(String(100))
    quantity = Column(Float)
    avg_buy_price = Column(Float)
    added_at = Column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    """ML model performance tracking."""
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50))
    ticker = Column(String(20))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    trained_at = Column(DateTime, default=datetime.utcnow)
    is_best = Column(Boolean, default=False)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized.")


def get_db():
    """Dependency for FastAPI / manual use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
