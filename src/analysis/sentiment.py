# src/analysis/sentiment.py
# ============================================================
# News Sentiment Analysis using VADER + NewsAPI
# ============================================================

import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import (
    NEWS_API_KEY, NEWS_LOOKBACK_DAYS, MAX_NEWS_PER_STOCK,
    SENTIMENT_POSITIVE_THRESHOLD, SENTIMENT_NEGATIVE_THRESHOLD
)
from config.stocks_list import get_company_name
from src.data.database import SessionLocal, SentimentScore, NewsArticle


class SentimentAnalyzer:
    """Fetches news and computes sentiment scores."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.has_api_key = bool(NEWS_API_KEY)

    # ----------------------------------------------------------
    def fetch_news(self, ticker: str, company_name: str = None) -> list:
        """Fetch news articles from NewsAPI."""
        if not self.has_api_key:
            return self._get_mock_news(ticker)

        name = company_name or get_company_name(ticker)
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        query = f"{name} OR {symbol} stock India"
        from_date = (datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": MAX_NEWS_PER_STOCK,
                "apiKey": NEWS_API_KEY,
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            articles = data.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "url": a.get("url", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "published_at": a.get("publishedAt", ""),
                }
                for a in articles if a.get("title")
            ]
        except Exception as e:
            logger.warning(f"NewsAPI failed for {ticker}: {e}")
            return self._get_mock_news(ticker)

    # ----------------------------------------------------------
    def analyze_sentiment(self, articles: list) -> dict:
        """Score each article and compute aggregate sentiment."""
        if not articles:
            return {
                "positive_count": 0, "negative_count": 0, "neutral_count": 0,
                "compound_score": 0.0, "sentiment_label": "Neutral",
                "articles": [], "headline_summary": "No recent news available.",
            }

        scored = []
        pos, neg, neu = 0, 0, 0
        compound_total = 0.0

        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".strip()
            scores = self.vader.polarity_scores(text)
            compound = scores["compound"]
            compound_total += compound

            if compound >= SENTIMENT_POSITIVE_THRESHOLD:
                label = "Positive"
                pos += 1
            elif compound <= SENTIMENT_NEGATIVE_THRESHOLD:
                label = "Negative"
                neg += 1
            else:
                label = "Neutral"
                neu += 1

            scored.append({
                **article,
                "compound_score": round(compound, 4),
                "sentiment_label": label,
            })

        avg_compound = compound_total / len(articles) if articles else 0
        if avg_compound >= SENTIMENT_POSITIVE_THRESHOLD:
            overall = "Bullish"
        elif avg_compound <= SENTIMENT_NEGATIVE_THRESHOLD:
            overall = "Bearish"
        else:
            overall = "Neutral"

        top_titles = [a["title"] for a in scored[:3] if a.get("title")]
        summary = " | ".join(top_titles[:2]) if top_titles else "No headlines."

        return {
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "total_articles": len(articles),
            "compound_score": round(avg_compound, 4),
            "sentiment_label": overall,
            "articles": scored[:10],
            "headline_summary": summary,
        }

    # ----------------------------------------------------------
    def get_stock_sentiment(self, ticker: str) -> dict:
        """Full pipeline: fetch → analyze → return result."""
        company = get_company_name(ticker)
        articles = self.fetch_news(ticker, company)
        result = self.analyze_sentiment(articles)
        result["ticker"] = ticker
        result["company_name"] = company
        result["analyzed_at"] = datetime.now().isoformat()
        return result

    # ----------------------------------------------------------
    def save_to_db(self, ticker: str, sentiment: dict):
        db = SessionLocal()
        try:
            now = datetime.now()
            score_record = SentimentScore(
                ticker=ticker, date=now,
                positive_count=sentiment["positive_count"],
                negative_count=sentiment["negative_count"],
                neutral_count=sentiment["neutral_count"],
                compound_score=sentiment["compound_score"],
                sentiment_label=sentiment["sentiment_label"],
                headline_summary=sentiment.get("headline_summary", ""),
            )
            db.add(score_record)

            for a in sentiment.get("articles", []):
                pub = None
                if a.get("published_at"):
                    try:
                        pub = datetime.fromisoformat(a["published_at"].replace("Z", "+00:00"))
                    except Exception:
                        pub = now
                art = NewsArticle(
                    ticker=ticker, title=a.get("title", ""),
                    url=a.get("url", ""), source=a.get("source", ""),
                    published_at=pub,
                    sentiment_score=a.get("compound_score", 0),
                    sentiment_label=a.get("sentiment_label", "Neutral"),
                )
                db.add(art)
            db.commit()
        finally:
            db.close()

    # ----------------------------------------------------------
    def _get_mock_news(self, ticker: str) -> list:
        """Return mock news when API key not configured."""
        symbol = ticker.replace(".NS", "")
        company = get_company_name(ticker)
        return [
            {
                "title": f"{company} reports strong Q3 earnings, beats estimates",
                "description": f"{company} delivered better-than-expected quarterly results driven by strong domestic demand.",
                "url": "#", "source": "Economic Times",
                "published_at": datetime.now().isoformat(),
            },
            {
                "title": f"Analysts upgrade {symbol} to BUY with revised target",
                "description": f"Multiple brokerages have raised their price targets for {company} citing positive momentum.",
                "url": "#", "source": "Moneycontrol",
                "published_at": (datetime.now() - timedelta(days=1)).isoformat(),
            },
            {
                "title": f"{company} announces expansion plans",
                "description": f"{company} plans to invest in capacity expansion as demand outlook remains positive.",
                "url": "#", "source": "Business Standard",
                "published_at": (datetime.now() - timedelta(days=2)).isoformat(),
            },
        ]
