# tests/test_pipeline.py
# ============================================================
# Unit tests for core modules
# Run: pytest tests/
# ============================================================

import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ====== Fixtures ======

@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    close = 1000 + np.cumsum(np.random.randn(n) * 15)
    close = np.clip(close, 100, 5000)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + abs(np.random.randn(n)) * 0.01),
        "low": close * (1 - abs(np.random.randn(n)) * 0.01),
        "close": close,
        "volume": np.random.randint(100000, 5000000, n).astype(float),
    }, index=dates)
    return df


# ====== Tests: Data Processor ======

class TestFeatureEngineer:
    def test_build_features_output_shape(self, sample_ohlcv):
        from src.data.processor import FeatureEngineer
        eng = FeatureEngineer()
        result = eng.build_features(sample_ohlcv)
        assert not result.empty, "Feature output should not be empty"
        assert "rsi" in result.columns, "RSI should be computed"
        assert "macd" in result.columns, "MACD should be computed"
        assert "target" in result.columns, "Target variable should exist"

    def test_rsi_bounds(self, sample_ohlcv):
        from src.data.processor import compute_rsi
        rsi = compute_rsi(sample_ohlcv["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all(), "RSI should be >= 0"
        assert (valid <= 100).all(), "RSI should be <= 100"

    def test_feature_columns_present(self, sample_ohlcv):
        from src.data.processor import FeatureEngineer
        eng = FeatureEngineer()
        df = eng.build_features(sample_ohlcv)
        feature_cols = eng.get_feature_columns()
        for col in feature_cols:
            assert col in df.columns, f"Feature '{col}' missing"


# ====== Tests: Technical Analysis ======

class TestTechnicalAnalyzer:
    def test_analyze_returns_signals(self, sample_ohlcv):
        from src.analysis.technical import TechnicalAnalyzer
        ta = TechnicalAnalyzer()
        result = ta.analyze(sample_ohlcv)
        assert "signals" in result
        assert "overall" in result
        assert result["overall"] in ["Bullish", "Bearish", "Neutral"]

    def test_signal_count(self, sample_ohlcv):
        from src.analysis.technical import TechnicalAnalyzer
        ta = TechnicalAnalyzer()
        result = ta.analyze(sample_ohlcv)
        total = result["bullish_signals"] + result["bearish_signals"] + result["neutral_signals"]
        assert total == len(result["signals"]), "Signal counts should match total"


# ====== Tests: Risk Analysis ======

class TestRiskAnalyzer:
    def test_risk_category_valid(self, sample_ohlcv):
        from src.analysis.risk import RiskAnalyzer
        ra = RiskAnalyzer()
        result = ra.analyze(sample_ohlcv)
        assert result["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]

    def test_sharpe_ratio_computed(self, sample_ohlcv):
        from src.analysis.risk import RiskAnalyzer
        ra = RiskAnalyzer()
        result = ra.analyze(sample_ohlcv)
        assert "sharpe_ratio" in result
        assert isinstance(result["sharpe_ratio"], float)

    def test_max_drawdown_negative(self, sample_ohlcv):
        from src.analysis.risk import RiskAnalyzer
        ra = RiskAnalyzer()
        result = ra.analyze(sample_ohlcv)
        assert result["max_drawdown"] <= 0, "Max drawdown should be negative or zero"


# ====== Tests: Recommendation Engine ======

class TestRecommendationEngine:
    def test_action_valid(self, sample_ohlcv):
        from src.analysis.technical import TechnicalAnalyzer
        from src.analysis.risk import RiskAnalyzer
        from src.analysis.recommendation import RecommendationEngine
        ta = TechnicalAnalyzer()
        ra = RiskAnalyzer()
        re = RecommendationEngine()
        tech = ta.analyze(sample_ohlcv)
        risk = ra.analyze(sample_ohlcv)
        pred = {"prob_increase": 65, "prob_decrease": 35, "trend": "Bullish",
                "current_price": 1000}
        sent = {"sentiment_label": "Bullish", "positive_count": 5,
                "negative_count": 1, "compound_score": 0.3}
        rec = re.generate("TEST.NS", pred, tech, sent, risk)
        assert rec["action"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= rec["confidence"] <= 100


# ====== Tests: Database ======

class TestDatabase:
    def test_db_init(self):
        from src.data.database import init_db, SessionLocal, StockPrice
        init_db()
        db = SessionLocal()
        count = db.query(StockPrice).count()
        db.close()
        assert count >= 0

    def test_portfolio_add(self):
        from src.data.database import init_db, SessionLocal, PortfolioHolding
        init_db()
        db = SessionLocal()
        try:
            holding = PortfolioHolding(
                ticker="TEST.NS", company_name="Test Company",
                quantity=10, avg_buy_price=100.0
            )
            db.add(holding)
            db.commit()
            retrieved = db.query(PortfolioHolding).filter(
                PortfolioHolding.ticker == "TEST.NS"
            ).first()
            assert retrieved is not None
            # Cleanup
            db.delete(retrieved)
            db.commit()
        finally:
            db.close()


# ====== Tests: Sentiment ======

class TestSentiment:
    def test_analyze_empty(self):
        from src.analysis.sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer()
        result = sa.analyze_sentiment([])
        assert result["compound_score"] == 0.0
        assert result["sentiment_label"] == "Neutral"

    def test_positive_sentiment(self):
        from src.analysis.sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer()
        articles = [
            {"title": "Stock surges to record high on strong earnings beat",
             "description": "Excellent quarterly results drive massive gains"},
        ]
        result = sa.analyze_sentiment(articles)
        assert result["total_articles"] == 1
        assert "sentiment_label" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
