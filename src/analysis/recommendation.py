# src/analysis/recommendation.py
# ============================================================
# BUY / SELL / HOLD Recommendation Engine
# ============================================================

from datetime import datetime
from loguru import logger
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import BUY_CONFIDENCE_THRESHOLD, SELL_CONFIDENCE_THRESHOLD
from src.data.database import SessionLocal, Recommendation
import json


class RecommendationEngine:
    """Synthesizes technical, sentiment, and ML signals into a recommendation."""

    def generate(
        self,
        ticker: str,
        prediction: dict,
        technical: dict,
        sentiment: dict,
        risk: dict,
    ) -> dict:
        """
        Combines all signals to produce BUY / SELL / HOLD with explanation.
        """
        factors_bullish = []
        factors_bearish = []
        score = 0.0
        max_score = 0.0

        # --- ML Prediction (weight: 35%) ---
        max_score += 35
        prob_up = prediction.get("prob_increase", 50) / 100
        ml_contribution = prob_up * 35
        score += ml_contribution
        if prob_up > 0.65:
            factors_bullish.append(
                f"✓ AI Model predicts {prediction.get('prob_increase')}% probability of price increase"
            )
        elif prob_up < 0.40:
            factors_bearish.append(
                f"✗ AI Model predicts only {prediction.get('prob_increase')}% probability of increase"
            )

        # --- Technical Analysis (weight: 35%) ---
        max_score += 35
        bull_score = technical.get("bull_score", 50) / 100
        tech_contribution = bull_score * 35
        score += tech_contribution

        bull_sig = technical.get("bullish_signals", 0)
        bear_sig = technical.get("bearish_signals", 0)

        # RSI signal
        rsi_val = technical.get("summary", {}).get("rsi", 50)
        if rsi_val < 40:
            factors_bullish.append(f"✓ RSI at {rsi_val:.0f} — oversold territory (bullish reversal zone)")
        elif rsi_val > 65:
            factors_bearish.append(f"✗ RSI at {rsi_val:.0f} — overbought (correction risk)")

        # Moving Average signal
        if technical.get("summary", {}).get("golden_cross"):
            factors_bullish.append("✓ Golden Cross active — SMA50 above SMA200 (long-term bullish)")
        else:
            factors_bearish.append("✗ Death Cross in effect — bearish long-term trend")

        # VWAP
        vwap = technical.get("summary", {}).get("vwap", 0)
        cp = technical.get("current_price", 0)
        if cp > vwap:
            factors_bullish.append(f"✓ Price (₹{cp}) trading above VWAP (₹{vwap:.0f})")
        else:
            factors_bearish.append(f"✗ Price (₹{cp}) below VWAP (₹{vwap:.0f})")

        # Volume
        vol_ratio = technical.get("summary", {}).get("volume_ratio", 1)
        if vol_ratio > 1.5:
            factors_bullish.append(f"✓ Strong volume at {vol_ratio:.1f}x average — institutional interest")

        # MACD
        macd_hist = technical.get("summary", {}).get("macd_hist", 0)
        if macd_hist > 0:
            factors_bullish.append("✓ MACD histogram positive — bullish momentum")
        else:
            factors_bearish.append("✗ MACD histogram negative — bearish momentum")

        # --- Sentiment (weight: 20%) ---
        max_score += 20
        sentiment_label = sentiment.get("sentiment_label", "Neutral")
        pos = sentiment.get("positive_count", 0)
        neg = sentiment.get("negative_count", 0)
        compound = sentiment.get("compound_score", 0)

        if sentiment_label == "Bullish":
            score += 20
            factors_bullish.append(
                f"✓ Positive news sentiment: {pos} positive vs {neg} negative articles"
            )
        elif sentiment_label == "Bearish":
            score += 5
            factors_bearish.append(
                f"✗ Negative news sentiment: {neg} negative articles dominate"
            )
        else:
            score += 10
            factors_bullish.append(f"~ Neutral news sentiment (compound score: {compound:.2f})")

        # --- Risk Adjustment (weight: 10%) ---
        max_score += 10
        risk_cat = risk.get("risk_category", "Medium Risk")
        sharpe = risk.get("sharpe_ratio", 0)

        if risk_cat == "Low Risk":
            score += 10
            factors_bullish.append(f"✓ Low risk profile — stable stock suitable for all investors")
        elif risk_cat == "Medium Risk":
            score += 6
        else:
            score += 2
            factors_bearish.append(f"✗ High risk/volatility ({risk.get('volatility_annual', 0):.0f}% annualized)")

        if sharpe > 1:
            factors_bullish.append(f"✓ Strong Sharpe Ratio ({sharpe:.2f}) — good risk-adjusted returns")
        elif sharpe < 0:
            factors_bearish.append(f"✗ Negative Sharpe Ratio ({sharpe:.2f}) — poor risk-adjusted returns")

        # --- Compute Confidence & Action ---
        confidence = score / max_score if max_score > 0 else 0.5

        if confidence >= BUY_CONFIDENCE_THRESHOLD:
            action = "BUY"
            action_emoji = "🟢"
            expected_trend = prediction.get("trend", "Bullish")
        elif confidence <= SELL_CONFIDENCE_THRESHOLD:
            action = "SELL"
            action_emoji = "🔴"
            expected_trend = "Bearish"
        else:
            action = "HOLD"
            action_emoji = "🟡"
            expected_trend = "Neutral"

        # --- Risk Level ---
        risk_level = risk.get("risk_category", "Medium Risk").replace(" Risk", "")

        # --- Human-readable explanation ---
        explanation = self._build_explanation(
            ticker, action, factors_bullish, factors_bearish,
            prediction, technical, sentiment, risk, confidence
        )

        result = {
            "ticker": ticker,
            "action": action,
            "action_emoji": action_emoji,
            "confidence": round(confidence * 100, 1),
            "risk_level": risk_level,
            "expected_trend": expected_trend,
            "bullish_factors": factors_bullish,
            "bearish_factors": factors_bearish,
            "explanation": explanation,
            "score": round(score, 1),
            "max_score": max_score,
            "generated_at": datetime.now().isoformat(),
        }
        return result

    # ----------------------------------------------------------
    def _build_explanation(
        self, ticker, action, bull_factors, bear_factors,
        prediction, technical, sentiment, risk, confidence
    ) -> str:
        company = ticker.replace(".NS", "").replace(".BO", "")
        trend = prediction.get("trend", "Neutral")
        prob = prediction.get("prob_increase", 50)
        price = prediction.get("current_price", 0)
        rsi = technical.get("summary", {}).get("rsi", 50)
        sent = sentiment.get("sentiment_label", "Neutral")

        lines = [
            f"**{company}** is recommended to **{action}** with {confidence*100:.0f}% confidence.",
            "",
            f"The AI model assigns a **{prob}%** probability that the stock will rise tomorrow. "
            f"Current price is ₹{price}.",
            "",
            f"Technical indicators show a **{trend.lower()}** bias (RSI: {rsi:.0f}). "
            f"News sentiment is **{sent.lower()}** based on recent coverage.",
            "",
        ]
        if bull_factors:
            lines.append("**Supporting factors:**")
            lines.extend(bull_factors)
        if bear_factors:
            lines.append("")
            lines.append("**Risk factors:**")
            lines.extend(bear_factors)

        return "\n".join(lines)

    # ----------------------------------------------------------
    def save_to_db(self, ticker: str, rec: dict):
        db = SessionLocal()
        try:
            record = Recommendation(
                ticker=ticker, date=datetime.now(),
                action=rec["action"], confidence=rec["confidence"] / 100,
                risk_level=rec["risk_level"], expected_trend=rec["expected_trend"],
                explanation=rec["explanation"],
                factors=json.dumps(rec["bullish_factors"] + rec["bearish_factors"]),
            )
            db.add(record)
            db.commit()
        finally:
            db.close()
