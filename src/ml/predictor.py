# src/ml/predictor.py
# ============================================================
# Inference Engine: generate predictions for any ticker
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.settings import MODELS_DIR, FORECAST_DAYS
from src.data.processor import FeatureEngineer
from src.data.fetcher import StockDataFetcher
from src.data.database import SessionLocal, Prediction
from src.ml.trainer import ModelTrainer


class StockPredictor:
    """Generates predictions for individual stocks."""

    def __init__(self):
        self.engineer = FeatureEngineer()
        self.feature_cols = self.engineer.get_feature_columns()
        self.fetcher = StockDataFetcher()
        self._trainers = {}  # Cache loaded trainers

    # ----------------------------------------------------------
    def _get_trainer(self, ticker: str) -> ModelTrainer:
        if ticker in self._trainers:
            return self._trainers[ticker]
        trainer = ModelTrainer(self.feature_cols)
        # Try ticker-specific model, fallback to global
        if not trainer.load_models(ticker):
            trainer.load_models("global")
        self._trainers[ticker] = trainer
        return trainer

    # ----------------------------------------------------------
    def predict_tomorrow(self, ticker: str, df: pd.DataFrame = None) -> dict:
        """Predict tomorrow's movement probability."""
        try:
            if df is None:
                df = self.fetcher.load_from_db(ticker, days=365)
                if df.empty:
                    df_raw = self.fetcher.fetch_single(ticker, period="1y")
                    if df_raw.empty:
                        return {"error": f"No data for {ticker}"}
                    df = df_raw

            features_df = self.engineer.build_features(df)
            if features_df.empty:
                return {"error": "Feature engineering failed"}

            trainer = self._get_trainer(ticker)
            if trainer.best_model is None:
                return {"error": "No trained model available. Run training first."}

            last_row = features_df[self.feature_cols].iloc[-1:].values
            last_scaled = trainer.scaler.transform(last_row)
            prob_increase = float(trainer.best_model.predict_proba(last_scaled)[0][1])
            prob_decrease = 1 - prob_increase

            current_price = float(df["Close"].iloc[-1])
            trend = "Bullish" if prob_increase > 0.6 else ("Bearish" if prob_increase < 0.4 else "Neutral")

            return {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "prob_increase": round(prob_increase * 100, 1),
                "prob_decrease": round(prob_decrease * 100, 1),
                "trend": trend,
                "model_used": trainer.best_model_name or "ensemble",
                "prediction_date": datetime.now().isoformat(),
                "target_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            }
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            return {"error": str(e)}

    # ----------------------------------------------------------
    def forecast_multi_horizon(self, ticker: str, df: pd.DataFrame = None) -> dict:
        """Generate forecasts for 1, 7, 30, 90-day horizons."""
        if df is None:
            df = self.fetcher.load_from_db(ticker, days=500)
            if df.empty:
                df = self.fetcher.fetch_single(ticker, period="2y")

        if df.empty:
            return {"error": "No data available"}

        features_df = self.engineer.build_features(df)
        trainer = self._get_trainer(ticker)
        if trainer.best_model is None:
            return {"error": "Model not trained"}

        current_price = float(df["Close"].iloc[-1])
        volatility = float(features_df["volatility_20d"].iloc[-1])
        last_scaled = trainer.scaler.transform(
            features_df[self.feature_cols].iloc[-1:].values
        )
        base_prob = float(trainer.best_model.predict_proba(last_scaled)[0][1])

        forecasts = {}
        for days in FORECAST_DAYS:
            # Simulate price drift using Monte Carlo approximation
            daily_vol = volatility / np.sqrt(252)
            drift = (base_prob - 0.5) * 0.002 * days
            noise_factor = daily_vol * np.sqrt(days)

            expected = current_price * (1 + drift)
            price_low = expected * (1 - noise_factor * 1.5)
            price_high = expected * (1 + noise_factor * 1.5)
            trend = "Bullish" if drift > 0.01 else ("Bearish" if drift < -0.01 else "Neutral")
            change_pct = round((expected / current_price - 1) * 100, 1)

            forecasts[f"{days}d"] = {
                "horizon_days": days,
                "expected_price": round(expected, 2),
                "price_low": round(price_low, 2),
                "price_high": round(price_high, 2),
                "trend": trend,
                "change_pct": change_pct,
                "confidence": round(abs(base_prob - 0.5) * 200, 1),
            }

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "forecasts": forecasts,
            "volatility": round(volatility * 100, 1),
            "base_probability": round(base_prob * 100, 1),
        }

    # ----------------------------------------------------------
    def get_top_predictions(self, tickers: list, top_n: int = 10) -> dict:
        """Rank all tickers and return top risers and fallers."""
        predictions = []
        for ticker in tickers:
            pred = self.predict_tomorrow(ticker)
            if "error" not in pred:
                predictions.append(pred)

        predictions.sort(key=lambda x: x["prob_increase"], reverse=True)
        return {
            "top_risers": predictions[:top_n],
            "top_fallers": list(reversed(predictions[-top_n:])),
            "generated_at": datetime.now().isoformat(),
        }

    # ----------------------------------------------------------
    def save_predictions_to_db(self, ticker: str, pred: dict, forecast: dict):
        db = SessionLocal()
        try:
            now = datetime.now()
            # Tomorrow prediction
            if "prob_increase" in pred:
                record = Prediction(
                    ticker=ticker,
                    prediction_date=now,
                    target_date=now + timedelta(days=1),
                    horizon_days=1,
                    prob_increase=pred["prob_increase"] / 100,
                    prob_decrease=pred["prob_decrease"] / 100,
                    predicted_price=pred.get("current_price"),
                    trend=pred.get("trend"),
                    model_used=pred.get("model_used"),
                    confidence=pred["prob_increase"] / 100,
                )
                db.add(record)

            # Multi-horizon forecasts
            if "forecasts" in forecast:
                for key, f in forecast["forecasts"].items():
                    rec = Prediction(
                        ticker=ticker,
                        prediction_date=now,
                        target_date=now + timedelta(days=f["horizon_days"]),
                        horizon_days=f["horizon_days"],
                        predicted_price=f["expected_price"],
                        price_low=f["price_low"],
                        price_high=f["price_high"],
                        trend=f["trend"],
                        confidence=f["confidence"] / 100,
                    )
                    db.add(rec)
            db.commit()
        finally:
            db.close()
