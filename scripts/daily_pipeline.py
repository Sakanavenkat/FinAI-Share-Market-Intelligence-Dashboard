# scripts/daily_pipeline.py
# ============================================================
# Daily cron job: fetch → process → predict → save
# Run: python scripts/daily_pipeline.py
# Init: python scripts/daily_pipeline.py --init
# ============================================================

import sys, os, argparse
from loguru import logger
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.database import init_db
from src.data.fetcher import StockDataFetcher
from src.data.processor import FeatureEngineer
from src.ml.predictor import StockPredictor
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.risk import RiskAnalyzer
from src.analysis.recommendation import RecommendationEngine
from config.stocks_list import NIFTY50, get_yf_tickers
from config.settings import MAX_STOCKS


def run_pipeline(tickers: list, full_history: bool = False):
    logger.info(f"=== Daily Pipeline Started: {datetime.now()} ===")
    logger.info(f"Processing {len(tickers)} stocks")

    fetcher = StockDataFetcher()
    engineer = FeatureEngineer()
    predictor = StockPredictor()
    tech = TechnicalAnalyzer()
    sent = SentimentAnalyzer()
    risk = RiskAnalyzer()
    rec_engine = RecommendationEngine()

    # --- Step 1: Fetch Data ---
    logger.info("Step 1/5: Fetching market data...")
    period = "3y" if full_history else "6mo"
    data = fetcher.fetch_bulk(tickers, period=period)
    fetcher.save_to_db(data)
    fetcher.save_to_csv(data)
    logger.info(f"Fetched {len(data)} stocks")

    # --- Step 2: Feature Engineering + Predictions ---
    logger.info("Step 2/5: Running predictions...")
    successful = 0
    for ticker, df in data.items():
        try:
            features_df = engineer.build_features(df)
            if features_df.empty:
                continue
            pred = predictor.predict_tomorrow(ticker, df)
            forecast = predictor.forecast_multi_horizon(ticker, df)
            predictor.save_predictions_to_db(ticker, pred, forecast)
            successful += 1
        except Exception as e:
            logger.warning(f"Prediction failed for {ticker}: {e}")
    logger.info(f"Predictions generated for {successful} stocks")

    # --- Step 3: Technical Analysis ---
    logger.info("Step 3/5: Technical analysis...")
    for ticker, df in list(data.items())[:20]:  # Limit for performance
        try:
            tech.analyze(df)
        except Exception as e:
            logger.warning(f"Tech analysis failed for {ticker}: {e}")

    # --- Step 4: Sentiment ---
    logger.info("Step 4/5: Sentiment analysis (top 10 stocks)...")
    priority = [t for t in tickers if any(s in t for s in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"])]
    for ticker in priority[:10]:
        try:
            result = sent.get_stock_sentiment(ticker)
            sent.save_to_db(ticker, result)
        except Exception as e:
            logger.warning(f"Sentiment failed for {ticker}: {e}")

    # --- Step 5: Recommendations ---
    logger.info("Step 5/5: Generating recommendations...")
    for ticker in priority[:10]:
        try:
            if ticker in data:
                df = data[ticker]
                pred = predictor.predict_tomorrow(ticker, df)
                t_result = tech.analyze(df)
                s_result = sent.get_stock_sentiment(ticker)
                r_result = risk.analyze(df)
                recommendation = rec_engine.generate(ticker, pred, t_result, s_result, r_result)
                rec_engine.save_to_db(ticker, recommendation)
        except Exception as e:
            logger.warning(f"Recommendation failed for {ticker}: {e}")

    logger.info(f"=== Pipeline Complete: {datetime.now()} ===")
    return {"stocks_processed": len(data), "predictions": successful}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Full historical fetch")
    args = parser.parse_args()

    init_db()
    tickers = get_yf_tickers(NIFTY50[:MAX_STOCKS])
    result = run_pipeline(tickers, full_history=args.init)
    print(f"\n✅ Pipeline done: {result}")
