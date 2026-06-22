# scripts/train_models.py
# ============================================================
# Train / retrain all ML models on available data
# Run: python scripts/train_models.py
# Options: --ticker RELIANCE.NS  (train for specific stock)
#          --global               (train global model on all stocks)
# ============================================================

import sys, os, argparse, glob
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.database import init_db
from src.data.fetcher import StockDataFetcher
from src.data.processor import FeatureEngineer
from src.ml.trainer import ModelTrainer
from config.settings import RAW_DIR, MODELS_DIR
from config.stocks_list import NIFTY50, get_yf_tickers


def train_global_model():
    """Train one model on all stocks combined — fastest approach."""
    logger.info("Training GLOBAL model on all available data...")
    fetcher = StockDataFetcher()
    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns()
    trainer = ModelTrainer(feature_cols)

    all_features = []
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        logger.info("No local CSV files. Fetching NIFTY50 data...")
        tickers = get_yf_tickers(NIFTY50[:20])
        data = fetcher.fetch_bulk(tickers, period="2y")
        fetcher.save_to_csv(data)
        csv_files = list(RAW_DIR.glob("*.csv"))

    logger.info(f"Processing {len(csv_files)} CSV files...")
    for csv_path in tqdm(csv_files[:50]):  # Limit for speed
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    break
            else:
                features = engineer.build_features(df)
                if not features.empty and "target" in features.columns:
                    all_features.append(features)
        except Exception as e:
            logger.warning(f"Skipping {csv_path.name}: {e}")

    if not all_features:
        logger.error("No training data available!")
        return

    combined = pd.concat(all_features, ignore_index=True)
    combined = combined.dropna()
    logger.info(f"Total training samples: {len(combined)}")

    if len(combined) < 500:
        logger.warning("Very few samples. Consider fetching more data.")

    # Train classical models
    results = trainer.train_all(combined)
    trainer.save_models("global")
    trainer.save_metrics_to_db("global")

    logger.info("\n=== MODEL COMPARISON ===")
    for name, result in results.items():
        m = result["metrics"]
        logger.info(
            f"{name:20s} | Acc: {m['accuracy']:.3f} | "
            f"F1: {m['f1']:.3f} | AUC: {m['roc_auc']:.3f} "
            f"{'← BEST' if name == trainer.best_model_name else ''}"
        )

    # Try LSTM
    logger.info("\nTraining LSTM...")
    lstm_metrics = trainer.train_lstm(combined.tail(5000), "global")
    if lstm_metrics:
        logger.info(f"LSTM | F1: {lstm_metrics.get('f1', 0):.3f} | AUC: {lstm_metrics.get('roc_auc', 0):.3f}")

    logger.info(f"\n✅ Best model: {trainer.best_model_name}")
    logger.info(f"Models saved to {MODELS_DIR}")


def train_ticker_model(ticker: str):
    """Train a model specific to one ticker."""
    logger.info(f"Training model for {ticker}...")
    fetcher = StockDataFetcher()
    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns()
    trainer = ModelTrainer(feature_cols)

    df = fetcher.fetch_single(ticker, period="3y")
    if df.empty:
        logger.error(f"No data for {ticker}")
        return

    features = engineer.build_features(df)
    if len(features) < 200:
        logger.warning(f"Only {len(features)} samples for {ticker}. May not be enough.")

    results = trainer.train_all(features)
    trainer.save_models(ticker)
    trainer.save_metrics_to_db(ticker)

    logger.info(f"\n✅ {ticker} model trained. Best: {trainer.best_model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NSE/BSE stock models")
    parser.add_argument("--ticker", type=str, help="Train for specific ticker (e.g., RELIANCE.NS)")
    parser.add_argument("--global", dest="global_model", action="store_true",
                        help="Train global model (default)")
    args = parser.parse_args()

    init_db()

    if args.ticker:
        train_ticker_model(args.ticker)
    else:
        train_global_model()
