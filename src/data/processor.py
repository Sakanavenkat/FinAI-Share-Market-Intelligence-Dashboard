# src/data/processor.py
# ============================================================
# Feature Engineering for ML models
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, SMA_PERIODS, EMA_PERIODS
)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger(series: pd.Series, period=20, std=2):
    sma = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, sma, lower


def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_vwap(high, low, close, volume):
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume
    return tp_vol.cumsum() / volume.cumsum()


def compute_obv(close, volume):
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def compute_support_resistance(df: pd.DataFrame, window: int = 20):
    support = df["Low"].rolling(window).min()
    resistance = df["High"].rolling(window).max()
    return support, resistance


class FeatureEngineer:
    """Computes all technical indicators and ML features."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature pipeline. Input: OHLCV DataFrame. Output: feature matrix."""
        df = df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # --- RSI ---
        df["rsi"] = compute_rsi(close, RSI_PERIOD)

        # --- MACD ---
        df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
            close, MACD_FAST, MACD_SLOW, MACD_SIGNAL
        )

        # --- Bollinger Bands ---
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = compute_bollinger(
            close, BB_PERIOD, BB_STD
        )
        df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # --- Moving Averages ---
        for period in SMA_PERIODS:
            df[f"sma_{period}"] = close.rolling(period).mean()
        for period in EMA_PERIODS:
            df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # --- Price vs MA signals ---
        df["price_above_sma20"] = (close > df["sma_20"]).astype(int)
        df["price_above_sma50"] = (close > df["sma_50"]).astype(int)
        df["price_above_sma200"] = (close > df["sma_200"]).astype(int)
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

        # --- ATR ---
        df["atr"] = compute_atr(high, low, close)
        df["atr_pct"] = df["atr"] / close

        # --- VWAP ---
        df["vwap"] = compute_vwap(high, low, close, volume)
        df["price_above_vwap"] = (close > df["vwap"]).astype(int)

        # --- OBV ---
        df["obv"] = compute_obv(close, volume)
        df["obv_change"] = df["obv"].pct_change()

        # --- Support / Resistance ---
        df["support"], df["resistance"] = compute_support_resistance(df)
        df["near_support"] = ((close - df["support"]) / close < 0.02).astype(int)
        df["near_resistance"] = ((df["resistance"] - close) / close < 0.02).astype(int)

        # --- Returns ---
        for n in [1, 2, 3, 5, 10, 20]:
            df[f"return_{n}d"] = close.pct_change(n)

        # --- Volatility ---
        df["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)
        df["volatility_5d"] = close.pct_change().rolling(5).std() * np.sqrt(252)

        # --- Volume features ---
        df["volume_sma20"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_sma20"]
        df["high_volume"] = (df["volume_ratio"] > 1.5).astype(int)

        # --- Candlestick patterns ---
        df["body"] = abs(close - df["Open"])
        df["upper_shadow"] = high - pd.concat([close, df["Open"]], axis=1).max(axis=1)
        df["lower_shadow"] = pd.concat([close, df["Open"]], axis=1).min(axis=1) - low
        df["doji"] = (df["body"] < df["atr"] * 0.1).astype(int)
        df["hammer"] = (
            (df["lower_shadow"] > 2 * df["body"]) &
            (df["upper_shadow"] < df["body"] * 0.5)
        ).astype(int)

        # --- Price change from open ---
        df["intraday_change"] = (close - df["Open"]) / df["Open"]

        # --- Target variable: will price increase tomorrow? ---
        df["target"] = (close.shift(-1) > close).astype(int)
        df["target_return"] = close.shift(-1) / close - 1

        df.dropna(inplace=True)
        return df

    def get_feature_columns(self) -> list:
        """Return list of feature column names used for ML."""
        return [
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_pct", "bb_width",
            "sma_20", "sma_50", "sma_200",
            "ema_9", "ema_21",
            "price_above_sma20", "price_above_sma50", "price_above_sma200",
            "golden_cross",
            "atr_pct", "price_above_vwap",
            "obv_change",
            "near_support", "near_resistance",
            "return_1d", "return_2d", "return_3d", "return_5d", "return_10d",
            "volatility_20d", "volatility_5d",
            "volume_ratio", "high_volume",
            "doji", "hammer",
            "intraday_change",
        ]
