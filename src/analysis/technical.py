# src/analysis/technical.py
# ============================================================
# Technical Analysis: compute + interpret all indicators
# ============================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.processor import (
    compute_rsi, compute_macd, compute_bollinger,
    compute_atr, compute_vwap, compute_obv,
    compute_support_resistance
)
from config.settings import RSI_PERIOD


@dataclass
class TechnicalSignal:
    indicator: str
    value: float
    signal: str       # Bullish / Bearish / Neutral
    interpretation: str


class TechnicalAnalyzer:
    """Full technical analysis with human-readable interpretations."""

    def analyze(self, df: pd.DataFrame) -> dict:
        """Run full technical analysis on OHLCV DataFrame."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        signals = []
        summary = {}

        # --- RSI ---
        rsi = compute_rsi(close, RSI_PERIOD)
        rsi_val = float(rsi.iloc[-1])
        summary["rsi"] = round(rsi_val, 2)
        if rsi_val < 30:
            sig, interp = "Bullish", f"RSI {rsi_val:.1f} — Oversold zone. Potential reversal upward."
        elif rsi_val > 70:
            sig, interp = "Bearish", f"RSI {rsi_val:.1f} — Overbought zone. Potential pullback expected."
        else:
            sig, interp = "Neutral", f"RSI {rsi_val:.1f} — Neutral momentum zone."
        signals.append(TechnicalSignal("RSI", rsi_val, sig, interp))

        # --- MACD ---
        macd, macd_sig, macd_hist = compute_macd(close)
        macd_val = float(macd.iloc[-1])
        hist_val = float(macd_hist.iloc[-1])
        prev_hist = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0
        summary["macd"] = round(macd_val, 4)
        summary["macd_hist"] = round(hist_val, 4)
        if hist_val > 0 and hist_val > prev_hist:
            sig, interp = "Bullish", "MACD histogram rising above zero — bullish momentum building."
        elif hist_val < 0 and hist_val < prev_hist:
            sig, interp = "Bearish", "MACD histogram falling below zero — bearish pressure increasing."
        else:
            sig, interp = "Neutral", f"MACD showing mixed signals (hist={hist_val:.4f})."
        signals.append(TechnicalSignal("MACD", macd_val, sig, interp))

        # --- Bollinger Bands ---
        bb_upper, bb_mid, bb_lower = compute_bollinger(close)
        cur_close = float(close.iloc[-1])
        bb_u = float(bb_upper.iloc[-1])
        bb_l = float(bb_lower.iloc[-1])
        bb_m = float(bb_mid.iloc[-1])
        bb_pct = (cur_close - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5
        summary["bb_upper"] = round(bb_u, 2)
        summary["bb_lower"] = round(bb_l, 2)
        summary["bb_pct"] = round(bb_pct, 3)
        if cur_close > bb_u:
            sig, interp = "Bearish", f"Price (₹{cur_close:.0f}) broke above upper Bollinger Band — potential reversal."
        elif cur_close < bb_l:
            sig, interp = "Bullish", f"Price (₹{cur_close:.0f}) below lower Bollinger Band — oversold bounce likely."
        else:
            sig, interp = "Neutral", f"Price within Bollinger Bands. Position: {bb_pct*100:.0f}% of band."
        signals.append(TechnicalSignal("Bollinger Bands", bb_pct, sig, interp))

        # --- Moving Averages ---
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        summary["sma_20"] = round(sma20, 2)
        summary["sma_50"] = round(sma50, 2)
        summary["sma_200"] = round(sma200, 2)
        ma_score = sum([cur_close > sma20, cur_close > sma50, cur_close > sma200])
        if ma_score == 3:
            sig, interp = "Bullish", "Price above all SMAs (20/50/200) — strong bullish alignment."
        elif ma_score == 0:
            sig, interp = "Bearish", "Price below all SMAs — bearish trend in all timeframes."
        else:
            sig, interp = "Neutral", f"Price above {ma_score}/3 key moving averages."
        signals.append(TechnicalSignal("Moving Averages", float(ma_score), sig, interp))

        # Golden/Death Cross
        if sma50 > sma200:
            golden = True
            gc_interp = "Golden Cross active (SMA50 > SMA200) — long-term bullish signal."
            gc_sig = "Bullish"
        else:
            golden = False
            gc_interp = "Death Cross active (SMA50 < SMA200) — long-term bearish signal."
            gc_sig = "Bearish"
        summary["golden_cross"] = golden
        signals.append(TechnicalSignal("Golden/Death Cross", 1 if golden else 0, gc_sig, gc_interp))

        # --- ATR ---
        atr = float(compute_atr(high, low, close).iloc[-1])
        atr_pct = atr / cur_close * 100
        summary["atr"] = round(atr, 2)
        summary["atr_pct"] = round(atr_pct, 2)

        # --- VWAP ---
        vwap = float(compute_vwap(high, low, close, volume).iloc[-1])
        summary["vwap"] = round(vwap, 2)
        if cur_close > vwap:
            sig, interp = "Bullish", f"Price (₹{cur_close:.0f}) above VWAP (₹{vwap:.0f}) — intraday buyers in control."
        else:
            sig, interp = "Bearish", f"Price (₹{cur_close:.0f}) below VWAP (₹{vwap:.0f}) — sellers dominant."
        signals.append(TechnicalSignal("VWAP", vwap, sig, interp))

        # --- Volume ---
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        cur_vol = float(volume.iloc[-1])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1
        summary["volume_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 1.5:
            sig, interp = "Bullish", f"Volume {vol_ratio:.1f}x above average — strong institutional interest."
        elif vol_ratio < 0.7:
            sig, interp = "Bearish", f"Volume only {vol_ratio:.1f}x average — weak conviction in current move."
        else:
            sig, interp = "Neutral", f"Volume at {vol_ratio:.1f}x average — normal trading activity."
        signals.append(TechnicalSignal("Volume", vol_ratio, sig, interp))

        # --- Support / Resistance ---
        support = float(low.rolling(20).min().iloc[-1])
        resistance = float(high.rolling(20).max().iloc[-1])
        summary["support"] = round(support, 2)
        summary["resistance"] = round(resistance, 2)
        dist_support = (cur_close - support) / support * 100
        dist_resist = (resistance - cur_close) / cur_close * 100
        sr_interp = (
            f"Support: ₹{support:.0f} ({dist_support:.1f}% away) | "
            f"Resistance: ₹{resistance:.0f} ({dist_resist:.1f}% away)"
        )
        signals.append(TechnicalSignal("Support/Resistance", support, "Neutral", sr_interp))

        # --- Overall Score ---
        bull_count = sum(1 for s in signals if s.signal == "Bullish")
        bear_count = sum(1 for s in signals if s.signal == "Bearish")
        total = len(signals)
        bull_score = bull_count / total * 100

        if bull_score >= 60:
            overall = "Bullish"
        elif bull_score <= 40:
            overall = "Bearish"
        else:
            overall = "Neutral"

        return {
            "signals": [
                {"indicator": s.indicator, "value": s.value,
                 "signal": s.signal, "interpretation": s.interpretation}
                for s in signals
            ],
            "summary": summary,
            "overall": overall,
            "bullish_signals": bull_count,
            "bearish_signals": bear_count,
            "neutral_signals": total - bull_count - bear_count,
            "bull_score": round(bull_score, 1),
            "current_price": round(cur_close, 2),
            "rsi_series": rsi.dropna().tail(60).tolist(),
            "macd_series": macd.dropna().tail(60).tolist(),
            "macd_hist_series": macd_hist.dropna().tail(60).tolist(),
            "bb_upper_series": bb_upper.dropna().tail(60).tolist(),
            "bb_lower_series": bb_lower.dropna().tail(60).tolist(),
        }
