# src/analysis/risk.py
# ============================================================
# Risk Analysis: volatility, drawdown, VaR, Beta
# ============================================================

import numpy as np
import pandas as pd
from loguru import logger
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import HIGH_RISK_VOLATILITY, MEDIUM_RISK_VOLATILITY


class RiskAnalyzer:
    """Computes risk metrics for individual stocks."""

    def analyze(self, df: pd.DataFrame, market_df: pd.DataFrame = None) -> dict:
        """
        df: OHLCV DataFrame for the stock
        market_df: OHLCV for Nifty 50 (for Beta calculation) — optional
        """
        close = df["Close"] if "Close" in df.columns else df.iloc[:,3]
        returns = close.pct_change().dropna()

        # --- Annualized Volatility ---
        volatility = float(returns.std() * np.sqrt(252))
        volatility_30d = float(returns.tail(30).std() * np.sqrt(252))

        # --- Max Drawdown ---
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # --- Value at Risk (95% confidence) ---
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))

        # --- Sharpe Ratio (risk-free rate ~7% India) ---
        risk_free_daily = 0.07 / 252
        excess_returns = returns - risk_free_daily
        sharpe = float(excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0

        # --- Beta vs Market ---
        beta = 1.0  # Default
        if market_df is not None:
            try:
                market_returns = market_df["Close"].pct_change().dropna()
                aligned = pd.concat([returns, market_returns], axis=1).dropna()
                aligned.columns = ["stock", "market"]
                cov = aligned.cov().iloc[0, 1]
                market_var = aligned["market"].var()
                beta = cov / market_var if market_var > 0 else 1.0
            except Exception:
                pass

        # --- Calmar Ratio ---
        annual_return = float(returns.mean() * 252)
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # --- Risk Category ---
        if volatility > HIGH_RISK_VOLATILITY:
            risk_category = "High Risk"
            risk_emoji = "🔴"
            risk_description = "Highly volatile stock. Suitable only for aggressive investors with high risk tolerance."
        elif volatility > MEDIUM_RISK_VOLATILITY:
            risk_category = "Medium Risk"
            risk_emoji = "🟡"
            risk_description = "Moderate volatility. Suitable for balanced investors."
        else:
            risk_category = "Low Risk"
            risk_emoji = "🟢"
            risk_description = "Low volatility. Suitable for conservative investors."

        # --- Drawdown Risk ---
        if max_drawdown < -0.40:
            drawdown_risk = "High"
        elif max_drawdown < -0.20:
            drawdown_risk = "Medium"
        else:
            drawdown_risk = "Low"

        # --- Volatility Score (0-100) ---
        vol_score = min(int(volatility * 200), 100)

        # --- Current streak ---
        recent = close.tail(10)
        up_days = sum(recent.diff().dropna() > 0)
        down_days = sum(recent.diff().dropna() < 0)
        if up_days > down_days:
            momentum = "Positive"
        elif down_days > up_days:
            momentum = "Negative"
        else:
            momentum = "Sideways"

        return {
            "volatility_annual": round(volatility * 100, 2),
            "volatility_30d": round(volatility_30d * 100, 2),
            "volatility_score": vol_score,
            "max_drawdown": round(max_drawdown * 100, 2),
            "drawdown_risk": drawdown_risk,
            "var_95": round(var_95 * 100, 2),
            "var_99": round(var_99 * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "calmar_ratio": round(calmar, 3),
            "beta": round(beta, 3),
            "annual_return": round(annual_return * 100, 2),
            "risk_category": risk_category,
            "risk_emoji": risk_emoji,
            "risk_description": risk_description,
            "momentum": momentum,
            "up_days_10": up_days,
            "down_days_10": down_days,
            "returns_series": returns.tail(252).tolist(),
            "drawdown_series": drawdown.tail(252).tolist(),
        }

    def calculate_portfolio_risk(self, holdings: list, prices: dict) -> dict:
        """Calculate portfolio-level risk metrics."""
        total_value = 0
        weights = {}
        for h in holdings:
            ticker = h["ticker"]
            price = prices.get(ticker, h.get("avg_buy_price", 0))
            val = price * h["quantity"]
            total_value += val
            weights[ticker] = val

        if total_value == 0:
            return {"error": "Empty portfolio"}

        for ticker in weights:
            weights[ticker] /= total_value

        # Weighted average volatility as proxy
        avg_vol = 0
        for ticker, weight in weights.items():
            avg_vol += weight * 0.25  # Placeholder; real app loads per-stock vol

        risk_cat = "High Risk" if avg_vol > 0.35 else ("Medium Risk" if avg_vol > 0.20 else "Low Risk")

        return {
            "total_value": round(total_value, 2),
            "portfolio_volatility": round(avg_vol * 100, 1),
            "risk_category": risk_cat,
            "weights": {k: round(v * 100, 1) for k, v in weights.items()},
        }
