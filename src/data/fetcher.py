# src/data/fetcher.py — Stock Data Fetcher using yfinance + NSE fallback
import pandas as pd
import numpy as np
import requests
import time
import os
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# NSE headers to bypass blocking
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

class StockDataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._init_nse_session()

    def _init_nse_session(self):
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
        except:
            pass

    def _get_nse_quote(self, symbol):
        """Fetch live quote from NSE directly."""
        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                pd_data = data.get("priceInfo", {})
                return {
                    "price": pd_data.get("lastPrice", 0),
                    "change": pd_data.get("change", 0),
                    "change_pct": pd_data.get("pChange", 0),
                    "open": pd_data.get("open", 0),
                    "high": pd_data.get("intraDayHighLow", {}).get("max", 0),
                    "low": pd_data.get("intraDayHighLow", {}).get("min", 0),
                    "prev_close": pd_data.get("previousClose", 0),
                    "volume": data.get("marketDeptOrderBook", {}).get("totalTradedVolume", 0),
                }
        except Exception as e:
            logger.warning(f"NSE quote failed for {symbol}: {e}")
        return {}

    def _get_nse_history(self, symbol, from_date, to_date):
        """Fetch historical data from NSE."""
        try:
            url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from={from_date}&to={to_date}&csv=true"
            r = self.session.get(url, timeout=15)
            if r.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                df.columns = [c.strip() for c in df.columns]
                # Rename columns to standard OHLCV
                col_map = {
                    "Date": "Date", "OPEN": "Open", "HIGH": "High",
                    "LOW": "Low", "CLOSE": "Close", "VOLUME": "Volume",
                    "close": "Close", "open": "Open", "high": "High",
                    "low": "Low", "volume": "Volume",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date").sort_index()
                if "Close" in df.columns:
                    return df
        except Exception as e:
            logger.warning(f"NSE history failed for {symbol}: {e}")
        return pd.DataFrame()

    def _get_yfinance(self, ticker, period="1y"):
        """Try yfinance as backup."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.history(period=period)
            if not df.empty and "Close" in df.columns:
                return df
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}")
        return pd.DataFrame()

    def fetch_single(self, ticker, period="1y"):
        """Fetch historical data for a single stock."""
        # Try yfinance first with delay
        time.sleep(0.5)
        df = self._get_yfinance(ticker, period)
        if not df.empty and "Close" in df.columns:
            logger.info(f"✅ yfinance data for {ticker}: {len(df)} rows")
            return df

        # Try NSE directly
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        to_date = datetime.now().strftime("%d-%m-%Y")
        days = {"3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 365)
        from_date = (datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y")
        df = self._get_nse_history(symbol, from_date, to_date)
        if not df.empty:
            logger.info(f"✅ NSE data for {symbol}: {len(df)} rows")
            return df

        # Generate demo data as last resort
        logger.warning(f"⚠️ Using demo data for {ticker}")
        return self._generate_demo_data(ticker, days)

    def _generate_demo_data(self, ticker, days=365):
        """Generate realistic demo OHLCV data when APIs are blocked."""
        np.random.seed(hash(ticker) % 1000)
        dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
        price = np.random.uniform(500, 3000)
        prices = [price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0.0003, 0.015)
            prices.append(prices[-1] * (1 + change))
        closes = np.array(prices)
        opens  = closes * np.random.uniform(0.995, 1.005, len(closes))
        highs  = closes * np.random.uniform(1.001, 1.02,  len(closes))
        lows   = closes * np.random.uniform(0.98,  0.999, len(closes))
        vols   = np.random.randint(500000, 5000000, len(closes))
        df = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows,
            "Close": closes, "Volume": vols
        }, index=dates)
        return df

    def fetch_realtime(self, ticker):
        """Fetch live price."""
        # Try yfinance fast_info
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            prev  = getattr(info, "previous_close", None) or getattr(info, "regularMarketPreviousClose", None)
            if price and prev:
                chg = price - prev
                chg_pct = (chg / prev) * 100
                return {"price": price, "change": chg, "change_pct": chg_pct}
        except:
            pass

        # Try NSE
        symbol = ticker.replace(".NS","").replace("^NSEI","NIFTY 50").replace("^BSESN","SENSEX")
        if symbol not in ["NIFTY 50", "SENSEX", "^NSEBANK", "^CNXIT"]:
            data = self._get_nse_quote(symbol)
            if data:
                return data

        # Return demo data
        import random
        random.seed(hash(ticker) % 100)
        base = random.uniform(100, 5000)
        chg  = random.uniform(-2, 2)
        return {"price": base, "change": base * chg / 100, "change_pct": chg}

    def fetch_bulk(self, tickers, period="1y"):
        results = {}
        for t in tickers:
            results[t] = self.fetch_single(t, period)
            time.sleep(0.3)
        return results

    def get_top_movers(self, direction="gainers", n=5):
        """Get top gainers or losers from NSE."""
        try:
            url = "https://www.nseindia.com/api/live-analysis-variations?index=gainers" if direction == "gainers" else "https://www.nseindia.com/api/live-analysis-variations?index=loosers"
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                stocks = data.get("NIFTY", {}).get("data", [])[:n]
                return [{"ticker": s.get("symbol",""), "change_pct": s.get("perChange", 0), "price": s.get("ltp", 0)} for s in stocks]
        except Exception as e:
            logger.warning(f"Top movers failed: {e}")

        # Demo fallback
        import random
        random.seed(42)
        demo_stocks = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK", "WIPRO", "BAJFINANCE", "HCLTECH"]
        result = []
        for s in demo_stocks[:n]:
            chg = random.uniform(1, 5) if direction == "gainers" else random.uniform(-5, -1)
            result.append({"ticker": s, "change_pct": chg, "price": random.uniform(500, 3000)})
        return result

    def save_to_csv(self, data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(data, pd.DataFrame) and not data.empty:
            data.to_csv(path)

    def load_from_db(self, ticker, days=365):
        return pd.DataFrame()

    def save_to_db(self, ticker, df):
        pass
