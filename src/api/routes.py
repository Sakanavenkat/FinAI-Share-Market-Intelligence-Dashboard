# src/api/routes.py
# ============================================================
# FastAPI REST API — all endpoints
# ============================================================

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.fetcher import StockDataFetcher
from src.data.processor import FeatureEngineer
from src.ml.predictor import StockPredictor
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.risk import RiskAnalyzer
from src.analysis.recommendation import RecommendationEngine
from src.chatbot.agent import FinancialChatbot
from src.data.database import (
    SessionLocal, PortfolioHolding, Recommendation as RecDB,
    Prediction, init_db
)
from config.stocks_list import ALL_NSE_STOCKS, get_company_name, COMPANY_NAMES

# Initialize
init_db()
app = FastAPI(
    title="NSE/BSE Stock Intelligence API",
    description="AI-powered Indian stock market analysis, predictions, and recommendations",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

fetcher = StockDataFetcher()
engineer = FeatureEngineer()
predictor = StockPredictor()
tech_analyzer = TechnicalAnalyzer()
sent_analyzer = SentimentAnalyzer()
risk_analyzer = RiskAnalyzer()
rec_engine = RecommendationEngine()
chatbot = FinancialChatbot()


# ====== Pydantic Models ======

class ChatRequest(BaseModel):
    message: str
    ticker: Optional[str] = None
    context: Optional[dict] = None

class PortfolioAdd(BaseModel):
    ticker: str
    quantity: float
    avg_buy_price: float


# ====== Utility ======

def get_ticker(symbol: str) -> str:
    """Ensure ticker has .NS suffix."""
    symbol = symbol.upper().strip()
    if not symbol.endswith((".NS", ".BO")):
        return f"{symbol}.NS"
    return symbol


# ====== Endpoints ======

@app.get("/")
def root():
    return {"message": "NSE/BSE Stock Intelligence API", "status": "running", "version": "1.0.0"}


@app.get("/api/stocks/search")
def search_stocks(q: str = Query(..., min_length=1)):
    """Search companies by name or symbol."""
    q_lower = q.lower()
    results = []
    for symbol, name in COMPANY_NAMES.items():
        if q_lower in symbol.lower() or q_lower in name.lower():
            results.append({"symbol": symbol, "ticker": f"{symbol}.NS", "name": name})
    return {"results": results[:20], "query": q}


@app.get("/api/stocks/{ticker}/realtime")
def get_realtime(ticker: str):
    """Get current price and metadata."""
    t = get_ticker(ticker)
    data = fetcher.fetch_realtime(t)
    if not data:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    return data


@app.get("/api/stocks/{ticker}/history")
def get_history(ticker: str, period: str = "1y"):
    """Get historical OHLCV data."""
    t = get_ticker(ticker)
    df = fetcher.fetch_single(t, period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail="No history found")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return {"ticker": t, "period": period, "data": df.to_dict(orient="records")}


@app.get("/api/stocks/{ticker}/predict")
def predict_stock(ticker: str):
    """AI prediction for tomorrow's movement."""
    t = get_ticker(ticker)
    df = fetcher.fetch_single(t, period="1y")
    if df.empty:
        raise HTTPException(status_code=404, detail="Insufficient data")
    pred = predictor.predict_tomorrow(t, df)
    if "error" in pred:
        raise HTTPException(status_code=500, detail=pred["error"])
    return pred


@app.get("/api/stocks/{ticker}/forecast")
def forecast_stock(ticker: str):
    """Multi-horizon forecast (1d, 7d, 30d, 90d)."""
    t = get_ticker(ticker)
    df = fetcher.fetch_single(t, period="2y")
    if df.empty:
        raise HTTPException(status_code=404, detail="Insufficient data")
    return predictor.forecast_multi_horizon(t, df)


@app.get("/api/stocks/{ticker}/technical")
def get_technical(ticker: str):
    """Full technical analysis."""
    t = get_ticker(ticker)
    df = fetcher.fetch_single(t, period="1y")
    if df.empty:
        raise HTTPException(status_code=404, detail="No data")
    return tech_analyzer.analyze(df)


@app.get("/api/stocks/{ticker}/sentiment")
def get_sentiment(ticker: str):
    """News sentiment analysis."""
    t = get_ticker(ticker)
    return sent_analyzer.get_stock_sentiment(t)


@app.get("/api/stocks/{ticker}/risk")
def get_risk(ticker: str):
    """Risk analysis."""
    t = get_ticker(ticker)
    df = fetcher.fetch_single(t, period="2y")
    if df.empty:
        raise HTTPException(status_code=404, detail="No data")
    return risk_analyzer.analyze(df)


@app.get("/api/stocks/{ticker}/recommend")
def get_recommendation(ticker: str):
    """Full BUY/SELL/HOLD recommendation."""
    t = get_ticker(ticker)
    df = fetcher.fetch_single(t, period="2y")
    if df.empty:
        raise HTTPException(status_code=404, detail="No data")

    features_df = engineer.build_features(df)
    pred = predictor.predict_tomorrow(t, df)
    tech = tech_analyzer.analyze(df)
    sent = sent_analyzer.get_stock_sentiment(t)
    risk = risk_analyzer.analyze(df)
    rec = rec_engine.generate(t, pred, tech, sent, risk)
    return rec


@app.get("/api/stocks/{ticker}/full")
def get_full_analysis(ticker: str):
    """Complete analysis bundle for dashboard."""
    t = get_ticker(ticker)
    df = fetcher.fetch_single(t, period="2y")
    if df.empty:
        raise HTTPException(status_code=404, detail="No data")

    realtime = fetcher.fetch_realtime(t)
    pred = predictor.predict_tomorrow(t, df)
    forecast = predictor.forecast_multi_horizon(t, df)
    tech = tech_analyzer.analyze(df)
    sent = sent_analyzer.get_stock_sentiment(t)
    risk = risk_analyzer.analyze(df)
    rec = rec_engine.generate(t, pred, tech, sent, risk)

    return {
        "ticker": t,
        "company_name": get_company_name(t),
        "realtime": realtime,
        "prediction": pred,
        "forecast": forecast,
        "technical": tech,
        "sentiment": sent,
        "risk": risk,
        "recommendation": rec,
        "generated_at": datetime.now().isoformat(),
    }


@app.get("/api/market/top-picks")
def get_top_picks(n: int = 10):
    """Top predicted risers and fallers for tomorrow."""
    from config.stocks_list import NIFTY50
    tickers = [f"{s}.NS" for s in NIFTY50[:30]]
    return predictor.get_top_predictions(tickers, top_n=n)


@app.get("/api/market/movers")
def get_movers():
    """Top gainers and losers today."""
    return fetcher.get_top_movers(n=10)


@app.get("/api/market/sectors")
def get_sectors():
    """Sector performance overview."""
    from config.settings import SECTORS
    sector_data = {}
    for sector, stocks in SECTORS.items():
        changes = []
        for sym in stocks[:3]:
            data = fetcher.fetch_realtime(f"{sym}.NS")
            if data and "change_pct" in data:
                changes.append(data["change_pct"])
        avg_change = sum(changes) / len(changes) if changes else 0
        sector_data[sector] = {
            "avg_change_pct": round(avg_change, 2),
            "trend": "Bullish" if avg_change > 0 else ("Bearish" if avg_change < 0 else "Neutral"),
            "stocks": stocks[:5],
        }
    return sector_data


@app.post("/api/chat")
def chat(req: ChatRequest):
    """AI chatbot endpoint."""
    context = req.context or {}
    if req.ticker:
        context["ticker"] = get_ticker(req.ticker)
    reply = chatbot.chat(req.message, context)
    return {"reply": reply, "timestamp": datetime.now().isoformat()}


@app.post("/api/portfolio")
def add_holding(holding: PortfolioAdd):
    """Add stock to portfolio."""
    db = SessionLocal()
    try:
        t = get_ticker(holding.ticker)
        record = PortfolioHolding(
            ticker=t,
            company_name=get_company_name(t),
            quantity=holding.quantity,
            avg_buy_price=holding.avg_buy_price,
        )
        db.add(record)
        db.commit()
        return {"message": "Added to portfolio", "ticker": t}
    finally:
        db.close()


@app.get("/api/portfolio")
def get_portfolio():
    """Get all portfolio holdings with current P&L."""
    db = SessionLocal()
    try:
        holdings = db.query(PortfolioHolding).all()
        result = []
        total_invested = 0
        total_current = 0
        for h in holdings:
            rt = fetcher.fetch_realtime(h.ticker)
            cur_price = rt.get("current_price", h.avg_buy_price) if rt else h.avg_buy_price
            invested = h.avg_buy_price * h.quantity
            current_val = cur_price * h.quantity
            pnl = current_val - invested
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0
            total_invested += invested
            total_current += current_val
            result.append({
                "ticker": h.ticker,
                "company_name": h.company_name,
                "quantity": h.quantity,
                "avg_buy_price": h.avg_buy_price,
                "current_price": cur_price,
                "invested_value": round(invested, 2),
                "current_value": round(current_val, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "change_pct": rt.get("change_pct", 0) if rt else 0,
            })
        return {
            "holdings": result,
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current, 2),
            "total_pnl": round(total_current - total_invested, 2),
            "total_pnl_pct": round((total_current / total_invested - 1) * 100, 2) if total_invested > 0 else 0,
        }
    finally:
        db.close()
