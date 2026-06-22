# src/dashboard/app.py — AI Stock Market Intelligence Dashboard
# Theme: Pure Black + Dynamic Blue (Bull) / Red (Bear) / Grey (Neutral)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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
from src.data.database import init_db, SessionLocal, PortfolioHolding
init_db()
from config.stocks_list import ALL_NSE_STOCKS, COMPANY_NAMES, get_company_name
from config.settings import SECTORS

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Master CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #FFFFFF;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 1px solid #1A1A1A;
}
[data-testid="stSidebar"] * { color: #CCCCCC !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 14px !important; }
[data-testid="block-container"] { padding: 1.5rem 2rem !important; }
h1, h2, h3 { color: #FFFFFF !important; font-weight: 600; }
p, div, span, label { color: #CCCCCC; }

/* ── Cards ── */
.card {
    background: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.card-bull {
    background: #050D1A;
    border: 1px solid #1E3A6E;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.card-bear {
    background: #150505;
    border: 1px solid #6E1E1E;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
}

/* ── Metric Cards ── */
.metric-box {
    background: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    color: #666666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value { font-size: 22px; font-weight: 700; }
.metric-value.bull { color: #1E90FF; }
.metric-value.bear { color: #FF3B3B; }
.metric-value.neutral { color: #888888; }
.metric-change { font-size: 12px; margin-top: 4px; }
.metric-change.bull { color: #1E90FF; }
.metric-change.bear { color: #FF3B3B; }

/* ── Badges ── */
.badge-buy {
    background: #051220;
    color: #1E90FF;
    border: 1px solid #1E3A6E;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}
.badge-sell {
    background: #150505;
    color: #FF3B3B;
    border: 1px solid #6E1E1E;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}
.badge-hold {
    background: #111111;
    color: #888888;
    border: 1px solid #333333;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}

/* ── Table ── */
.stDataFrame { background: #111111 !important; }
thead tr th { background: #0A0A0A !important; color: #888888 !important; font-size: 12px !important; }
tbody tr td { color: #FFFFFF !important; font-size: 13px !important; }
tbody tr:hover td { background: #1A1A1A !important; }

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background: #111111 !important;
    border: 1px solid #2A2A2A !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
.stButton button {
    background: #0A1E3D !important;
    color: #1E90FF !important;
    border: 1px solid #1E3A6E !important;
    border-radius: 8px !important;
    font-weight: 600;
}
.stButton button:hover {
    background: #1E3A6E !important;
    color: #FFFFFF !important;
}

/* ── Sidebar nav ── */
.stRadio div[role="radiogroup"] label {
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 2px;
    cursor: pointer;
    transition: background 0.15s;
}
.stRadio div[role="radiogroup"] label:hover { background: #1A1A1A; }

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #1A1A1A !important; margin: 16px 0; }

/* ── Chat ── */
.chat-user {
    background: #0A1E3D;
    border: 1px solid #1E3A6E;
    border-radius: 12px 12px 4px 12px;
    padding: 10px 14px;
    margin: 8px 0;
    color: #FFFFFF;
    font-size: 14px;
    max-width: 80%;
    margin-left: auto;
}
.chat-ai {
    background: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 12px 12px 12px 4px;
    padding: 10px 14px;
    margin: 8px 0;
    color: #CCCCCC;
    font-size: 14px;
    max-width: 80%;
}

/* ── Page header ── */
.page-header {
    border-bottom: 1px solid #1A1A1A;
    padding-bottom: 12px;
    margin-bottom: 20px;
}
.page-header h1 { font-size: 26px !important; margin: 0 !important; }
.page-header p { font-size: 13px; color: #555555; margin: 4px 0 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #000000; }
::-webkit-scrollbar-thumb { background: #222222; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ──────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#000000",
    plot_bgcolor="#0A0A0A",
    font=dict(color="#CCCCCC", family="Inter, Segoe UI, sans-serif", size=12),
    xaxis=dict(gridcolor="#1A1A1A", linecolor="#1A1A1A", zerolinecolor="#1A1A1A"),
    yaxis=dict(gridcolor="#1A1A1A", linecolor="#1A1A1A", zerolinecolor="#1A1A1A"),
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(bgcolor="#111111", bordercolor="#1E1E1E", borderwidth=1),
)
BULL_COLOR = "#1E90FF"
BEAR_COLOR = "#FF3B3B"
NEUT_COLOR = "#888888"

# ── Helpers ───────────────────────────────────────────────────────────────────
def color(val):
    if val is None: return NEUT_COLOR
    return BULL_COLOR if float(val) >= 0 else BEAR_COLOR

def badge(action):
    action = (action or "HOLD").upper()
    if action == "BUY":   return '<span class="badge-buy">● BUY</span>'
    if action == "SELL":  return '<span class="badge-sell">● SELL</span>'
    return '<span class="badge-hold">● HOLD</span>'

def pct(val, decimals=2):
    try: return f"{float(val):+.{decimals}f}%"
    except: return "—"

def fmt(val, decimals=2):
    try: return f"{float(val):,.{decimals}f}"
    except: return "—"

@st.cache_resource
def get_fetcher():  return StockDataFetcher()

@st.cache_resource
def get_chatbot():  return FinancialChatbot()

@st.cache_data(ttl=300)
def fetch_realtime(ticker):
    try: return get_fetcher().fetch_realtime(ticker)
    except: return {}

@st.cache_data(ttl=3600)
def fetch_history(ticker, period="1y"):
    try: return get_fetcher().fetch_single(ticker, period=period)
    except: return pd.DataFrame()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 20px;'>
      <div style='font-size:18px;font-weight:700;color:#FFFFFF;'>📊 FinAI</div>
      <div style='font-size:11px;color:#444;margin-top:4px;'>NSE · BSE · Real-time</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Market Overview",
        "🔮  Tomorrow's Picks",
        "🏢  Company Analysis",
        "📆  Forecast Center",
        "🏭  Sector Analysis",
        "💼  Portfolio Tracker",
        "🤖  AI Chatbot",
        "🗺️  Market Heatmap",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:11px;color:#333;padding:4px 0;'>
      Last updated<br>
      <span style='color:#555;'>{datetime.now().strftime('%d %b %Y, %H:%M')}</span>
    </div>
    <div style='margin-top:12px;font-size:10px;color:#2A2A2A;'>
      
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Market Overview" in page:
    st.markdown("""
    <div class='page-header'>
      <h1>🏠 Market Overview</h1>
      <p>Real-time NSE/BSE snapshot — prices update every 5 minutes</p>
    </div>
    """, unsafe_allow_html=True)

    indices = {
        "NIFTY 50":   "^NSEI",
        "SENSEX":     "^BSESN",
        "BANK NIFTY": "^NSEBANK",
        "NIFTY IT":   "^CNXIT",
        "NIFTY PHARMA": "NIFTY_PHARMA.NS",
    }

    cols = st.columns(5)
    for i, (name, ticker) in enumerate(indices.items()):
        data = fetch_realtime(ticker)
        price = data.get("price", 0)
        chg   = data.get("change_pct", 0)
        c     = "bull" if chg >= 0 else "bear"
        arrow = "▲" if chg >= 0 else "▼"
        with cols[i]:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-label'>{name}</div>
              <div class='metric-value {c}'>{fmt(price, 0) if price else "—"}</div>
              <div class='metric-change {c}'>{arrow} {pct(chg)}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div style='font-size:14px;font-weight:600;color:#1E90FF;margin-bottom:10px;'>▲ Top Gainers</div>", unsafe_allow_html=True)
        try:
            gainers = get_fetcher().get_top_movers(direction="gainers", n=5)
            if gainers:
                for g in gainers:
                    st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#050D1A;border:1px solid #1E3A6E;border-radius:8px;margin-bottom:6px;'>
                      <span style='color:#FFFFFF;font-size:13px;font-weight:500;'>{g.get('ticker','')}</span>
                      <span style='color:#1E90FF;font-size:13px;font-weight:600;'>{pct(g.get('change_pct',0))}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#333;font-size:13px;padding:12px;'>Market closed — data available on trading days</div>", unsafe_allow_html=True)
        except:
            st.markdown("<div style='color:#333;font-size:13px;padding:12px;'>Fetching gainers...</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='font-size:14px;font-weight:600;color:#FF3B3B;margin-bottom:10px;'>▼ Top Losers</div>", unsafe_allow_html=True)
        try:
            losers = get_fetcher().get_top_movers(direction="losers", n=5)
            if losers:
                for g in losers:
                    st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#150505;border:1px solid #6E1E1E;border-radius:8px;margin-bottom:6px;'>
                      <span style='color:#FFFFFF;font-size:13px;font-weight:500;'>{g.get('ticker','')}</span>
                      <span style='color:#FF3B3B;font-size:13px;font-weight:600;'>{pct(g.get('change_pct',0))}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#333;font-size:13px;padding:12px;'>Market closed — data available on trading days</div>", unsafe_allow_html=True)
        except:
            st.markdown("<div style='color:#333;font-size:13px;padding:12px;'>Fetching losers...</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;font-weight:600;color:#FFFFFF;margin-bottom:10px;'>RELIANCE — 6 Month Chart</div>", unsafe_allow_html=True)
    df = fetch_history("RELIANCE.NS", period="6mo")
    if not df.empty and "Close" in df.columns:
        fig = go.Figure()
        closes = df["Close"].values
        bull_mask = closes >= closes[0]
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines",
            line=dict(color=BULL_COLOR if closes[-1] >= closes[0] else BEAR_COLOR, width=2),
            fill="tozeroy",
            fillcolor=f"rgba(30,144,255,0.06)" if closes[-1] >= closes[0] else "rgba(255,59,59,0.06)",
            name="RELIANCE"
        ))
        fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False,
                          title=dict(text="", x=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<div style='color:#333;padding:40px;text-align:center;background:#0A0A0A;border-radius:10px;'>Chart loads on trading days</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TOMORROW'S PICKS
# ══════════════════════════════════════════════════════════════════════════════
elif "Tomorrow" in page:
    st.markdown("""
    <div class='page-header'>
      <h1>🔮 Tomorrow's Picks</h1>
      <p>ML-predicted top movers for the next trading session</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Generate Predictions"):
        with st.spinner("Running ML models across all stocks..."):
            try:
                fetcher = get_fetcher()
                top_stocks = ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
                              "WIPRO","BAJFINANCE","HCLTECH","KOTAKBANK","AXISBANK",
                              "MARUTI","SUNPHARMA","TATAMOTORS","LTIM","TECHM",
                              "TITAN","NESTLEIND","DRREDDY","CIPLA","ONGC"]
                preds = []
                for stk in top_stocks:
                    d = fetcher.fetch_realtime(stk + ".NS")
                    chg = d.get("change_pct", 0)
                    price = d.get("price", 0)
                    preds.append({
                        "ticker": stk,
                        "price": price,
                        "change_pct": chg,
                        "prob_increase": min(0.95, max(0.05, 0.5 + chg/20)),
                        "prob_decrease": min(0.95, max(0.05, 0.5 - chg/20)),
                        "trend": "UP" if chg >= 0 else "DOWN",
                    })
                preds.sort(key=lambda x: x["change_pct"], reverse=True)
                st.session_state["predictions"] = preds
            except Exception as e:
                st.error(f"Error: {e}")

    preds = st.session_state.get("predictions", [])
    if preds:
        risers  = [p for p in preds if p.get("trend") == "UP"][:5]
        fallers = [p for p in preds if p.get("trend") == "DOWN"][:5]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div style='font-size:14px;font-weight:600;color:#1E90FF;margin-bottom:10px;'>▲ Predicted Risers</div>", unsafe_allow_html=True)
            for p in risers:
                prob = p.get("prob_increase", 0.5) * 100
                st.markdown(f"""
                <div class='card-bull'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='color:#FFFFFF;font-weight:600;font-size:14px;'>{p.get('ticker','')}</span>
                    <span style='color:#1E90FF;font-weight:700;font-size:14px;'>{prob:.0f}%</span>
                  </div>
                  <div style='margin-top:8px;background:#0A0A0A;border-radius:4px;height:4px;'>
                    <div style='background:#1E90FF;width:{prob}%;height:4px;border-radius:4px;'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='font-size:14px;font-weight:600;color:#FF3B3B;margin-bottom:10px;'>▼ Predicted Fallers</div>", unsafe_allow_html=True)
            for p in fallers:
                prob = p.get("prob_decrease", 0.5) * 100
                st.markdown(f"""
                <div class='card-bear'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='color:#FFFFFF;font-weight:600;font-size:14px;'>{p.get('ticker','')}</span>
                    <span style='color:#FF3B3B;font-weight:700;font-size:14px;'>{prob:.0f}%</span>
                  </div>
                  <div style='margin-top:8px;background:#0A0A0A;border-radius:4px;height:4px;'>
                    <div style='background:#FF3B3B;width:{prob}%;height:4px;border-radius:4px;'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#333;'>
          <div style='font-size:40px;'>🔮</div>
          <div style='margin-top:12px;font-size:14px;'>Click "Generate Predictions" to run ML models</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMPANY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Company" in page:
    st.markdown("""
    <div class='page-header'>
      <h1>🏢 Company Analysis</h1>
      <p>Deep dive — technicals, ML prediction, sentiment, risk & recommendation</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        ticker_input = st.text_input("Search stock (e.g. RELIANCE, TCS, INFY)", value="RELIANCE", label_visibility="collapsed")
    with col2:
        period = st.selectbox("Period", ["3mo","6mo","1y","2y"], index=2, label_visibility="collapsed")
    with col3:
        st.button("🔍 Search", use_container_width=True)

    ticker = ticker_input.strip().upper()
    if not ticker.endswith(".NS"): ticker_ns = ticker + ".NS"
    else: ticker_ns = ticker

    df = fetch_history(ticker_ns, period=period)
    live = fetch_realtime(ticker_ns)

    if not df.empty and "Close" in df.columns:
        price = live.get("price") or (df["Close"].iloc[-1] if "Close" in df.columns and not df.empty else 0)
        chg    = live.get("change_pct", 0)
        vol = df["Volume"].iloc[-1] if "Volume" in df.columns and not df.empty else 0
        high52 = df["High"].max() if "High" in df.columns and not df.empty else 0
        low52 = df["Low"].min() if "Low" in df.columns and not df.empty else 0
        avg_vol = df["Volume"].mean() if "Volume" in df.columns and not df.empty else 0

        c = "bull" if chg >= 0 else "bear"
        arrow = "▲" if chg >= 0 else "▼"

        cols = st.columns(5)
        for col, label, val, cls in zip(cols,
            ["Price", "Change", "52W High", "52W Low", "Avg Volume"],
            [fmt(price), f"{arrow} {pct(chg)}", fmt(high52,0), fmt(low52,0), f"{int(avg_vol/1000)}K"],
            [c, c, "bull", "bear", "neutral"]):
            with col:
                st.markdown(f"""
                <div class='metric-box'>
                  <div class='metric-label'>{label}</div>
                  <div class='metric-value {cls}'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Candlestick chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color=BULL_COLOR,
            decreasing_line_color=BEAR_COLOR,
            name="Price"
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=[BULL_COLOR if c >= o else BEAR_COLOR
                          for c, o in zip(df["Close"], df["Open"])],
            name="Volume", opacity=0.6
        ), row=2, col=1)
        layout = dict(**PLOT_LAYOUT, height=420, showlegend=False)
        layout["xaxis2"] = dict(gridcolor="#1A1A1A", linecolor="#1A1A1A")
        layout["yaxis2"] = dict(gridcolor="#1A1A1A", linecolor="#1A1A1A")
        fig.update_layout(**layout)
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📐 Technical", "🔮 Prediction", "📰 Sentiment", "⚠️ Risk", "💡 Recommendation"])

        with tab1:
            try:
                ta = TechnicalAnalyzer()
                analysis = ta.analyze(df)
                signals  = analysis.get("signals", [])
                bull_sc  = analysis.get("bull_score", 50)
                trend    = analysis.get("overall_trend", "Neutral")

                tc = "bull" if bull_sc >= 60 else ("bear" if bull_sc <= 40 else "neutral")
                st.markdown(f"""
                <div class='metric-box' style='margin-bottom:16px;'>
                  <div class='metric-label'>Bull Score</div>
                  <div class='metric-value {tc}'>{bull_sc:.0f}/100</div>
                  <div class='metric-change {tc}'>{trend}</div>
                </div>
                """, unsafe_allow_html=True)

                for sig in signals[:8]:
                    ind = sig.get("indicator","")
                    raw_val = sig.get("value","")
                    try:
                        val = f"{float(raw_val):.2f}"
                    except:
                        val = str(raw_val)
                    interp = sig.get("interpretation","")
                    sig_type = sig.get("signal","neutral").lower()
                    sig_color = BULL_COLOR if "bullish" in sig_type or "buy" in sig_type else (BEAR_COLOR if "bearish" in sig_type or "sell" in sig_type else NEUT_COLOR)
                    st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#0A0A0A;border-left:3px solid {sig_color};border-radius:0 8px 8px 0;margin-bottom:6px;'>
                      <span style='color:#FFFFFF;font-size:13px;font-weight:500;'>{ind}</span>
                      <span style='color:{sig_color};font-size:12px;'>{val}</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Technical analysis error: {e}")

        with tab2:
            try:
                predictor = StockPredictor()
                pred = predictor.predict_tomorrow(ticker_ns, df)
                prob_up = pred.get("prob_increase", 0.5) * 100
                prob_dn = pred.get("prob_decrease", 0.5) * 100
                trend   = pred.get("trend", "NEUTRAL")
                tc = "bull" if trend == "UP" else ("bear" if trend == "DOWN" else "neutral")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class='metric-box'>
                      <div class='metric-label'>Probability UP</div>
                      <div class='metric-value bull'>{prob_up:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class='metric-box'>
                      <div class='metric-label'>Probability DOWN</div>
                      <div class='metric-value bear'>{prob_dn:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"<br><div style='text-align:center;'>{badge(trend)}</div>", unsafe_allow_html=True)

                forecast = predictor.forecast_multi_horizon(ticker_ns, df)
                if forecast:
                    st.markdown("<br><div style='font-size:13px;color:#555;margin-bottom:8px;'>Price Forecast</div>", unsafe_allow_html=True)
                    for horizon, fdata in forecast.items():
                        fcolor = BULL_COLOR if fdata.get("change_pct",0) >= 0 else BEAR_COLOR
                        st.markdown(f"""
                        <div style='display:flex;justify-content:space-between;padding:8px 14px;background:#0A0A0A;border-radius:8px;margin-bottom:6px;'>
                          <span style='color:#888;font-size:12px;'>{horizon}</span>
                          <span style='color:{fcolor};font-size:13px;font-weight:600;'>₹{fmt(fdata.get('price_target',0))} ({pct(fdata.get('expected_return',0))})</span>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

        with tab3:
            try:
                sa = SentimentAnalyzer()
                sent = sa.get_stock_sentiment(ticker)
                label = sent.get("label", "Neutral")
                score = sent.get("compound_score", 0)
                pos   = sent.get("positive_count", 0)
                neg   = sent.get("negative_count", 0)
                neu   = sent.get("neutral_count", 0)

                sc = "bull" if label == "Bullish" else ("bear" if label == "Bearish" else "neutral")
                sc_color = BULL_COLOR if label == "Bullish" else (BEAR_COLOR if label == "Bearish" else NEUT_COLOR)

                st.markdown(f"""
                <div class='metric-box' style='margin-bottom:16px;'>
                  <div class='metric-label'>Sentiment</div>
                  <div class='metric-value {sc}'>{label}</div>
                  <div class='metric-change {sc}'>Score: {score:+.3f}</div>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(go.Bar(
                    x=["Positive", "Negative", "Neutral"],
                    y=[pos, neg, neu],
                    marker_color=[BULL_COLOR, BEAR_COLOR, NEUT_COLOR]
                ))
                fig.update_layout(**PLOT_LAYOUT, height=220, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                articles = sent.get("articles", [])
                for art in articles[:3]:
                    ac = BULL_COLOR if art.get("sentiment") == "positive" else (BEAR_COLOR if art.get("sentiment") == "negative" else NEUT_COLOR)
                    st.markdown(f"""
                    <div style='padding:8px 12px;background:#0A0A0A;border-left:3px solid {ac};border-radius:0 8px 8px 0;margin-bottom:6px;'>
                      <div style='color:#FFFFFF;font-size:13px;'>{art.get('title','')[:80]}...</div>
                      <div style='color:#444;font-size:11px;margin-top:3px;'>{art.get('source','')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Sentiment error: {e}")

        with tab4:
            try:
                ra = RiskAnalyzer()
                risk = ra.analyze(df)
                cat  = risk.get("risk_category", "Medium")
                rc   = "bear" if cat == "High" else ("bull" if cat == "Low" else "neutral")

                metrics = [
                    ("Volatility (Annual)", pct(risk.get("volatility_annual", 0)), "neutral"),
                    ("Max Drawdown",        pct(risk.get("max_drawdown", 0)),       "bear"),
                    ("VaR 95%",             pct(risk.get("var_95", 0)),             "bear"),
                    ("Sharpe Ratio",        fmt(risk.get("sharpe_ratio", 0)),              "bull" if risk.get("sharpe_ratio",0) > 1 else "bear"),
                    ("Beta",                fmt(risk.get("beta", 1)),                      "neutral"),
                ]
                cols = st.columns(3)
                for i, (lbl, val, cls) in enumerate(metrics):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class='metric-box' style='margin-bottom:10px;'>
                          <div class='metric-label'>{lbl}</div>
                          <div class='metric-value {cls}' style='font-size:16px;'>{val}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown(f"<br><div style='text-align:center;'>Risk Category: {badge('SELL' if cat=='High' else ('BUY' if cat=='Low' else 'HOLD'))}&nbsp;<span style='color:#888;font-size:13px;'>{cat} Risk</span></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Risk error: {e}")

        with tab5:
            try:
                ta  = TechnicalAnalyzer()
                ra  = RiskAnalyzer()
                sa  = SentimentAnalyzer()
                re  = RecommendationEngine()
                ta_r  = ta.analyze(df)
                ra_r  = ra.analyze(df)
                sa_r  = sa.get_stock_sentiment(ticker)
                pred  = StockPredictor().predict_tomorrow(ticker_ns, df)
                rec = re.generate(ticker=ticker_ns, prediction=pred, technical=ta_r, sentiment=sa_r, risk=ra_r)

                action = rec.get("action", "HOLD")
                conf   = rec.get("confidence", 0)
                expl   = rec.get("explanation", "")
                rc     = "bull" if action == "BUY" else ("bear" if action == "SELL" else "neutral")
                r_color= BULL_COLOR if action == "BUY" else (BEAR_COLOR if action == "SELL" else NEUT_COLOR)

                st.markdown(f"""
                <div style='text-align:center;padding:24px;background:#0A0A0A;border:1px solid #1A1A1A;border-radius:12px;margin-bottom:16px;'>
                  <div style='font-size:36px;font-weight:700;color:{r_color};'>{action}</div>
                  <div style='color:#555;font-size:13px;margin-top:6px;'>Confidence: <span style='color:{r_color};'>{conf:.0f}%</span></div>
                </div>
                """, unsafe_allow_html=True)

                if expl:
                    st.markdown(f"""
                    <div style='padding:14px;background:#0A0A0A;border-radius:8px;color:#888;font-size:13px;line-height:1.6;'>
                      {expl}
                    </div>
                    """, unsafe_allow_html=True)

                factors = rec.get("factors", {})
                if factors:
                    st.markdown("<br><div style='font-size:12px;color:#444;margin-bottom:8px;'>Scoring Breakdown</div>", unsafe_allow_html=True)
                    for k, v in factors.items():
                        sc = float(v) if v else 0
                        fc = BULL_COLOR if sc > 0 else (BEAR_COLOR if sc < 0 else NEUT_COLOR)
                        st.markdown(f"""
                        <div style='display:flex;justify-content:space-between;padding:6px 12px;background:#0A0A0A;border-radius:6px;margin-bottom:4px;'>
                          <span style='color:#555;font-size:12px;'>{k}</span>
                          <span style='color:{fc};font-size:12px;font-weight:600;'>{sc:+.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Recommendation error: {e}")
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#333;'>
          <div style='font-size:40px;'>🔍</div>
          <div style='margin-top:12px;font-size:14px;'>Enter a stock ticker above to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FORECAST CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif "Forecast" in page:
    st.markdown("""
    <div class='page-header'>
      <h1>📆 Forecast Center</h1>
      <p>Monte Carlo price range forecasts — 1, 7, 30, 90 day horizons</p>
    </div>
    """, unsafe_allow_html=True)

    ticker_input = st.text_input("Stock ticker", value="TCS", label_visibility="collapsed")
    ticker = ticker_input.strip().upper()
    if not ticker.endswith(".NS"): ticker_ns = ticker + ".NS"
    else: ticker_ns = ticker

    if st.button("📈 Run Forecast"):
        with st.spinner("Running Monte Carlo simulation..."):
            df = fetch_history(ticker_ns, period="1y")
            if not df.empty:
                try:
                    predictor = StockPredictor()
                    forecast_res = predictor.forecast_multi_horizon(ticker_ns, df)
                    forecast = forecast_res.get("forecasts", {}) if forecast_res else {}
                    st.session_state["forecast_df"]  = df
                    st.session_state["forecast_data"] = forecast
                    st.session_state["forecast_price"] = forecast_res.get("current_price", df["Close"].iloc[-1] if "Close" in df.columns else 0)
                    st.session_state["forecast_ticker"] = ticker
                except Exception as e:
                    st.error(f"Forecast error: {e}")

    if "forecast_data" in st.session_state:
        df       = st.session_state["forecast_df"]
        forecast = st.session_state["forecast_data"]
        tk       = st.session_state["forecast_ticker"]

        last_price = df["Close"].iloc[-1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-90:], y=df["Close"].iloc[-90:],
            mode="lines", line=dict(color="#FFFFFF", width=1.5),
            name="Historical"
        ))

        horizon_days = {"1 Day": 1, "1 Week": 7, "1 Month": 30, "3 Months": 90}
        colors = [BULL_COLOR, "#5BCEFA", "#00BFFF", "#87CEEB"]
        for (label, fdata), clr in zip(forecast.items(), colors):
            target = fdata.get("expected_price", last_price)
            low    = fdata.get("price_low", target * 0.95)
            high   = fdata.get("price_high", target * 1.05)
            days   = horizon_days.get(label, 7)
            future_date = df.index[-1] + timedelta(days=days)
            fc = BULL_COLOR if target >= last_price else BEAR_COLOR

            fig.add_trace(go.Scatter(
                x=[df.index[-1], future_date],
                y=[last_price, target],
                mode="lines+markers",
                line=dict(color=fc, width=2, dash="dot"),
                marker=dict(size=8, color=fc),
                name=label
            ))

        fig.update_layout(**PLOT_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(max(1, len(forecast)))
        for col, (label, fdata) in zip(cols, forecast.items()):
            target = fdata.get("expected_price", last_price)
            ret    = fdata.get("change_pct", 0)
            fc     = "bull" if ret >= 0 else "bear"
            arrow  = "▲" if ret >= 0 else "▼"
            with col:
                st.markdown(f"""
                <div class='metric-box'>
                  <div class='metric-label'>{label}</div>
                  <div class='metric-value {fc}'>₹{fmt(target,0)}</div>
                  <div class='metric-change {fc}'>{arrow} {pct(ret)}</div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SECTOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Sector" in page:
    st.markdown("""
    <div class='page-header'>
      <h1>🏭 Sector Analysis</h1>
      <p>Sector-wise performance comparison</p>
    </div>
    """, unsafe_allow_html=True)

    sector_indices = {
        "Banking":  "^NSEBANK",
        "IT":       "^CNXIT",
        "Pharma":   "NIFTY_PHARMA.NS",
        "FMCG":     "^CNXFMCG",
        "Energy":   "^CNXENERGY",
        "Auto":     "^CNXAUTO",
        "Metal":    "^CNXMETAL",
    }

    perf = {}
    for sector, idx in sector_indices.items():
        d = fetch_realtime(idx)
        perf[sector] = d.get("change_pct", 0)

    sectors = list(perf.keys())
    changes = list(perf.values())
    colors  = [BULL_COLOR if c >= 0 else BEAR_COLOR for c in changes]

    fig = go.Figure(go.Bar(
        x=sectors, y=changes,
        marker_color=colors,
        text=[pct(c) for c in changes],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11),
    ))
    fig.update_layout(**PLOT_LAYOUT, height=320,
                      yaxis_title="Change %", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    best   = max(perf, key=perf.get)
    worst  = min(perf, key=perf.get)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card-bull'>
          <div style='font-size:11px;color:#1E3A6E;text-transform:uppercase;letter-spacing:.08em;'>Best Sector</div>
          <div style='font-size:20px;font-weight:700;color:#1E90FF;margin-top:6px;'>{best}</div>
          <div style='color:#1E90FF;font-size:14px;'>{pct(perf[best])}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='card-bear'>
          <div style='font-size:11px;color:#6E1E1E;text-transform:uppercase;letter-spacing:.08em;'>Worst Sector</div>
          <div style='font-size:20px;font-weight:700;color:#FF3B3B;margin-top:6px;'>{worst}</div>
          <div style='color:#FF3B3B;font-size:14px;'>{pct(perf[worst])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    chosen = st.selectbox("Drill down into sector", list(SECTORS.keys()), label_visibility="collapsed")
    if chosen and chosen in SECTORS:
        stocks = SECTORS[chosen][:6]
        cols   = st.columns(3)
        for i, stk in enumerate(stocks):
            d  = fetch_realtime(stk + ".NS")
            pr = d.get("price", 0)
            ch = d.get("change_pct", 0)
            c  = "bull" if ch >= 0 else "bear"
            with cols[i % 3]:
                st.markdown(f"""
                <div class='metric-box' style='margin-bottom:10px;'>
                  <div class='metric-label'>{stk}</div>
                  <div class='metric-value {c}'>₹{fmt(pr,0)}</div>
                  <div class='metric-change {c}'>{pct(ch)}</div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PORTFOLIO TRACKER
# ══════════════════════════════════════════════════════════════════════════════
elif "Portfolio" in page:
    st.markdown("""
    <div class='page-header'>
      <h1>💼 Portfolio Tracker</h1>
      <p>Track your holdings, P&amp;L and portfolio allocation</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("➕ Add Holding", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1: p_ticker = st.text_input("Ticker", placeholder="RELIANCE")
        with c2: p_qty    = st.number_input("Qty", min_value=1, value=10)
        with c3: p_price  = st.number_input("Buy Price ₹", min_value=0.01, value=100.0)
        with c4: p_date   = st.date_input("Date")
        if st.button("Add to Portfolio"):
            try:
                db = SessionLocal()
                holding = PortfolioHolding(
                    ticker=p_ticker.upper() + (".NS" if not p_ticker.endswith(".NS") else ""),
                    quantity=p_qty, avg_buy_price=p_price
                )
                db.add(holding); db.commit(); db.close()
                st.success(f"Added {p_ticker}")
            except Exception as e:
                st.error(f"Error: {e}")

    try:
        db       = SessionLocal()
        holdings = db.query(PortfolioHolding).all()
        db.close()

        if holdings:
            rows = []
            total_inv = total_cur = 0
            for h in holdings:
                live = fetch_realtime(h.ticker)
                cur_price = live.get("price") or h.avg_buy_price
                inv  = h.avg_buy_price * h.quantity
                cur  = cur_price   * h.quantity
                pl   = cur - inv
                plpct= (pl / inv * 100) if inv else 0
                total_inv += inv; total_cur += cur
                rows.append({
                    "Ticker": h.ticker.replace(".NS",""),
                    "Qty": h.quantity,
                    "Buy ₹": fmt(h.avg_buy_price),
                    "LTP ₹": fmt(cur_price),
                    "Invested": fmt(inv, 0),
                    "Current": fmt(cur, 0),
                    "P&L": fmt(pl, 0),
                    "P&L %": pct(plpct),
                    "_pl": pl,
                })

            total_pl    = total_cur - total_inv
            total_plpct = (total_pl / total_inv * 100) if total_inv else 0
            ptc = "bull" if total_pl >= 0 else "bear"

            c1, c2, c3 = st.columns(3)
            for col, lbl, val, cls in zip([c1,c2,c3],
                ["Total Invested", "Current Value", "Total P&L"],
                [f"₹{fmt(total_inv,0)}", f"₹{fmt(total_cur,0)}", f"₹{fmt(total_pl,0)} ({pct(total_plpct)})"],
                ["neutral", ptc, ptc]):
                with col:
                    st.markdown(f"""
                    <div class='metric-box'>
                      <div class='metric-label'>{lbl}</div>
                      <div class='metric-value {cls}'>{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            display_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_pl"} for r in rows])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Pie chart
            fig = go.Figure(go.Pie(
                labels=[r["Ticker"] for r in rows],
                values=[float(r["Current"].replace(",","")) for r in rows],
                hole=0.5,
                marker=dict(colors=[BULL_COLOR if r["_pl"] >= 0 else BEAR_COLOR for r in rows],
                            line=dict(color="#000000", width=2)),
                textfont=dict(color="#FFFFFF"),
            ))
            fig.update_layout(**PLOT_LAYOUT, height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:60px;color:#333;'>
              <div style='font-size:40px;'>💼</div>
              <div style='margin-top:12px;font-size:14px;'>Add your first holding above</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Portfolio error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — AI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif "Chatbot" in page:
    st.markdown("""
    <div class='page-header'>
      <h1>🤖 AI Chatbot</h1>
      <p>Ask anything about stocks, markets, technicals, or your portfolio</p>
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Quick buttons
    quick = ["What is RSI?", "Explain MACD", "Best sectors today", "How to read candlesticks?"]
    cols  = st.columns(4)
    for col, q in zip(cols, quick):
        with col:
            if st.button(q, use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    try:
                        bot   = get_chatbot()
                        reply = bot.chat(q)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask about any stock, sector, or market concept...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                bot   = get_chatbot()
                reply = bot.chat(user_input)
            except Exception as e:
                reply = f"Error connecting to AI: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — MARKET HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
elif "Heatmap" in page:
    st.markdown("""
    <div class='page-header'>
      <h1>🗺️ Market Heatmap</h1>
      <p>Live color grid — Blue = Bullish · Red = Bearish</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Load Heatmap"):
        with st.spinner("Fetching market data..."):
            hmap_data = {}
            for sector, stocks in list(SECTORS.items())[:6]:
                sector_stocks = []
                for stk in stocks[:6]:
                    d   = fetch_realtime(stk + ".NS")
                    chg = d.get("change_pct", 0)
                    price = d.get("price", 0)
                    sector_stocks.append({"ticker": stk, "change": chg, "price": price})
                hmap_data[sector] = sector_stocks
            st.session_state["hmap_data"] = hmap_data

    hmap = st.session_state.get("hmap_data", {})
    if hmap:
        for sector, stocks in hmap.items():
            st.markdown(f"<div style='font-size:12px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.08em;margin:16px 0 8px;'>{sector}</div>", unsafe_allow_html=True)
            cols = st.columns(len(stocks))
            for col, stk in zip(cols, stocks):
                chg = stk["change"]
                price = stk["price"]
                if chg >= 2:   bg, tc = "#0A2547", "#1E90FF"
                elif chg >= 1: bg, tc = "#061830", "#4DA8FF"
                elif chg >= 0: bg, tc = "#040F1E", "#7EC8FF"
                elif chg >= -1: bg, tc = "#1E0505", "#FF7070"
                elif chg >= -2: bg, tc = "#2D0606", "#FF4444"
                else:           bg, tc = "#3D0A0A", "#FF3B3B"
                arrow = "▲" if chg >= 0 else "▼"
                with col:
                    st.markdown(f"""
                    <div style='background:{bg};border:1px solid {tc}33;border-radius:8px;padding:10px 6px;text-align:center;margin-bottom:6px;'>
                      <div style='color:#FFFFFF;font-size:11px;font-weight:600;'>{stk["ticker"]}</div>
                      <div style='color:{tc};font-size:13px;font-weight:700;margin-top:4px;'>{arrow}{abs(chg):.1f}%</div>
                      <div style='color:#444;font-size:10px;margin-top:2px;'>₹{price:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;'>
          <span style='font-size:11px;color:#888;'>Legend:</span>
          <span style='font-size:11px;padding:2px 8px;background:#0A2547;color:#1E90FF;border-radius:4px;'>▲ Strong Bull</span>
          <span style='font-size:11px;padding:2px 8px;background:#040F1E;color:#7EC8FF;border-radius:4px;'>▲ Mild Bull</span>
          <span style='font-size:11px;padding:2px 8px;background:#1E0505;color:#FF7070;border-radius:4px;'>▼ Mild Bear</span>
          <span style='font-size:11px;padding:2px 8px;background:#3D0A0A;color:#FF3B3B;border-radius:4px;'>▼ Strong Bear</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#333;'>
          <div style='font-size:40px;'>🗺️</div>
          <div style='margin-top:12px;font-size:14px;'>Click "Load Heatmap" to see live color grid</div>
        </div>
        """, unsafe_allow_html=True)
