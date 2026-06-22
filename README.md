# 🤖 AI-Powered NSE/BSE Stock Market Intelligence Dashboard

A production-ready, end-to-end stock market intelligence platform for Indian markets (NSE & BSE) powered by machine learning, real-time data, sentiment analysis, and a conversational AI chatbot.

---

## 🚀 Features

| Module | Description |
|---|---|
| 📈 **Market Overview** | NIFTY50, SENSEX, BankNifty live KPIs, top movers |
| 🔮 **Tomorrow's Picks** | ML-predicted top 10 risers & fallers for next day |
| 🏢 **Company Analysis** | Deep dive: price, technicals, sentiment, risk, recommendation |
| 📆 **Forecast Center** | 1 / 7 / 30 / 90-day price range forecasts (Monte Carlo) |
| 🏭 **Sector Analysis** | Performance heatmap across Banking, IT, Pharma, FMCG, Auto, Energy |
| 💼 **Portfolio Tracker** | Add holdings, track P&L, AI portfolio recommendations |
| 🤖 **AI Chatbot** | Groq-powered Llama3 assistant for natural language stock queries |
| 🗺️ **Market Heatmap** | Treemap view — sector → stock, colored by daily performance |

---

## 🏗️ Architecture

```
yfinance API
     │
     ▼
StockDataFetcher ──► SQLite / PostgreSQL (StockPrice, TechnicalIndicator, Prediction...)
     │
     ▼
FeatureEngineer (35+ features: RSI, MACD, BB, VWAP, returns, volatility, OBV...)
     │
     ▼
ModelTrainer ──► RandomForest │ XGBoost │ LightGBM │ LSTM  ──► best model by F1
     │
     ▼
StockPredictor ──► tomorrow probability + Monte Carlo multi-horizon forecast
     │
     ├──► TechnicalAnalyzer  (RSI/MACD/BB signals + human interpretation)
     ├──► SentimentAnalyzer  (NewsAPI + VADER → Bullish / Bearish / Neutral)
     ├──► RiskAnalyzer       (VaR, Sharpe, max drawdown, Beta)
     └──► RecommendationEngine (BUY / SELL / HOLD + confidence score)
                │
                ▼
     Streamlit Dashboard (8 pages)  +  FastAPI REST API  +  Groq Chatbot
```

---

## 📁 Project Structure

```
nse_bse_dashboard/
├── config/
│   ├── settings.py          # Central config (paths, API keys, ML params)
│   └── stocks_list.py       # NIFTY50 / NIFTY_NEXT50 ticker lists
├── src/
│   ├── data/
│   │   ├── database.py      # SQLAlchemy ORM models & init
│   │   ├── fetcher.py       # yfinance data fetcher
│   │   └── processor.py     # Feature engineering (35+ features)
│   ├── ml/
│   │   ├── trainer.py       # Train RF / XGB / LGB / LSTM; auto-select best
│   │   └── predictor.py     # Tomorrow prediction + multi-horizon forecast
│   ├── analysis/
│   │   ├── technical.py     # RSI, MACD, BB, SMA, EMA, VWAP, support/resistance
│   │   ├── sentiment.py     # NewsAPI + VADER sentiment scoring
│   │   ├── risk.py          # VaR, Sharpe, drawdown, Beta, risk category
│   │   └── recommendation.py# BUY/SELL/HOLD with weighted scoring
│   ├── chatbot/
│   │   └── agent.py         # Groq Llama3-70b financial assistant
│   ├── api/
│   │   └── routes.py        # FastAPI REST endpoints
│   └── dashboard/
│       └── app.py           # Streamlit multi-page dashboard (8 pages)
├── scripts/
│   ├── daily_pipeline.py    # 5-step daily cron: fetch→predict→technical→sentiment→recommend
│   └── train_models.py      # Train global or per-ticker ML models
├── tests/
│   └── test_pipeline.py     # pytest test suite
├── docs/
│   └── deployment_guide.md  # Railway, Docker, Render, Streamlit Cloud
├── data/
│   ├── raw/                 # Raw CSVs from yfinance
│   ├── processed/           # Feature-engineered CSVs
│   └── models/              # Trained .pkl / .pt model files
├── .env.example
├── .gitignore
├── requirements.txt
├── Procfile
├── Dockerfile
├── docker-compose.yml
└── railway.toml
```

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone https://github.com/Sakanavenkat/nse-bse-dashboard.git
cd nse-bse-dashboard

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   GROQ_API_KEY  → https://console.groq.com (free)
#   NEWS_API_KEY  → https://newsapi.org (free tier: 100 req/day)
```

### 3. Initialize Database

```bash
python -c "from src.data.database import init_db; init_db()"
```

### 4. Fetch Historical Data & Train Models

```bash
# Fetch 2 years of historical data for all NIFTY50 stocks
python scripts/daily_pipeline.py --init

# Train ML models (RF, XGBoost, LightGBM, LSTM)
python scripts/train_models.py
```

### 5. Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 6. (Optional) Launch REST API

```bash
uvicorn src.api.routes:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

---

## 🔑 API Keys Required

| Key | Where to Get | Free Tier |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | ✅ Free |
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org) | ✅ 100 req/day free |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Optional fallback |

> **Note:** The dashboard works without `NEWS_API_KEY` — it uses mock news data as fallback.

---

## 🤖 ML Models

| Model | Type | Use Case |
|---|---|---|
| Random Forest | Ensemble | Tomorrow up/down classification |
| XGBoost | Gradient Boost | Primary classifier (fast, accurate) |
| LightGBM | Gradient Boost | Large-scale training fallback |
| LSTM | Deep Learning | Sequential pattern detection |

Best model is auto-selected by F1 score on holdout test set.

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/stocks/{ticker}/realtime` | Live price + change |
| GET | `/api/stocks/{ticker}/predict` | Tomorrow's probability |
| GET | `/api/stocks/{ticker}/forecast` | 1/7/30/90-day forecast |
| GET | `/api/stocks/{ticker}/technical` | All technical indicators |
| GET | `/api/stocks/{ticker}/recommend` | BUY/SELL/HOLD recommendation |
| GET | `/api/market/top-picks` | Top 10 predicted risers/fallers |
| POST | `/api/chat` | AI chatbot query |
| GET/POST | `/api/portfolio` | Portfolio management |

Full docs at `/docs` when API is running.

---

## ⏰ Daily Automation

```bash
# Add to crontab (runs at 6:30 AM IST every weekday)
30 1 * * 1-5 cd /path/to/project && venv/bin/python scripts/daily_pipeline.py >> logs/daily.log 2>&1
```

---

## 🚢 Deployment

| Platform | Command |
|---|---|
| **Railway** | Push repo → set env vars → auto-deploy via `railway.toml` |
| **Docker** | `docker-compose up --build` |
| **Render** | Connect repo → set `streamlit run src/dashboard/app.py` as start command |
| **Streamlit Cloud** | Connect GitHub repo → set secrets in dashboard |

See `docs/deployment_guide.md` for detailed instructions.

---

## ⚠️ Disclaimer

> This platform is for **educational and research purposes only**. Stock predictions are probabilistic and NOT financial advice. Always consult a SEBI-registered financial advisor before investing. Past performance does not guarantee future results.

---

## 👤 Author

**Sakanav** — B.E. CSE (AI & ML)  
GitHub: [github.com/Sakanavenkat](https://github.com/Sakanavenkat)  
Email: sakanav03@gmail.com
