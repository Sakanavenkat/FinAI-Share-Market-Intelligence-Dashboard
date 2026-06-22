# docs/deployment_guide.md
# ============================================================
# Deployment Guide — NSE/BSE AI Dashboard
# ============================================================

## 🖥️ Option 1: Local Development

```bash
# 1. Clone and setup
git clone <your-repo>
cd nse_bse_dashboard
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add GROQ_API_KEY and NEWS_API_KEY

# 3. Initialize and run
python -c "from src.data.database import init_db; init_db()"
python scripts/daily_pipeline.py --init    # Fetch 3yr data (takes 10-20 min)
python scripts/train_models.py             # Train ML models
streamlit run src/dashboard/app.py         # Launch dashboard
```

---

## 🚂 Option 2: Railway (Recommended — Free Tier)

### Step 1: Prepare
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
```

### Step 2: Create `railway.toml`
```toml
[build]
builder = "NIXPACKS"
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "streamlit run src/dashboard/app.py --server.port $PORT --server.address 0.0.0.0"
healthcheckPath = "/"
```

### Step 3: Create `Procfile`
```
web: streamlit run src/dashboard/app.py --server.port $PORT --server.address 0.0.0.0
worker: python scripts/daily_pipeline.py
```

### Step 4: Deploy
```bash
railway init
railway up
# Set environment variables in Railway dashboard
# GROQ_API_KEY, NEWS_API_KEY, DATABASE_URL
```

---

## 🐳 Option 3: Docker

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -c "from src.data.database import init_db; init_db()"

EXPOSE 8501

ENV PYTHONPATH=/app
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### Build & Run
```bash
docker build -t nse-dashboard .
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_key \
  -e NEWS_API_KEY=your_key \
  nse-dashboard
```

### Docker Compose (with daily pipeline)
```yaml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    env_file: .env
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  scheduler:
    build: .
    command: python -c "
      import schedule, time
      import subprocess
      schedule.every().day.at('06:00').do(
        lambda: subprocess.run(['python', 'scripts/daily_pipeline.py'])
      )
      while True: schedule.run_pending(); time.sleep(60)
    "
    env_file: .env
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

---

## 🌐 Option 4: Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `streamlit run src/dashboard/app.py --server.port $PORT --server.address 0.0.0.0`
6. Add environment variables: `GROQ_API_KEY`, `NEWS_API_KEY`

---

## ⚡ Option 5: Streamlit Cloud (Easiest)

1. Push to GitHub (repo must be public or you need Streamlit Cloud Pro)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App**
4. Set **Main file path:** `src/dashboard/app.py`
5. Add secrets in the Streamlit Cloud UI:
   ```toml
   GROQ_API_KEY = "your_key"
   NEWS_API_KEY = "your_key"
   ```

---

## 📅 Production Cron Setup

### Linux/Mac (crontab)
```bash
crontab -e
# Add: Run daily at 6:00 AM on weekdays
0 6 * * 1-5 /path/to/venv/bin/python /path/to/scripts/daily_pipeline.py >> /var/log/stock_pipeline.log 2>&1
```

### Windows Task Scheduler
- Action: Start a program
- Program: `C:\path\to\venv\Scripts\python.exe`
- Arguments: `C:\path\to\scripts\daily_pipeline.py`
- Schedule: Daily, 6:00 AM, Monday–Friday

### APScheduler (in-process)
```python
from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(run_pipeline, 'cron', hour=6, minute=0, day_of_week='mon-fri')
scheduler.start()
```

---

## 🔒 Production Checklist

- [ ] `.env` file NOT committed to git (add to `.gitignore`)
- [ ] `DATABASE_URL` points to persistent storage (not SQLite on Railway/Render)
- [ ] GROQ_API_KEY set in environment
- [ ] Initial data fetch complete (`--init` flag)
- [ ] Models trained (`train_models.py`)
- [ ] Daily cron job configured
- [ ] Streamlit secrets configured if using Streamlit Cloud

---

## 📊 Database: Upgrade to PostgreSQL (Production)

```bash
pip install psycopg2-binary
```

Set `DATABASE_URL` in `.env`:
```
DATABASE_URL=postgresql://user:password@host:5432/stocks_db
```

Free PostgreSQL options:
- [Neon.tech](https://neon.tech) — 512MB free tier
- [Railway PostgreSQL](https://railway.app) — 1GB free tier
- [Supabase](https://supabase.com) — 500MB free tier
