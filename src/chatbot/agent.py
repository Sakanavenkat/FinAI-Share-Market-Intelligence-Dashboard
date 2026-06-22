# src/chatbot/agent.py
# ============================================================
# AI Financial Chatbot — powered by Groq (Llama 3)
# ============================================================

from groq import Groq
from datetime import datetime
from loguru import logger
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS
from config.stocks_list import get_company_name, ALL_NSE_STOCKS


SYSTEM_PROMPT = """You are FINAI, an expert Indian stock market analyst and financial advisor AI assistant for NSE and BSE markets.

You have deep knowledge of:
- Indian stock market (NSE/BSE), SEBI regulations, Nifty 50, Sensex
- Technical analysis (RSI, MACD, Bollinger Bands, VWAP, Moving Averages)
- Fundamental analysis (P/E ratio, EPS, revenue growth, debt ratios)
- Market sectors: Banking, IT, Pharma, FMCG, Energy, Auto, Telecom, Metal
- Portfolio management and risk assessment
- Economic factors affecting Indian markets (RBI policy, FII/DII activity, inflation, INR/USD)

When answering:
- Be precise, data-driven, and professional
- Always mention that recommendations are for educational purposes only
- Explain technical terms in simple language
- Use ₹ (Indian Rupee) for prices
- Reference specific NSE/BSE listed companies when relevant
- Structure complex answers with clear sections

Current context: {context}

DISCLAIMER: This is an AI assistant. All analysis is for educational purposes only. Always consult a SEBI-registered advisor before investing."""


class FinancialChatbot:
    """AI chatbot for stock market questions."""

    def __init__(self):
        if GROQ_API_KEY:
            self.client = Groq(api_key=GROQ_API_KEY)
            self.available = True
        else:
            self.client = None
            self.available = False
        self.conversation_history = []

    # ----------------------------------------------------------
    def chat(self, user_message: str, context: dict = None) -> str:
        """Generate a response to a user's stock market question."""
        context_str = self._build_context(context)

        if not self.available:
            return self._rule_based_response(user_message)

        system = SYSTEM_PROMPT.format(context=context_str)
        self.conversation_history.append({"role": "user", "content": user_message})

        # Keep last 10 turns for context
        messages = self.conversation_history[-10:]

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=GROQ_MAX_TOKENS,
                temperature=0.3,
            )
            reply = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._rule_based_response(user_message)

    # ----------------------------------------------------------
    def _build_context(self, context: dict) -> str:
        if not context:
            return f"Current date: {datetime.now().strftime('%Y-%m-%d')}. Market is open."
        parts = [f"Date: {datetime.now().strftime('%Y-%m-%d')}"]
        if context.get("ticker"):
            parts.append(f"Stock in focus: {context['ticker']} ({get_company_name(context['ticker'])})")
        if context.get("current_price"):
            parts.append(f"Current price: ₹{context['current_price']}")
        if context.get("prediction"):
            pred = context["prediction"]
            parts.append(f"AI Prediction: {pred.get('prob_increase', 'N/A')}% probability of rise tomorrow")
        if context.get("recommendation"):
            rec = context["recommendation"]
            parts.append(f"Recommendation: {rec.get('action')} (Confidence: {rec.get('confidence')}%)")
        if context.get("sentiment"):
            sent = context["sentiment"]
            parts.append(f"News Sentiment: {sent.get('sentiment_label')} (Score: {sent.get('compound_score')})")
        return " | ".join(parts)

    # ----------------------------------------------------------
    def _rule_based_response(self, message: str) -> str:
        """Fallback responses when API key not configured."""
        msg_lower = message.lower()

        if any(w in msg_lower for w in ["rsi", "what is rsi"]):
            return (
                "**RSI (Relative Strength Index)** is a momentum indicator ranging from 0–100.\n\n"
                "- **RSI < 30**: Oversold — potential buy signal 🟢\n"
                "- **RSI 30–70**: Neutral zone\n"
                "- **RSI > 70**: Overbought — potential sell signal 🔴\n\n"
                "It measures the speed and magnitude of recent price changes to evaluate "
                "whether a stock is overbought or oversold."
            )
        elif "macd" in msg_lower:
            return (
                "**MACD (Moving Average Convergence Divergence)** tracks trend and momentum.\n\n"
                "- **MACD crosses above Signal line** → Bullish signal 🟢\n"
                "- **MACD crosses below Signal line** → Bearish signal 🔴\n"
                "- **Histogram positive & rising** → Strong bullish momentum\n\n"
                "It uses 12-day and 26-day EMAs with a 9-day signal line."
            )
        elif any(w in msg_lower for w in ["buy", "should i"]):
            return (
                "I can analyze any stock for you! Use the **Company Analysis** page to get:\n"
                "- AI-generated Buy/Sell/Hold recommendation\n"
                "- Confidence score and risk level\n"
                "- Technical indicator summary\n"
                "- News sentiment analysis\n\n"
                "⚠️ *This dashboard is for educational purposes. Consult a SEBI-registered advisor before investing.*"
            )
        elif any(w in msg_lower for w in ["tomorrow", "predict", "rise", "increase"]):
            return (
                "Go to the **Tomorrow's Picks** page to see:\n"
                "- 🟢 Top 10 stocks predicted to rise tomorrow\n"
                "- 🔴 Top 10 stocks predicted to fall tomorrow\n\n"
                "Our AI model analyzes RSI, MACD, volume patterns, news sentiment, and historical "
                "patterns to generate these predictions with confidence scores."
            )
        elif any(w in msg_lower for w in ["sector", "banking", "it stock", "pharma"]):
            return (
                "**Indian Market Sectors:**\n\n"
                "🏦 **Banking**: HDFCBANK, ICICIBANK, KOTAKBANK, SBIN, AXISBANK\n"
                "💻 **IT**: TCS, INFY, WIPRO, HCLTECH, TECHM\n"
                "💊 **Pharma**: SUNPHARMA, DRREDDY, CIPLA, DIVISLAB\n"
                "🛒 **FMCG**: HINDUNILVR, ITC, NESTLEIND, BRITANNIA\n"
                "⚡ **Energy**: RELIANCE, NTPC, ONGC, POWERGRID\n"
                "🚗 **Auto**: MARUTI, TATAMOTORS, M&M, BAJAJ-AUTO\n\n"
                "Check the **Sector Analysis** page for real-time sector performance."
            )
        elif any(w in msg_lower for w in ["portfolio", "my stocks"]):
            return (
                "Use the **Portfolio Tracker** page to:\n"
                "- Add your stock holdings\n"
                "- Track real-time P&L\n"
                "- Get AI recommendations for each holding\n"
                "- View your overall portfolio risk score\n"
            )
        else:
            return (
                "I'm FINAI, your Indian stock market AI assistant! I can help you with:\n\n"
                "- 📊 Technical analysis (RSI, MACD, Bollinger Bands)\n"
                "- 🔮 Stock predictions and forecasts\n"
                "- 📰 News sentiment analysis\n"
                "- 💼 Portfolio recommendations\n"
                "- 🏭 Sector analysis (Banking, IT, Pharma, etc.)\n\n"
                "Try asking: *'Should I buy Reliance?'* or *'Which IT stocks are bullish?'*\n\n"
                "💡 *Configure GROQ_API_KEY in .env for full AI-powered responses.*"
            )

    # ----------------------------------------------------------
    def clear_history(self):
        self.conversation_history = []

    # ----------------------------------------------------------
    def get_quick_analysis(self, ticker: str, data_bundle: dict) -> str:
        """Generate a quick natural-language stock summary."""
        name = get_company_name(ticker)
        price = data_bundle.get("current_price", "N/A")
        pred = data_bundle.get("prediction", {})
        rec = data_bundle.get("recommendation", {})
        sent = data_bundle.get("sentiment", {})

        prompt = (
            f"Give me a 3-paragraph analysis of {name} ({ticker}) as of today. "
            f"Current price: ₹{price}. "
            f"AI Prediction: {pred.get('prob_increase', 'N/A')}% chance of rise tomorrow. "
            f"Recommendation: {rec.get('action', 'N/A')} with {rec.get('confidence', 'N/A')}% confidence. "
            f"News sentiment: {sent.get('sentiment_label', 'Neutral')}. "
            "Cover: 1) Current market position, 2) Key technical signals, 3) Investment outlook. "
            "Use simple language and mention specific numbers."
        )
        return self.chat(prompt, data_bundle)
