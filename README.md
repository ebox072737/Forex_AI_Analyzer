# Forex_AI_Analyzer

A Streamlit web app that fetches macro data from FRED, pulls multi-timeframe candles from TradingView via tvDatafeed (OANDA), and asks Groq (llama3-70b-8192) for short-term trading analysis. It also supports one‑click Chinese translation.

## Features
- Input symbol (default: XAUUSD)
- Timeframes: 5m / 15m / 1h / 4h (last 100 bars)
- FRED macro indicators (CPI, UNRATE, FEDFUNDS, M2SL, GS10, PAYEMS)
- Built‑in prompts for **long/short 50–100 pips**, or custom prompt
- AI analysis via Groq (429 backoff)
- Mplfinance candlesticks
- Optional zh‑TW translation
- Export analysis as TXT

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secrets
Create `.streamlit/secrets.toml` (or configure on Streamlit Cloud / HF Spaces):
```toml
FRED_API_KEY = "YOUR_FRED_API_KEY"
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

## Deploy
- **Streamlit Community Cloud**: connect this repo → set **Secrets** → Deploy
- **Hugging Face Spaces**: create Space (Streamlit) → upload files → set **Secrets**
