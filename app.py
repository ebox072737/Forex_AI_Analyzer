import os
import time
import datetime
import requests
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
from fredapi import Fred
# === 只用 tvdatafeed；相容大小寫命名，並提供本地 vendor 後備 ===
import sys, pathlib
try:
    from tvDatafeed import TvDatafeed, Interval    # 大寫 D 版本
except Exception:
    try:
        from tvdatafeed import TvDatafeed, Interval  # 小寫版本
    except Exception:
        # 如果雲端還是裝不到，走本地 vendor 後備（見步驟 3）
        vendor_path = pathlib.Path(__file__).parent / "vendor"
        sys.path.append(str(vendor_path))
        from tvdatafeed import TvDatafeed, Interval
from deep_translator import GoogleTranslator

# 🧱 字型設定（顯示中文與負號）
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 🔐 讀取金鑰（建議從 st.secrets 讀）
FRED_API_KEY = st.secrets.get("FRED_API_KEY", os.getenv("FRED_API_KEY", ""))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

# 🕒 時間框架
TIMEFRAMES = {
    "5min": Interval.in_5_minute,
    "15min": Interval.in_15_minute,
    "1h": Interval.in_1_hour,
    "4h": Interval.in_4_hour,
}

# 🧮 指標對照
INDICATORS = {
    "CPI": "CPIAUCSL",
    "Unemployment Rate": "UNRATE",
    "Federal Funds Rate": "FEDFUNDS",
    "M2 Money Supply": "M2SL",
    "10Y Treasury Yield": "GS10",
    "Nonfarm Payrolls": "PAYEMS",
}

# 📥 初始化 tvdatafeed（匿名可用，但資料可能受限）
tv = TvDatafeed()  # 不登入


# ========= 資料抓取 =========
def fetch_macro_data():
    try:
        fred = Fred(api_key=FRED_API_KEY)
        out = {}
        for name, code in INDICATORS.items():
            s = fred.get_series(code)
            out[name] = round(float(s.iloc[-1]), 2)
        return out
    except Exception as e:
        return f"❌ 總經資料抓取失敗: {e}"


def fetch_candles(symbol="XAUUSD", label="15min", limit=100):
    try:
        df = tv.get_hist(
            symbol=symbol,
            exchange="OANDA",
            interval=TIMEFRAMES[label],
            n_bars=limit,
        )
        if df is None or df.empty:
            raise Exception("無法取得資料")

        df = df.reset_index().rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

        # 過濾週末與零量
        df = df[df["datetime"].dt.weekday < 5]
        df = df[df["volume"] > 0]

        # 格式化給 prompt
        df_fmt = df.copy()
        df_fmt["datetime"] = df_fmt["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        candles_for_prompt = df_fmt[
            ["datetime", "open", "high", "low", "close", "volume"]
        ].to_dict(orient="records")

        # 圖表用索引
        df = df.set_index("datetime")
        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]
        ].astype(float)

        return candles_for_prompt, df
    except Exception as e:
        return f"❌ {label} K 線抓取失敗: {e}", None


# ========= Prompt 組裝 =========
def builtin_long_prompt(symbol: str) -> str:
    return (
        "Act as a short-term forex analyst. Based on the provided macroeconomic data "
        f"and multi-timeframe candlestick charts, analyze the current market condition of {symbol} on the 5-minute timeframe.\\n\\n"
        "I am looking for a potential short-term long (buy) trade setup, targeting a profit of 50–100 pips.\\n\\n"
        "However, do not conclude that the market is bullish or that it is the right entry point just because I am looking for a buy setup. "
        "Provide your independent professional judgment. If the market is not favorable for a long position, explain why.\\n\\n"
        "Please include:\\n"
        "1. Current market bias: bullish or bearish?\\n"
        "2. Trade direction: buy or sell?\\n"
        "3. Suggested entry price\\n"
        "4. Take profit (TP) level — target 50–100 pips above entry\\n"
        "5. Stop loss (SL) level\\n"
        "6. Reasoning: key technical factors (support/resistance, candlestick patterns, momentum) and relevant macro influences."
    )


def builtin_short_prompt(symbol: str) -> str:
    return (
        "Act as a short-term forex analyst. Based on the provided macroeconomic data "
        f"and multi-timeframe candlestick charts, analyze the current market condition of {symbol} on the 5-minute timeframe.\\n\\n"
        "I am looking for a potential short-term short (sell) trade setup, targeting a profit of 50–100 pips.\\n\\n"
        "However, do not conclude that the market is bearish or that it is the right entry point just because I am looking for a sell setup. "
        "Provide your independent professional judgment. If the market is not favorable for a short position, explain why.\\n\\n"
        "Please include:\\n"
        "1. Current market bias: bullish or bearish?\\n"
        "2. Trade direction: buy or sell?\\n"
        "3. Suggested entry price\\n"
        "4. Take profit (TP) level — target 50–100 pips below entry\\n"
        "5. Stop loss (SL) level\\n"
        "6. Reasoning: key technical factors (resistance, candlestick patterns, momentum) and relevant macro influences."
    )


def make_prompt(macro_data: dict, kline_dict_for_prompt: dict, user_instruction: str, symbol: str) -> str:
    max_candle_config = {"5min": 50, "15min": 30, "1h": 20, "4h": 15}
    summary = ""
    for tf, records in kline_dict_for_prompt.items():
        if isinstance(records, str):
            summary += f"[{tf}] 抓取失敗\\n"
            continue
        max_bars = max_candle_config.get(tf, 20)
        summary += f"\\n[{tf}] 最近 {max_bars} 根K線：\\n"
        for c in records[-max_bars:]:
            summary += (
                f"{c['datetime']} | "
                f"O: {float(c['open']):.2f}, "
                f"H: {float(c['high']):.2f}, "
                f"L: {float(c['low']):.2f}, "
                f"C: {float(c['close']):.2f}, "
                f"V: {float(c['volume']):.0f}\\n"
            )

    prompt = f"""
📊 Latest U.S. macroeconomic indicators:
- CPI: {macro_data.get('CPI')}
- Unemployment rate: {macro_data.get('Unemployment Rate')}%
- Federal funds rate: {macro_data.get('Federal Funds Rate')}%
- M2 money supply: {macro_data.get('M2 Money Supply')}
- 10-year Treasury yield: {macro_data.get('10Y Treasury Yield')}%
- Nonfarm Payrolls: {macro_data.get('Nonfarm Payrolls')} thousand jobs

🕒 {symbol} multi-timeframe candlestick data:
{summary}

📌 Your task:
{user_instruction}
"""
    return prompt


# ========= AI 呼叫 =========
def analyze_with_groq(prompt: str, max_retries=3, backoff_factor=2):
    if not GROQ_API_KEY:
        return "❌ AI 分析錯誤: 缺少 GROQ_API_KEY（請在 Streamlit secrets 設定）。"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=90)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429 and attempt < max_retries:
                wait = backoff_factor ** (attempt - 1)
                time.sleep(wait)
                continue
            if status == 429:
                return "❌ AI 分析錯誤: 請求過於頻繁，稍後再試。"
            return f"❌ AI 分析錯誤: {e}"
        except Exception as e:
            return f"❌ AI 分析錯誤: {e}"


def translate_to_zh(text: str):
    try:
        return GoogleTranslator(source="en", target="zh-TW").translate(text)
    except Exception as e:
        return f"❌ 翻譯失敗: {e}"


# ========= Streamlit UI =========
st.set_page_config(page_title="📈 Forex_AI_Analyzer", layout="wide")

st.title("📈 Forex AI Analyzer (Streamlit)")

with st.sidebar:
    st.subheader("參數設定")
    symbol = st.text_input("商品代碼（例如 XAUUSD）", value="XAUUSD").strip().upper()

    strategy = st.selectbox("策略選擇", ["短多 50–100p", "短空 50–100p", "自訂 Prompt"])

    custom_prompt = st.text_area(
        "自訂分析指令（選擇『自訂 Prompt』時生效）",
        value=(
            "You are a professional forex market analyst. Based on the macroeconomic data and multi-timeframe candlestick "
            "price action data I provide, analyze current conditions and provide actionable short-term strategies focused on "
            "the 15m and 5m timeframes. Include: 1) key support/resistance, 2) long/short direction, 3) entry, "
            "4) TP & SL levels, with brief reasoning grounded in price structure and macro context."
        ),
        height=180,
    )

    run = st.button("🔍 開始分析", use_container_width=True)
    translate_flag = st.checkbox("📘 產出後自動翻譯成中文", value=False)

# 容器
macro_col, chart_col = st.columns([1, 2])
ai_col = st.container()

# 狀態暫存
if "ai_result_raw" not in st.session_state:
    st.session_state.ai_result_raw = ""
if "ai_result_final" not in st.session_state:
    st.session_state.ai_result_final = ""

# 執行
if run:
    with st.spinner("抓取總經資料中…"):
        macro = fetch_macro_data()

    if isinstance(macro, str):
        st.error(macro)
    else:
        with macro_col:
            st.success("✅ 總經資料抓取成功")
            st.json(macro)

        k_for_prompt = {}
        k_for_plot = {}
        errors = []

        with st.spinner(f"抓取 {symbol} K 線中…"):
            for lbl in TIMEFRAMES.keys():
                res_prompt, res_plot = fetch_candles(symbol=symbol, label=lbl, limit=100)
                if isinstance(res_prompt, str):
                    errors.append(res_prompt)
                else:
                    k_for_prompt[lbl] = res_prompt
                    k_for_plot[lbl] = res_plot

        if errors and len(k_for_plot) == 0:
            st.error("；".join(errors))
        else:
            with chart_col:
                st.success("✅ K 線抓取完成")
                # 用 Tabs 顯示各時間框
                tabs = st.tabs(list(k_for_plot.keys()))
                for (lbl, df), tab in zip(k_for_plot.items(), tabs):
                    with tab:
                        if df is None or df.empty:
                            st.warning(f"{lbl} 沒有可用資料")
                            continue
                        # 畫 K 線
                        fig, ax = mpf.plot(
                            df[["open", "high", "low", "close"]],
                            type="candle",
                            style="charles",
                            mav=None,
                            volume=False,
                            datetime_format="%m-%d %H:%M",
                            show_nontrading=False,
                            warn_too_much_data=10000,
                            xrotation=0,
                            returnfig=True,
                            figsize=(10, 4),
                        )
                        ax.set_title(f"{lbl} 蠟燭圖", pad=6)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            # 決定指令
            if strategy == "短多 50–100p":
                instruction = builtin_long_prompt(symbol)
            elif strategy == "短空 50–100p":
                instruction = builtin_short_prompt(symbol)
            else:
                instruction = custom_prompt.strip() or builtin_long_prompt(symbol)

            # 組 prompt & 呼叫 AI
            prompt = make_prompt(macro, k_for_prompt, instruction, symbol)
            with st.spinner("呼叫 AI 分析中…"):
                ai_text = analyze_with_groq(prompt)

            st.session_state.ai_result_raw = ai_text
            st.session_state.ai_result_final = (
                translate_to_zh(ai_text) if translate_flag else ai_text
            )

# 顯示 AI 結果 + 下載
if st.session_state.ai_result_final:
    ai_col.subheader("🤖 AI 分析結果")
    st.text_area(
        "結果",
        value=st.session_state.ai_result_final,
        height=380,
    )
    st.download_button(
        "⬇️ 下載結果 (txt)",
        data=st.session_state.ai_result_final,
        file_name=f"{symbol}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )
