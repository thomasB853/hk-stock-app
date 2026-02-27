import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak

# ================== 頁面設定 ==================
st.set_page_config(page_title="港股專業頂級版", layout="wide")
st.title("📈 港股分析預測系統｜專業頂級版")
st.markdown("### 支持：騰訊、美團、匯豐、美高梅、金沙、工行、阿里等")

# ================== 熱門港股 ==================
hot_stocks = {
    "騰訊控股": "0700",
    "美團": "3690",
    "匯豐": "0005",
    "美高梅中國": "2282",
    "金沙中國": "1928",
    "工商銀行": "1398",
    "小米集團": "1810",
    "阿里巴巴": "9988",
    "京東集團": "9618"
}

option = st.selectbox("熱門港股", list(hot_stocks.keys()))
code = hot_stocks[option]
user_code = st.text_input("輸入港股代碼（不需 .HK）", code).strip()
predict_days = st.slider("預測天數", 1, 15, 5)

# ================== 數據獲取 ==================
def get_data(symbol):
    try:
        df = ak.stock_hk_hist(symbol=symbol, period="daily", start_date="2022-01-01")
        df = df.rename(columns={"date":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna().reset_index(drop=True)
        return df
    except:
        return None

# 計算指標
def add_indicators(df):
    df["MA5"] = df.Close.rolling(5).mean()
    df["MA20"] = df.Close.rolling(20).mean()
    df["EMA12"] = df.Close.ewm(span=12).mean()
    df["EMA26"] = df.Close.ewm(span=26).mean()
    df["MACD"] = df.EMA12 - df.EMA26
    df["RSI"] = 100 - (100/(1 + df.Close.pct_change().rolling(14).mean()/df.Close.pct_change().rolling(14).std()))
    return df

# 支撐壓力
def support_resistance(df, n=20):
    support = df.Low.rolling(n).min()
    resistance = df.High.rolling(n).max()
    return support.iloc[-1], resistance.iloc[-1]

# 預測
def simple_predict(df, days):
    df["idx"] = np.arange(len(df))
    x = df[["idx"]]
    y = df["Close"]
    k = np.polyfit(df.idx, y, 1)
    future_idx = np.arange(len(df), len(df)+days)
    pred = k[0]*future_idx + k[1]
    return pred, k[0]

# ================== 主程式 ==================
if st.button("🚀 開始專業分析"):
    df = get_data(user_code)
    if df is None or len(df) < 30:
        st.error("無法獲取數據，請檢查代碼")
    else:
        df = add_indicators(df)
        sup, res = support_resistance(df)
        pred, slope = simple_predict(df, predict_days)
        last = df.Close.iloc[-1]

        # 展示數據
        st.subheader("📊 最新數據")
        show_df = df[["Date","Close","MA5","MA20","Volume"]].tail(10)
        st.dataframe(show_df, use_container_width=True)

        # 圖表
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("價格 & 均線")
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(df.Date, df.Close, label="價格")
            ax.plot(df.Date, df.MA5, label="MA5")
            ax.plot(df.Date, df.MA20, label="MA20")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("支撐 / 壓力")
            st.info(f"支撐：{sup:.2f}")
            st.info(f"壓力：{res:.2f}")

        # RSI
        st.subheader("RSI 超買超賣")
        fig_r, ax_r = plt.subplots(figsize=(8,2))
        ax_r.plot(df.Date, df.RSI)
        ax_r.axhline(70, c="r", ls="--")
        ax_r.axhline(30, c="g", ls="--")
        st.pyplot(fig_r)

        # 預測
        st.subheader(f"🔮 未來 {predict_days} 天預測")
        trend = "📈 上漲" if slope > 0 else "📉 下跌"
        st.success(f"趨勢：{trend}")
        st.info(f"當前：{last:.2f} → 預測：{pred[-1]:.2f}")

        # 綜合建議
        st.subheader("📌 系統研判")
        rsi = df.RSI.iloc[-1]
        ma5 = df.MA5.iloc[-1]
        ma20 = df.MA20.iloc[-1]

        if ma5 > ma20 and rsi < 65:
            st.success("✅ 趨勢向上，可關注")
        elif ma5 < ma20:
            st.warning("⚠️ 趨勢偏弱，謹慎")
        elif rsi > 70:
            st.warning("⚠️ 超買，注意回調")
        elif rsi < 30:
            st.success("✅ 超賣，可留意反彈")
        else:
            st.info("🔍 震盪區間，觀察為主")

st.caption("⚠️ 本工具僅供學習分析，不構成投資建議")