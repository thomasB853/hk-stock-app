import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
# 核心修復：確保yfinance導入兼容（若安裝失敗給出提示）
try:
    import yfinance as yf
except ImportError:
    st.error("❌ 缺少yfinance庫，請確保requirements.txt包含yfinance>=0.2.30")
    st.stop()

warnings.filterwarnings('ignore')

# ================== 頁面設定 ==================
st.set_page_config(page_title="港股專業頂級版", layout="wide")
st.title("📈 港股分析預測系統｜專業頂級版")
st.markdown("### 支持：騰訊、美團、匯豐、美高梅、金沙、工行、阿里等")

# ================== 熱門港股（适配yfinance格式） ==================
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
default_code = hot_stocks[option]
user_code = st.text_input("輸入港股代碼（不需 .HK）", default_code).strip()
predict_days = st.slider("預測天數", 1, 15, 5)

# ================== 工具函數 ==================
def setup_chinese_font():
    """設置中文字體（适配Streamlit Cloud）"""
    try:
        plt.rcParams["font.family"] = ['DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams["axes.unicode_minus"] = False
    except:
        pass

setup_chinese_font()

def is_trading_day(date):
    """判斷港股交易日"""
    return date.weekday() not in [5, 6]

def get_trading_dates(start_date, days):
    """獲取未來港股交易日"""
    trading_dates = []
    current_date = start_date
    while len(trading_dates) < days:
        if is_trading_day(current_date):
            trading_dates.append(current_date)
        current_date += timedelta(days=1)
    return trading_dates

# ================== 數據獲取（基於yfinance，國際可訪問） ==================
@st.cache_data(ttl=3600)
def get_data(symbol):
    """使用yfinance獲取港股數據"""
    try:
        # 拼接yfinance格式：代碼.HK
        yf_symbol = f"{symbol}.HK"
        
        # 獲取過去3年數據
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        
        # 下載數據（關閉進度條，适配線上環境）
        df = yf.download(
            yf_symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        # 數據清洗與重命名
        if df.empty:
            st.error(f"❌ 未獲取到 {yf_symbol} 的數據，請確認代碼正確")
            return None
        
        df.reset_index(inplace=True)
        df.rename(columns={
            'Date': 'Date', 'Open': 'Open', 'High': 'High',
            'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        }, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        
        # 檢查數據量
        if len(df) < 30:
            st.warning(f"⚠️ 數據量較少（僅{len(df)}條），分析結果可能不准")
        return df
    
    except Exception as e:
        st.error(f"數據獲取失敗：{str(e)}")
        st.info("🔍 排查建議：")
        st.info("1. 港股代碼需為4-5位數字（如小米=1810）")
        st.info("2. 刷新頁面重試（網絡偶發波動）")
        st.info("3. 確認該股票在港交所正常上市交易")
        return None

# 計算技術指標
def add_indicators(df):
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    try:
        # 移動平均線
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        
        # MACD
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
        
        # RSI（避免除零錯誤）
        delta = df["Close"].pct_change()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)
        df["RSI"] = 100 - (100 / (1 + rs))
        
        return df
    except Exception as e:
        st.error(f"指標計算失敗：{str(e)}")
        return df

# 計算支撐壓力位
def support_resistance(df, n=20):
    try:
        support = df["Low"].rolling(window=n, min_periods=1).min().iloc[-1]
        resistance = df["High"].rolling(window=n, min_periods=1).max().iloc[-1]
        return round(support, 2), round(resistance, 2)
    except:
        return round(df["Low"].iloc[-1], 2), round(df["High"].iloc[-1], 2)

# 線性回歸價格預測
def simple_predict(df, days):
    try:
        df["idx"] = np.arange(len(df))
        x = df["idx"].values.reshape(-1, 1)
        y = df["Close"].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        
        future_idx = np.arange(len(df), len(df) + days).reshape(-1, 1)
        pred = model.predict(future_idx)
        slope = model.coef_[0]
        
        return pred, slope
    except Exception as e:
        st.warning(f"預測計算失敗，使用當前價格：{str(e)}")
        pred = [df["Close"].iloc[-1]] * days
        return pred, 0

# ================== 主程式執行 ==================
if st.button("🚀 開始專業分析"):
    # 驗證輸入格式
    if not user_code.isdigit() or len(user_code) not in [4,5]:
        st.error("❌ 請輸入有效的港股代碼（4-5位數字，如騰訊=0700）")
    else:
        # 獲取數據
        df = get_data(user_code)
        if df is None:
            st.stop()
        
        # 計算指標
        df = add_indicators(df)
        if df is None:
            st.stop()
        
        # 計算支撐壓力和預測
        sup, res = support_resistance(df)
        pred, slope = simple_predict(df, predict_days)
        last_close = df["Close"].iloc[-1]

        # 展示最新數據
        st.subheader("📊 最新10筆交易數據")
        show_df = df[["Date","Close","MA5","MA20","Volume"]].tail(10)
        show_df = show_df.round({"Close":2, "MA5":2, "MA20":2, "Volume":0})
        st.dataframe(show_df, use_container_width=True)

        # 繪製價格走勢圖
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("價格 & 均線走勢")
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(df["Date"], df["Close"], label="收盤價", linewidth=1.5)
            ax.plot(df["Date"], df["MA5"], label="MA5", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA20"], label="MA20", linewidth=1, alpha=0.8)
            ax.set_title(f"{option} ({user_code}.HK) 價格走勢", fontsize=10)
            ax.set_xlabel("日期", fontsize=8)
            ax.set_ylabel("價格 (HK$)", fontsize=8)
            ax.legend(fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            st.subheader("支撐 / 壓力位")
            st.info(f"📉 支撐位：{sup} HK$")
            st.info(f"📈 壓力位：{res} HK$")
            if last_close < sup:
                st.success(f"當前價 {last_close:.2f} HK$：低於支撐位（超賣區間）")
            elif last_close > res:
                st.warning(f"當前價 {last_close:.2f} HK$：高於壓力位（超買區間）")
            else:
                st.info(f"當前價 {last_close:.2f} HK$：處於支撐壓力區間")

        # 繪製RSI指標圖
        st.subheader("RSI 14日超買超賣指標")
        fig_r, ax_r = plt.subplots(figsize=(10,3))
        ax_r.plot(df["Date"], df["RSI"], color="purple", linewidth=1)
        ax_r.axhline(70, c="red", ls="--", alpha=0.7, label="超買線(70)")
        ax_r.axhline(30, c="green", ls="--", alpha=0.7, label="超賣線(30)")
        ax_r.axhline(50, c="gray", ls=":", alpha=0.5, label="中軸(50)")
        ax_r.set_title("RSI 走勢", fontsize=10)
        ax_r.set_xlabel("日期", fontsize=8)
        ax_r.set_ylabel("RSI 值", fontsize=8)
        ax_r.legend(fontsize=8)
        ax_r.tick_params(axis='both', labelsize=7)
        plt.xticks(rotation=45)
        st.pyplot(fig_r)

        # 展示價格預測結果
        st.subheader(f"🔮 未來 {predict_days} 天價格預測（線性回歸）")
        trend = "📈 上漲趨勢" if slope > 0 else "📉 下跌趨勢" if slope < 0 else "📊 平盤趨勢"
        st.success(f"整體趨勢：{trend} (斜率：{slope:.6f})")
        
        # 生成交易日預測日期
        last_trading_day = df["Date"].iloc[-1]
        pred_dates = get_trading_dates(last_trading_day + timedelta(days=1), predict_days)
        pred_df = pd.DataFrame({
            "預測日期": [d.strftime("%Y-%m-%d") for d in pred_dates],
            "預測價格 (HK$)": [round(p, 2) for p in pred[:len(pred_dates)]]
        })
        st.dataframe(pred_df, use_container_width=True)
        st.info(f"當前價：{last_close:.2f} HK$ → 最後預測價：{pred[-1]:.2f} HK$")

        # 綜合技術研判
        st.subheader("📌 系統研判（僅供參考）")
        rsi = df["RSI"].iloc[-1]
        ma5 = df["MA5"].iloc[-1]
        ma20 = df["MA20"].iloc[-1]

        col_advice1, col_advice2 = st.columns(2)
        with col_advice1:
            st.markdown("### 技術指標狀態")
            st.write(f"RSI當前值：{rsi:.1f}")
            st.write(f"MA5：{ma5:.2f} | MA20：{ma20:.2f}")
            st.write(f"價格/MA5：{'↑ 站穩' if last_close > ma5 else '↓ 跌破'}")
            st.write(f"MA5/MA20：{'↑ 金叉' if ma5 > ma20 else '↓ 死叉'}")

        with col_advice2:
            st.markdown("### 操作建議")
            if ma5 > ma20 and rsi < 65:
                st.success("✅ 趨勢向上，可適度關注")
            elif ma5 < ma20:
                st.warning("⚠️ 短期趨勢偏弱，謹慎操作")
            elif rsi > 70:
                st.warning("⚠️ RSI超買，注意回調風險")
            elif rsi < 30:
                st.success("✅ RSI超賣，可留意反彈機會")
            else:
                st.info("🔍 震盪區間，建議觀察為主")

st.caption("⚠️ 本工具僅供學習分析，不構成任何投資建議｜數據來源：Yahoo Finance")