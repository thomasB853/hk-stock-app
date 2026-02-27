import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
# 導入yfinance並增加異常處理
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

# ================== 數據獲取（核心修復：列名類型兼容） ==================
@st.cache_data(ttl=3600)
def get_data(symbol):
    """使用yfinance獲取港股數據，兼容列名類型和大小寫"""
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
        
        # 檢查數據是否為空
        if df.empty:
            st.error(f"❌ 未獲取到 {yf_symbol} 的數據，請確認代碼正確或該股票有公開交易數據")
            return None
        
        # 重置索引
        df.reset_index(inplace=True)
        
        # 核心修復：統一列名格式（處理元組/字符串混合的情況）
        new_columns = []
        for col in df.columns:
            # 如果是元組，取最後一個元素並轉字符串；如果是字符串直接使用
            if isinstance(col, tuple):
                col_str = str(col[-1])
            else:
                col_str = str(col)
            new_columns.append(col_str.lower())  # 轉小寫統一格式
        df.columns = new_columns
        
        # 統一列名映射（小寫→大寫）
        column_mapping = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adj close': 'Adj Close',
            'volume': 'Volume'
        }
        # 只重命名存在的列
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df.rename(columns=rename_dict, inplace=True)
        
        # 檢查必要列是否存在
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ 數據獲取失敗：缺少必要列 {missing_cols}")
            st.info("💡 可能原因：該股票暫無公開交易數據，或yfinance數據源暫時異常")
            return None
        
        # 數據清洗
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        
        # 檢查數據量
        if len(df) < 30:
            st.warning(f"⚠️ 數據量較少（僅{len(df)}條），分析結果可能不准")
        return df
    
    except Exception as e:
        st.error(f"❌ 數據獲取失敗：{str(e)}")
        st.info("🔍 排查建議：")
        st.info("1. 港股代碼需為4-5位數字（如小米=1810，騰訊=0700）")
        st.info("2. 刷新頁面重試（網絡偶發波動）")
        st.info("3. 確認該股票在港交所正常上市交易（非停牌/退市狀態）")
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
                st.warning(f"當前價 {