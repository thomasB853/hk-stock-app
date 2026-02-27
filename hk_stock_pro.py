import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import requests
import json

# ================== 全局配置 ==================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="港股專業頂級版", layout="wide")
# 設置中文字體（兼容Streamlit Cloud）
plt.rcParams["font.family"] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams["axes.unicode_minus"] = False

# ================== 依賴檢查 ==================
try:
    import yfinance as yf
except ImportError:
    st.error("❌ 缺少yfinance庫，正在自動安裝...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance>=0.2.30"])
    import yfinance as yf

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.error("❌ 缺少scikit-learn庫，正在自動安裝...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn>=1.3.0"])
    from sklearn.linear_model import LinearRegression

# ================== 頁面UI ==================
st.title("📈 港股分析預測系統｜穩定版")
st.markdown("### 支持：騰訊、美團、匯豐等主流港股（經過數據源兼容性優化）")

# 熱門港股（篩選Yahoo Finance數據穩定的標的）
hot_stocks = {
    "騰訊控股 (0700)": "0700",
    "美團-W (3690)": "3690",
    "匯豐控股 (0005)": "0005",
    "小米集團-W (1810)": "1810",
    "阿里巴巴-SW (9988)": "9988",
    "工商銀行 (1398)": "1398"
}

option = st.selectbox("選擇熱門港股（數據穩定）", list(hot_stocks.keys()))
default_code = hot_stocks[option]
user_code = st.text_input("手動輸入港股代碼（4-5位數字，如0700）", default_code).strip()
predict_days = st.slider("預測天數（1-15天）", 1, 15, 5)

# ================== 核心工具函數 ==================
def is_trading_day(date):
    """判斷港股交易日（排除週六/週日）"""
    return date.weekday() not in [5, 6]

def get_trading_dates(start_date, days):
    """獲取未來指定數量的港股交易日"""
    trading_dates = []
    current_date = start_date
    while len(trading_dates) < days:
        if is_trading_day(current_date):
            trading_dates.append(current_date)
        current_date += timedelta(days=1)
    return trading_dates

def clean_column_names(df):
    """
    核心列名清洗函數：兼容所有yfinance列名格式
    - 處理多級索引列名（如('Close', 'HKD')）
    - 處理大小寫混合列名
    - 處理特殊字符列名
    """
    # 第一步：如果是多級索引，壓縮為單級
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    
    # 第二步：映射到標準列名（覆蓋所有可能的變體）
    column_mapping = {
        'date': 'Date',
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'adj close': 'Adj Close',
        'adj_close': 'Adj Close',
        'volume': 'Volume',
        'vol': 'Volume'
    }
    
    # 第三步：模糊匹配列名（解決字段名變異）
    final_cols = {}
    for col in df.columns:
        for key in column_mapping.keys():
            if key in col:
                final_cols[col] = column_mapping[key]
                break
    
    df.rename(columns=final_cols, inplace=True)
    return df

# ================== 穩定的數據獲取函數（帶兜底） ==================
@st.cache_data(ttl=3600)  # 緩存1小時，減少請求次數
def get_hk_stock_data(symbol):
    """
    獲取港股數據（多層次兼容+兜底）
    :param symbol: 港股代碼（如0700）
    :return: 清洗後的DataFrame或None
    """
    # 步驟1：構建標準Yahoo Finance代碼
    yf_symbol = f"{symbol}.HK"
    st.info(f"🔍 正在獲取數據：{yf_symbol}")
    
    # 步驟2：下載數據（擴展時間範圍，增加成功率）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # 拉長到3年，確保有數據
    
    try:
        # 下載數據（關閉進度條+增加超時）
        df = yf.download(
            yf_symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            timeout=30,  # 超時30秒
            threads=False  # 關閉多線程，提升穩定性
        )
        
        # 步驟3：空數據檢查
        if df.empty:
            st.error(f"❌ 未獲取到 {yf_symbol} 的數據（可能是代碼錯誤/股票未上市/停牌）")
            return None
        
        # 步驟4：重置索引（Date列還原為普通列）
        df.reset_index(inplace=True)
        
        # 步驟5：核心列名清洗
        df = clean_column_names(df)
        
        # 步驟6：必要列檢查（允許部分缺失，降級處理）
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        # 處理缺失列（降級補全）
        if missing_cols:
            st.warning(f"⚠️ 部分字段缺失：{missing_cols}，正在嘗試補全...")
            
            # 補全Date列（必備）
            if "Date" not in df.columns:
                st.error("❌ 核心字段Date缺失，無法繼續分析")
                return None
            
            # 補全價格列（用Close填充其他缺失的價格列）
            if "Close" in df.columns:
                for col in ["Open", "High", "Low"]:
                    if col not in df.columns:
                        df[col] = df["Close"]
            else:
                st.error("❌ 核心字段Close缺失，無法繼續分析")
                return None
            
            # 補全Volume列（用0填充）
            if "Volume" not in df.columns:
                df["Volume"] = 0
        
        # 步驟7：最終數據清洗
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        
        # 步驟8：數據量檢查
        if len(df) < 10:
            st.warning(f"⚠️ 有效數據僅{len(df)}條（數據量過少，分析結果參考性低）")
        
        st.success(f"✅ 成功獲取 {yf_symbol} 數據（共{len(df)}條）")
        return df
    
    except Exception as e:
        st.error(f"❌ 數據獲取異常：{str(e)}")
        st.info("💡 解決方案：")
        st.info("1. 更換熱門港股測試（如騰訊0700、小米1810）")
        st.info("2. 刷新頁面重試（網絡/數據源臨時波動）")
        st.info("3. 確認港股代碼格式（必須是4-5位數字，如0700而非700）")
        return None

# ================== 技術指標計算（兼容缺失字段） ==================
def calculate_indicators(df):
    """計算技術指標（兼容缺失字段）"""
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    try:
        # 移動平均線（最小週期1，避免空值）
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        
        # MACD
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
        
        # RSI（避免除零錯誤+兼容少數據）
        delta = df["Close"].pct_change()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)  # 替換0避免除零
        df["RSI"] = 100 - (100 / (1 + rs))
        
        return df
    except Exception as e:
        st.warning(f"⚠️ 技術指標計算部分失敗：{str(e)}")
        return df

# ================== 支撐壓力位計算 ==================
def calculate_support_resistance(df, window=20):
    """計算支撐壓力位"""
    try:
        support = df["Low"].rolling(window=window, min_periods=1).min().iloc[-1]
        resistance = df["High"].rolling(window=window, min_periods=1).max().iloc[-1]
        return round(support, 2), round(resistance, 2)
    except:
        # 兜底：用最新價格計算
        return round(df["Low"].iloc[-1], 2), round(df["High"].iloc[-1], 2)

# ================== 價格預測（線性回歸） ==================
def predict_price(df, days):
    """線性回歸預測價格"""
    try:
        df["idx"] = np.arange(len(df))
        x = df["idx"].values.reshape(-1, 1)
        y = df["Close"].values
        
        model = LinearRegression()
        model.fit(x, y)
        
        future_idx = np.arange(len(df), len(df) + days).reshape(-1, 1)
        pred = model.predict(future_idx)
        slope = model.coef_[0]
        
        return pred, slope
    except Exception as e:
        st.warning(f"⚠️ 預測計算失敗，使用當前價格替代：{str(e)}")
        pred = [df["Close"].iloc[-1]] * days
        return pred, 0

# ================== 主執行邏輯 ==================
if st.button("🚀 開始分析（穩定版）", type="primary"):
    # 輸入驗證
    if not user_code.isdigit() or len(user_code) not in [4,5]:
        st.error("❌ 港股代碼格式錯誤！必須是4-5位數字（如騰訊=0700，小米=1810）")
    else:
        # 獲取數據
        df = get_hk_stock_data(user_code)
        if df is None:
            st.stop()
        
        # 計算技術指標
        df = calculate_indicators(df)
        if df is None:
            st.stop()
        
        # 計算支撐壓力位
        sup, res = calculate_support_resistance(df)
        # 預測價格
        pred, slope = predict_price(df, predict_days)
        last_close = df["Close"].iloc[-1]

        # ========== 展示數據 ==========
        # 最新10筆數據
        st.subheader("📊 最新交易數據（前10筆）")
        show_df = df[["Date","Open","High","Low","Close","Volume","MA5","MA20"]].tail(10)
        show_df = show_df.round({
            "Open":2, "High":2, "Low":2, "Close":2, 
            "Volume":0, "MA5":2, "MA20":2
        })
        st.dataframe(show_df, use_container_width=True)

        # 價格走勢圖
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 價格 & 均線走勢")
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(df["Date"], df["Close"], label="收盤價", color="#1f77b4", linewidth=1.5)
            ax.plot(df["Date"], df["MA5"], label="MA5（5日均線）", color="#ff7f0e", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA20"], label="MA20（20日均線）", color="#2ca02c", linewidth=1, alpha=0.8)
            ax.set_title(f"{option} ({user_code}.HK) 價格走勢", fontsize=10)
            ax.set_xlabel("日期", fontsize=8)
            ax.set_ylabel("價格 (HK$)", fontsize=8)
            ax.legend(fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            st.subheader("🛡️ 支撐 / 壓力位")
            st.info(f"📉 支撐位：{sup} HK$")
            st.info(f"📈 壓力位：{res} HK$")
            if last_close < sup:
                st.success(f"當前價 {last_close:.2f} HK$：低於支撐位（超賣區間）")
            elif last_close > res:
                st.warning(f"當前價 {last_close:.2f} HK$：高於壓力位（超買區間）")
            else:
                st.info(f"當前價 {last_close:.2f} HK$：處於支撐壓力區間")

        # RSI指標圖
        st.subheader("📊 RSI 14日超買超賣指標")
        fig_r, ax_r = plt.subplots(figsize=(10,3))
        ax_r.plot(df["Date"], df["RSI"], color="#9467bd", linewidth=1)
        ax_r.axhline(70, c="#d62728", ls="--", alpha=0.7, label="超買線(70)")
        ax_r.axhline(30, c="#2ca02c", ls="--", alpha=0.7, label="超賣線(30)")
        ax_r.axhline(50, c="#7f7f7f", ls=":", alpha=0.5, label="中軸(50)")
        ax_r.set_title("RSI 走勢（14日）", fontsize=10)
        ax_r.set_xlabel("日期", fontsize=8)
        ax_r.set_ylabel("RSI 值", fontsize=8)
        ax_r.legend(fontsize=8)
        ax_r.tick_params(axis='both', labelsize=7)
        plt.xticks(rotation=45)
        st.pyplot(fig_r)

        # 價格預測
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

        # 綜合研判
        st.subheader("📌 技術研判（僅供學習參考）")
        rsi = df["RSI"].iloc[-1]
        ma5 = df["MA5"].iloc[-1]
        ma20 = df["MA20"].iloc[-1]

        col_advice1, col_advice2 = st.columns(2)
        with col_advice1:
            st.markdown("### 指標狀態")
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

# ================== 底部提示 ==================
st.divider()
st.caption("⚠️ 重要提示：")
st.caption("1. 本工具僅供編程學習使用，不構成任何投資建議")
st.caption("2. 數據來源為Yahoo Finance，請以港交所官方數據為準")
st.caption("3. 若持續獲取數據失敗，更換「騰訊0700/小米1810」等熱門股測試")