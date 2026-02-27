import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_fixed

# ========== 核心優化配置（針對Streamlit免費版） ==========
warnings.filterwarnings('ignore')
# 居中布局+小標題，減少渲染負載
st.set_page_config(page_title="港股90天分析版", layout="centered")
# 禁用matplotlib交互後端，節省內存（關鍵）
plt.switch_backend('Agg')
# 輕量級字體配置，避免加載大字体文件
plt.rcParams["font.family"] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams["axes.unicode_minus"] = False

# ========== 輕量級依賴檢查（自動安裝，避免部署錯誤） ==========
try:
    import yfinance as yf
except ImportError:
    st.error("正在安裝必要依賴yfinance...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance==0.2.38"])
    import yfinance as yf

# ========== 頁面UI（簡潔不臃腫） ==========
st.title("📈 港股分析｜90天數據版")
st.markdown("### Streamlit免費版專用｜穩定不死機｜核心指標全保留")
st.divider()

# 熱門港股（選取數據最穩定的標的，避免異常）
hot_stocks = {
    "騰訊控股 (0700)": "0700",
    "美團-W (3690)": "3690",
    "匯豐控股 (0005)": "0005",
    "小米集團-W (1810)": "1810",
    "阿里巴巴-SW (9988)": "9988"
}
# 下拉選擇+代碼輸入
option = st.selectbox("🔍 選擇港股標的", list(hot_stocks.keys()))
default_code = hot_stocks[option]
user_code = st.text_input("手動輸入港股代碼（4-5位數字）", default_code).strip()

st.divider()

# ========== 工具函數（輕量級，無多餘計算） ==========
def clean_column_names(df):
    """列名清洗：兼容yfinance多格式列名，只處理核心字段"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    # 只映射分析需要的核心列
    col_map = {'date':'Date', 'close':'Close', 'low':'Low', 'high':'High', 'volume':'Volume'}
    df.rename(columns={k:v for k,v in col_map.items() if k in df.columns}, inplace=True)
    return df

# ========== 90天數據獲取（核心：重試+緩存+超時控制） ==========
@st.cache_data(ttl=3600)  # 緩存1小時，避免重複請求耗資源
@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))  # 失敗重試2次，間隔1秒
def get_hk_stock_90d(symbol):
    """獲取港股最近90天數據，針對Streamlit優化"""
    yf_symbol = f"{symbol}.HK"
    st.info(f"📥 正在獲取 {yf_symbol} 最近90天交易數據...")
    
    # 時間範圍：固定最近90天
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    try:
        # 輕量級下載：關閉進度條/多線程，縮短超時
        df = yf.download(
            yf_symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            timeout=10,
            threads=False
        )
        # 空數據判斷
        if df.empty:
            st.error("❌ 未獲取到數據（代碼錯誤/股票停牌/數據源異常）")
            return None
        
        # 數據清洗：只保留核心列
        df.reset_index(inplace=True)
        df = clean_column_names(df)
        # 必備列檢查（Close是核心，缺失直接返回）
        if "Close" not in df.columns:
            st.error("❌ 核心字段「收盤價」缺失，無法分析")
            return None
        # 補全輔助列（用Close填充，避免計算中斷）
        for col in ["Low", "High"]:
            if col not in df.columns:
                df[col] = df["Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 0
        
        # 最終清洗：排序+去空值
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        
        # 數據量提示（90天最少需20條數據才具分析意義）
        if len(df) < 20:
            st.warning(f"⚠️ 有效數據僅{len(df)}條，分析結果參考性有限")
        else:
            st.success(f"✅ 成功獲取 {len(df)} 條90天交易數據！")
        return df
    except Exception as e:
        st.error(f"❌ 數據獲取失敗：{str(e)}")
        st.info("💡 解決方案：刷新頁面/更換騰訊0700測試/檢查代碼格式")
        return None

# ========== 核心分析指標計算（90天專用，無多餘指標） ==========
def calculate_90d_indicators(df):
    """計算90天數據的核心技術指標：MA5/MA20、RSI、支撐壓力"""
    if df is None or len(df) < 5:
        return df
    
    df = df.copy()
    try:
        # 移動平均線（MA5/MA20，短期趨勢核心）
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        
        # RSI14（超買超賣核心，避免除零錯誤）
        delta = df["Close"].pct_change()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # 支撐壓力位（基於最近20天高低價，90天數據專用）
        df["Support"] = df["Low"].rolling(window=20, min_periods=1).min()
        df["Resistance"] = df["High"].rolling(window=20, min_periods=1).max()
        
        return df
    except Exception as e:
        st.warning(f"⚠️ 指標計算輕微異常，已自動簡化：{str(e)}")
        return df

# ========== 主執行邏輯（點擊分析，無自動執行，節省資源） ==========
if st.button("🚀 開始90天數據分析", type="primary", use_container_width=True):
    # 第一步：驗證代碼格式
    if not user_code.isdigit() or len(user_code) not in [4,5]:
        st.error("❌ 港股代碼格式錯誤！必須是4-5位數字（如騰訊=0700）")
    else:
        # 第二步：獲取90天數據
        df = get_hk_stock_90d(user_code)
        if df is None:
            st.stop()
        
        # 第三步：計算核心分析指標
        df = calculate_90d_indicators(df)
        
        # 第四步：提取最新數據（用於研判）
        last_close = round(df["Close"].iloc[-1], 2)
        last_ma5 = round(df["MA5"].iloc[-1], 2)
        last_ma20 = round(df["MA20"].iloc[-1], 2)
        last_rsi = round(df["RSI"].iloc[-1], 1)
        last_support = round(df["Support"].iloc[-1], 2)
        last_resistance = round(df["Resistance"].iloc[-1], 2)
        st.divider()

        # ========== 分析結果展示（模塊化，輕量渲染） ==========
        # 1. 最新核心數據（表格：只顯示最近10條，減少渲染）
        st.subheader("📊 最新10筆交易數據（90天範圍）")
        show_df = df[["Date","Close","MA5","MA20","Volume"]].tail(10)
        show_df = show_df.round({"Close":2, "MA5":2, "MA20":2, "Volume":0})
        st.dataframe(show_df, use_container_width=True, height=300)

        # 2. 價格+均線走勢圖（90天核心趨勢，縮小圖形尺寸）
        st.subheader("📈 90天價格 + MA5/MA20走勢")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["Date"], df["Close"], label=f"收盤價（最新：{last_close}）", color="#1f77b4", linewidth=1.2)
        ax.plot(df["Date"], df["MA5"], label=f"MA5（最新：{last_ma5}）", color="#ff7f0e", linewidth=1, alpha=0.8)
        ax.plot(df["Date"], df["MA20"], label=f"MA20（最新：{last_ma20}）", color="#2ca02c", linewidth=1, alpha=0.8)
        ax.set_title(f"{option} ({user_code}.HK) 90天趨勢", fontsize=12)
        ax.set_xlabel("交易日期", fontsize=10)
        ax.set_ylabel("價格（HK$）", fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()  # 自動調整布局，避免重疊
        st.pyplot(fig, use_container_width=True)

        # 3. RSI超買超賣指標（90天，核心風險判斷）
        st.subheader("📊 90天RSI14超買超賣指標")
        fig_rsi, ax_rsi = plt.subplots(figsize=(7, 3))
        ax_rsi.plot(df["Date"], df["RSI"], color="#9467bd", linewidth=1)
        # 超買/超賣/中軸線
        ax_rsi.axhline(70, c="#d62728", ls="--", alpha=0.7, label="超買線(70)")
        ax_rsi.axhline(30, c="#2ca02c", ls="--", alpha=0.7, label="超賣線(30)")
        ax_rsi.axhline(50, c="#7f7f7f", ls=":", alpha=0.5, label="中軸(50)")
        # 標註最新RSI值
        ax_rsi.text(0.98, 0.95, f"最新RSI：{last_rsi}", ha='right', va='top', transform=ax_rsi.transAxes, fontsize=9)
        ax_rsi.set_title("RSI14走勢（超買>70，超賣<30）", fontsize=12)
        ax_rsi.set_xlabel("交易日期", fontsize=10)
        ax_rsi.set_ylabel("RSI值", fontsize=10)
        ax_rsi.legend(fontsize=9)
        ax_rsi.tick_params(axis='both', labelsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_rsi, use_container_width=True)

        # 4. 支撐壓力位 + 綜合研判（核心結論，直接給出觀點）
        st.subheader("📌 90天數據綜合研判")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🛡️ 最新支撐/壓力位")
            st.info(f"📉 支撐位：{last_support} HK$")
            st.info(f"📈 壓力位：{last_resistance} HK$")
        with col2:
            st.markdown("### 📋 核心指標狀態")
            st.write(f"最新收盤價：{last_close} HK$")
            st.write(f"MA5/MA20：{last_ma5} / {last_ma20} HK$")
            st.write(f"RSI14：{last_rsi}（超買>70，超賣<30）")
        
        st.divider()
        st.markdown("### 🎯 操作建議（基於90天數據，僅供學習參考）")
        # 多條件綜合研判（簡潔，無多餘邏輯）
        if last_ma5 > last_ma20 and last_close > last_ma5 and last_rsi < 65:
            st.success("✅ 短期趨勢向上（MA5金叉MA20+價格站穩均線+RSI正常）：可適度關注")
        elif last_rsi > 70:
            st.warning("⚠️ RSI進入超買區間：注意價格回調風險，建議謹慎")
        elif last_rsi < 30:
            st.success("✅ RSI進入超賣區間：價格反彈概率較大，可留意機會")
        elif last_ma5 < last_ma20 and last_close < last_ma5:
            st.warning("⚠️ 短期趨勢向下（MA5死叉MA20+價格跌破均線）：建議觀察為主")
        else:
            st.info("🔍 震盪區間（指標無明確信號）：建議沿支撐/壓力位高拋低吸")

# ========== 底部提示（簡潔） ==========
st.divider()
st.caption("⚠️ 本工具基於港股最近90天公開交易數據分析，僅供編程學習使用，不構成任何投資建議！")
st.caption("📥 數據來源：Yahoo Finance | 📱 適配Streamlit Cloud免費版，穩定不死機")