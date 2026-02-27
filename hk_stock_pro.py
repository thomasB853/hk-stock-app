import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import warnings
from datetime import datetime, timedelta
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ================== 頁面設定 ==================
st.set_page_config(page_title="港股專業頂級版", layout="wide")
st.title("📈 港股分析預測系統｜專業頂級版")
st.markdown("### 支持：騰訊、美團、匯豐、美高梅、金沙、工行、阿里等")

# ================== 熱門港股 ==================
hot_stocks = {
    "騰訊控股": "00700",
    "美團": "03690",
    "匯豐": "00005",
    "美高梅中國": "02282",
    "金沙中國": "01928",
    "工商銀行": "01398",
    "小米集團": "01810",
    "阿里巴巴": "09988",
    "京東集團": "09618"
}

option = st.selectbox("熱門港股", list(hot_stocks.keys()))
default_code = hot_stocks[option]
user_code = st.text_input("輸入港股代碼（不需 .HK）", default_code).strip()
predict_days = st.slider("預測天數", 1, 15, 5)

# ================== 工具函數 ==================
def setup_chinese_font():
    """設置中文字體，增加兼容性"""
    try:
        # 優先使用系統中文字體
        font_paths = [
            'SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS',
            'PingFang SC', 'Microsoft YaHei', 'Apple LiGothic Medium'
        ]
        for font in font_paths:
            if fm.FontProperties(fname=font).get_name() != 'DejaVu Sans':
                plt.rcParams["font.family"] = font
                break
    except:
        # 備用方案：關閉中文顯示
        plt.rcParams["font.family"] = 'DejaVu Sans'
    finally:
        plt.rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題

# 初始化字體
setup_chinese_font()

def is_trading_day(date):
    """簡單判斷是否為港股交易日（排除週六週日）"""
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

# ================== 數據獲取 ==================
@st.cache_data(ttl=3600)  # 緩存數據1小時，減少重複請求
def get_data(symbol):
    """獲取港股歷史數據，增強異常處理和兼容性"""
    try:
        # 兼容akshare不同版本的參數
        try:
            # 新版本接口
            df = ak.stock_hk_hist(
                symbol=symbol,
                period="daily",
                start_date="2022-01-01",
                adjust="qfq"  # 前復權
            )
        except TypeError:
            # 舊版本接口（無adjust參數）
            df = ak.stock_hk_hist(
                symbol=symbol,
                period="daily",
                start_date="2022-01-01"
            )
        
        # 統一列名（兼容不同返回格式）
        column_mapping = {
            "日期": "Date", "date": "Date",
            "開盤": "Open", "open": "Open",
            "最高": "High", "high": "High",
            "最低": "Low", "low": "Low",
            "收盤": "Close", "close": "Close",
            "成交量": "Volume", "volume": "Volume"
        }
        # 只重命名存在的列
        df.columns = [column_mapping.get(col, col) for col in df.columns]
        
        # 數據清洗
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        
        # 檢查必要列是否存在
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"數據缺少必要列：{missing_cols}")
            return None
        
        # 去重
        df = df.drop_duplicates(subset=["Date"], keep="last")
        
        return df
    
    except Exception as e:
        st.error(f"數據獲取失敗：{str(e)}")
        st.info("請確認：1. 港股代碼正確 2. 網絡正常 3. akshare版本最新")
        return None

# 計算指標
def add_indicators(df):
    """計算技術指標，增加異常處理"""
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    try:
        # 移動平均線
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["