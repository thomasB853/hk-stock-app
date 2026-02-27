import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import subprocess
import sys
import os

# ------------------- 環境初始化與依賴檢查 -------------------
def check_and_upgrade_dependencies():
    """檢查並升級必要依賴庫"""
    try:
        # 檢查yfinance版本，低於0.2.31則升級
        if yf.__version__ < "0.2.31":
            st.warning("⚠️ yfinance版本過舊，正在自動升級...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yfinance>=0.2.31"])
            import importlib
            importlib.reload(yf)
    except AttributeError:
        # 版本屬性不存在時直接升級
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yfinance>=0.2.31"])
        import importlib
        importlib.reload(yf)
    except Exception as e:
        st.warning(f"⚠️ 依賴升級提示：{str(e)}")

# 初始化檢查
check_and_upgrade_dependencies()

# ------------------- 核心數據獲取函數 -------------------
@st.cache_data(ttl=3600)  # 緩存1小時，避免重複請求
def get_hk_stock_data(symbol):
    """
    獲取港股數據（雙數據源：yfinance + 備用接口）
    :param symbol: 港股代碼（數字，如700、0700）
    :return: 清洗後的數據DataFrame，失敗返回None
    """
    # 設置網絡代理（如有需要請取消註釋並修改）
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    
    # 標準化股票代碼
    yf_symbol = f"{symbol}.HK"
    st.info(f"🔍 正在獲取數據：{yf_symbol}")
    
    # 方案1：使用yfinance獲取（主數據源）
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)  # 獲取近3年數據
        
        # 增強版請求參數，提升兼容性
        df = yf.download(
            yf_symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            timeout=60,        # 超時延長到60秒
            threads=False,     # 關閉多線程避免異常
            auto_adjust=False, # 關閉自動調整
            back_adjust=False, # 關閉回調
            repair=True        # 開啟數據修復
        )
        
        # 檢查數據是否有效
        if not df.empty and len(df) > 0:
            # 數據清洗與格式整理
            df = df.reset_index()
            # 重命名列為中文，便於展示
            df.rename(columns={
                'Date': '日期',
                'Open': '開盤價',
                'High': '最高價',
                'Low': '最低價',
                'Close': '收盤價',
                'Adj Close': '調整後收盤價',
                'Volume': '成交量'
            }, inplace=True)
            
            # 格式轉換與去空值
            df['日期'] = pd.to_datetime(df['日期']).dt.date
            df = df.dropna(subset=['收盤價']).reset_index(drop=True)
            
            st.success(f"✅ 成功獲取{yf_symbol}數據，共{len(df)}條記錄")
            return df
    except Exception as e:
        st.warning(f"⚠️ yfinance獲取失敗：{str(e)}")

    # 方案2：備用數據源（解決yfinance兼容問題）
    st.warning("⚠️ 切換至備用數據源獲取...")
    try:
        # 使用免費港股接口（穩定兼容0700.HK）
        # 接口說明：基於公開數據源，僅供學習使用
        url = f"https://query1.finance.yahoo.com/v7/finance/chart/{yf_symbol}?range=3y&interval=1d&indicators=quote&includeTimestamps=true"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # 發送請求並解析數據
        response = requests.get(url, headers=headers, timeout=30)
        data = response.json()
        
        # 提取核心數據
        timestamp = data['chart']['result'][0]['timestamp']
        quote = data['chart']['result'][0]['indicators']['quote'][0]
        
        # 構建DataFrame
        df = pd.DataFrame({
            '日期': [datetime.fromtimestamp(t).date() for t in timestamp],
            '開盤價': quote['open'],
            '最高價': quote['high'],
            '最低價': quote['low'],
            '收盤價': quote['close'],
            '成交量': quote['volume']
        })
        
        # 數據清洗
        df = df.dropna(subset=['收盤價']).reset_index(drop=True)
        
        if len(df) > 0:
            st.success(f"✅ 備用數據源獲取成功，共{len(df)}條記錄")
            return df
        else:
            st.error("❌ 備用數據源返回空數據")
            return None
            
    except Exception as e:
        st.error(f"❌ 備用數據源獲取失敗：{str(e)}")
        return None

# ------------------- Streamlit 頁面佈局 -------------------
def main():
    """主頁面邏輯"""
    st.set_page_config(page_title="港股數據查詢工具", page_icon="📈", layout="wide")
    st.title("📈 港股數據查詢工具（穩定版）")
    st.divider()
    
    # 用戶輸入區域
    col1, col2 = st.columns([2, 1])
    with col1:
        user_code = st.text_input(
            "請輸入港股代碼（數字）", 
            placeholder="例如：騰訊輸入700，小米輸入1810",
            value="700"  # 默認騰訊
        )
    
    # 查詢按鈕與邏輯
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # 垂直居中
        if st.button("🚀 開始分析（穩定版）", type="primary"):
            # 輸入驗證與自動補零
            if not user_code.strip().isdigit():
                st.error("❌ 請輸入有效的數字代碼！")
            else:
                # 自動補全4位港股代碼（700 → 0700）
                user_code = user_code.strip().zfill(4)
                st.info(f"📌 標準化代碼：{user_code}")
                
                # 獲取數據
                stock_data = get_hk_stock_data(user_code)
                
                # 展示數據
                if stock_data is not None and len(stock_data) > 0:
                    st.subheader(f"📊 {user_code}.HK 近3年數據")
                    # 顯示最新5條數據
                    st.dataframe(stock_data.tail(), use_container_width=True)
                    
                    # 數據可視化
                    st.subheader("📉 股價走勢圖")
                    st.line_chart(
                        stock_data,
                        x="日期",
                        y=["開盤價", "最高價", "最低價", "收盤價"],
                        use_container_width=True
                    )
                    
                    # 下載數據按鈕
                    csv_data = stock_data.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="💾 下載數據（CSV）",
                        data=csv_data,
                        file_name=f"{user_code}_HK_數據_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"❌ 未獲取到 {user_code}.HK 的數據（可能是代碼錯誤/股票未上市/停牌）")

# ------------------- 程序入口 -------------------
if __name__ == "__main__":
    main()