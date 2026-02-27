import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import warnings
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
        except:
            # 舊版本接口
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
        df = df.rename(columns=lambda x: column_mapping.get(x, x))
        
        # 數據清洗
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        
        # 檢查必要列是否存在
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"數據缺少必要列：{missing_cols}")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"數據獲取失敗：{str(e)}")
        return None

# 計算指標
def add_indicators(df):
    """計算技術指標，增加異常處理"""
    df = df.copy()
    try:
        # 移動平均線
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        
        # MACD
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        
        # RSI (修正計算公式，避免除零錯誤)
        delta = df["Close"].pct_change()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)  # 避免除零
        df["RSI"] = 100 - (100 / (1 + rs))
        
        return df
    except Exception as e:
        st.error(f"指標計算失敗：{str(e)}")
        return df

# 支撐壓力
def support_resistance(df, n=20):
    """計算最新支撐壓力位"""
    try:
        support = df["Low"].rolling(window=n, min_periods=1).min().iloc[-1]
        resistance = df["High"].rolling(window=n, min_periods=1).max().iloc[-1]
        return round(support, 2), round(resistance, 2)
    except:
        return round(df["Low"].iloc[-1], 2), round(df["High"].iloc[-1], 2)

# 預測
def simple_predict(df, days):
    """線性回歸預測未來價格"""
    try:
        df["idx"] = np.arange(len(df))
        x = df["idx"].values
        y = df["Close"].values
        k = np.polyfit(x, y, 1)
        future_idx = np.arange(len(df), len(df) + days)
        pred = k[0] * future_idx + k[1]
        return pred, k[0]
    except Exception as e:
        st.warning(f"預測計算失敗，使用當前價格：{str(e)}")
        pred = [df["Close"].iloc[-1]] * days
        return pred, 0

# ================== 主程式 ==================
if st.button("🚀 開始專業分析"):
    # 驗證輸入
    if not user_code.isdigit() or len(user_code) not in [4,5]:
        st.error("請輸入有效的港股代碼（4-5位數字）")
    else:
        df = get_data(user_code)
        if df is None or len(df) < 30:
            st.error(f"無法獲取足夠數據（需要至少30條），當前獲取：{len(df) if df is not None else 0} 條")
            st.info("建議檢查：\n1. 港股代碼是否正確\n2. 網絡連接是否正常\n3. 該股票是否有足夠的歷史數據")
        else:
            df = add_indicators(df)
            sup, res = support_resistance(df)
            pred, slope = simple_predict(df, predict_days)
            last = df["Close"].iloc[-1]

            # 展示數據
            st.subheader("📊 最新數據")
            show_df = df[["Date","Close","MA5","MA20","Volume"]].tail(10)
            # 格式化數字顯示
            show_df = show_df.round({"Close":2, "MA5":2, "MA20":2, "Volume":0})
            st.dataframe(show_df, use_container_width=True)

            # 設置中文字體（解決matplotlib中文亂碼）
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
            plt.rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題

            # 圖表
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("價格 & 均線")
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(df["Date"], df["Close"], label="價格", linewidth=1.5)
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
                st.subheader("支撐 / 壓力")
                st.info(f"📉 支撐位：{sup} HK$")
                st.info(f"📈 壓力位：{res} HK$")
                # 當前價格位置
                if last < sup:
                    st.success(f"當前價 {last} HK$：低於支撐位（超賣區間）")
                elif last > res:
                    st.warning(f"當前價 {last} HK$：高於壓力位（超買區間）")
                else:
                    st.info(f"當前價 {last} HK$：處於支撐壓力區間")

            # RSI
            st.subheader("RSI 超買超賣指標 (14日)")
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

            # 預測
            st.subheader(f"🔮 未來 {predict_days} 天價格預測 (線性回歸)")
            trend = "📈 上漲趨勢" if slope > 0 else "📉 下跌趨勢" if slope < 0 else "📊 平盤趨勢"
            st.success(f"整體趨勢：{trend} (斜率：{slope:.6f})")
            
            # 創建預測數據表
            pred_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=predict_days)
            pred_df = pd.DataFrame({
                "預測日期": pred_dates.strftime("%Y-%m-%d"),
                "預測價格 (HK$)": [round(p, 2) for p in pred]
            })
            st.dataframe(pred_df, use_container_width=True)
            st.info(f"當前價：{last:.2f} HK$ → 最後預測價：{pred[-1]:.2f} HK$")

            # 綜合建議
            st.subheader("📌 系統研判 (僅供參考)")
            rsi = df["RSI"].iloc[-1]
            ma5 = df["MA5"].iloc[-1]
            ma20 = df["MA20"].iloc[-1]

            col_advice1, col_advice2 = st.columns(2)
            with col_advice1:
                st.markdown("### 技術指標狀態")
                st.write(f"RSI當前值：{rsi:.1f}")
                st.write(f"MA5：{ma5:.2f} | MA20：{ma20:.2f}")
                st.write(f"價格/MA5：{'↑ 站穩' if last > ma5 else '↓ 跌破'}")
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

st.caption("⚠️ 本工具僅供學習分析，不構成任何投資建議，投資有風險，入市需謹慎")