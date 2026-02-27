import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import requests
import json
import subprocess
import sys
import importlib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# ================== 全局配置 ==================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="港股專業頂級版", layout="wide")
# 增強中文字體配置（解決亂碼問題）
plt.rcParams["font.family"] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 額外增加字體保險

# ================== 依賴檢查&強制升級 ==================
# 強制升級yfinance到最新版，解決數據源兼容問題
try:
    import yfinance as yf
    # 檢查版本，低於0.2.31則自動升級
    if hasattr(yf, '__version__') and yf.__version__ < "0.2.31":
        st.warning("⚠️ yfinance版本過舊，正在自動升級至最新版...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yfinance>=0.2.31"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        importlib.reload(yf)
except ImportError:
    st.error("❌ 缺少yfinance庫，正在自動安裝...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance>=0.2.31"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import yfinance as yf

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.error("❌ 缺少scikit-learn庫，正在自動安裝...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn>=1.3.0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from sklearn.linear_model import LinearRegression

# ================== 頁面UI ==================
st.title("📈 港股分析預測系統｜增強版")
st.markdown("### 支持：騰訊、美團、匯豐等主流港股 + 恆生指數（預測模型升級：隨機森林+多特征）")

# 熱門港股（篩選Yahoo Finance數據穩定的標的）
hot_stocks = {
    "騰訊控股 (0700)": "0700",
    "美團-W (3690)": "3690",
    "匯豐控股 (0005)": "0005",
    "小米集團-W (1810)": "1810",
    "阿里巴巴-SW (9988)": "9988",
    "工商銀行 (1398)": "1398",
    "恆生指數 (^HSI)": "^HSI"
}
option = st.selectbox("選擇熱門港股/指數（數據穩定）", list(hot_stocks.keys()))
default_code = hot_stocks[option]
user_code = st.text_input("手動輸入港股代碼（4-5位數字，如0700）或恆生指數(^HSI)", default_code).strip()
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

# ================== 業績查詢函數 ==================
def get_stock_financials(stock_code):
    """獲取港股公司去年財務業績（基於公開API）"""
    if stock_code == "^HSI":
        return "恆生指數為市場指數，無單獨業績數據"
    
    try:
        # 使用財務數據API獲取業績（備用方案）
        # 方案1：直接從yfinance獲取財務數據
        yf_symbol = f"{stock_code}.HK"
        ticker = yf.Ticker(yf_symbol)
        
        # 獲取年度財務報表
        financials = ticker.financials
        if not financials.empty:
            # 取最新財務年度數據（去年）
            last_year = datetime.now().year - 1
            financials.columns = [pd.to_datetime(col).year for col in financials.columns]
            if last_year in financials.columns:
                year_data = financials[last_year]
                
                # 整理核心業績指標
                performance = {
                    "營業收入": year_data.get("Total Revenue", "N/A"),
                    "淨利潤": year_data.get("Net Income", "N/A"),
                    "每股收益": year_data.get("Basic EPS", "N/A"),
                    "總資產": year_data.get("Total Assets", "N/A"),
                    "總負債": year_data.get("Total Liabilities", "N/A")
                }
                
                # 格式化數據
                perf_df = pd.DataFrame(list(performance.items()), columns=["指標", "數值（HKD）"])
                perf_df["數值（HKD）"] = perf_df["數值（HKD）"].apply(lambda x: f"{x:,.2f}" if x != "N/A" else x)
                return perf_df
        
        # 方案2：備用API（如果yfinance財務數據缺失）
        url = f"https://api.finance.qq.com/stock/finance/hk/{stock_code}/index.json"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and "finance" in data["data"]:
                finance_data = data["data"]["finance"]
                performance = {
                    "營業收入": finance_data.get("operating_revenue", "N/A"),
                    "淨利潤": finance_data.get("net_profit", "N/A"),
                    "每股收益": finance_data.get("eps", "N/A"),
                    "資產負債率": finance_data.get("debt_ratio", "N/A"),
                    "股息率": finance_data.get("dividend_yield", "N/A")
                }
                perf_df = pd.DataFrame(list(performance.items()), columns=["指標", "數值"])
                return perf_df
        
        return "暫無該股票去年業績數據（數據源限制）"
    
    except Exception as e:
        st.warning(f"⚠️ 業績數據獲取失敗：{str(e)[:100]}")
        return "業績數據獲取失敗，請稍後再試"

# ================== 穩定的數據獲取函數 ==================
@st.cache_data(ttl=3600)  # 緩存1小時，減少請求次數
def get_hk_stock_data(symbol):
    """
    獲取港股/指數數據（多層次兼容+兜底+請求優化）
    :param symbol: 港股代碼（如0700）或恆生指數(^HSI)
    :return: 清洗後的DataFrame或None
    """
    # 步驟1：構建標準Yahoo Finance代碼
    if symbol == "^HSI":
        yf_symbol = "^HSI"
    else:
        yf_symbol = f"{symbol}.HK"
    st.info(f"🔍 正在獲取數據：{yf_symbol}")
    
    # 步驟2：下載數據（擴展時間範圍，增加成功率）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # 拉長到3年，確保有數據
    
    try:
        # 核心優化：提升港股/指數兼容性
        df = yf.download(
            yf_symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            timeout=60,        # 超時從30秒延長到60秒
            threads=False,     # 關閉多線程，提升穩定性
            auto_adjust=False, # 關閉自動調整，避免數據格式異常
            back_adjust=False, # 關閉回調，兼容港股原始數據
            repair=True        # 開啟數據修復
        )
        
        # 步驟3：空數據檢查（增加二次驗證）
        if df.empty or len(df) < 5:
            # 兜底嘗試：直接調用Yahoo Finance接口請求
            st.warning("⚠️ 默認方式獲取數據失敗，嘗試備用接口獲取...")
            url = f"https://query1.finance.yahoo.com/v7/finance/chart/{yf_symbol}?range=3y&interval=1d&indicators=quote&includeTimestamps=true"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=60)
            data = resp.json()
            # 解析備用接口數據
            if 'chart' in data and 'result' in data['chart'] and len(data['chart']['result'])>0:
                ts = data['chart']['result'][0]['timestamp']
                quote = data['chart']['result'][0]['indicators']['quote'][0]
                df = pd.DataFrame({
                    'Date': [datetime.fromtimestamp(t) for t in ts],
                    'Open': quote['open'],
                    'High': quote['high'],
                    'Low': quote['low'],
                    'Close': quote['close'],
                    'Volume': quote.get('volume', [0]*len(ts))
                })
                # 去除空值
                df = df.dropna(subset=['Close'])
            else:
                st.error(f"❌ 未獲取到 {yf_symbol} 的數據（可能是代碼錯誤/指數未上市/停牌）")
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
        st.error(f"❌ 數據獲取異常：{str(e)[:100]}")
        st.info("💡 解決方案：")
        st.info("1. 刷新頁面重試（網絡/數據源臨時波動）")
        st.info("2. 確認港股代碼格式（必須是4-5位數字，如0700而非700）或輸入^HSI查詢恆生指數")
        st.info("3. 更換熱門港股測試（如騰訊0700、小米1810）")
        return None

# ================== 技術指標計算（新增MA30/50/100） ==================
def calculate_indicators(df):
    """計算技術指標（兼容缺失字段，新增MA30/50/100）"""
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    try:
        # 移動平均線（最小週期1，避免空值）
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["MA30"] = df["Close"].rolling(window=30, min_periods=1).mean()  # 新增
        df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()  # 新增
        df["MA100"] = df["Close"].rolling(window=100, min_periods=1).mean()  # 新增
        
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

# ================== 優化版價格預測模塊（核心修改） ==================
def clean_outliers(df, column="Close"):
    """處理股價異常值（IQR方法）"""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_clean

def prepare_features(df):
    """構建多特征數據集（替代單一時間索引）"""
    df_feat = df.copy()
    
    # 基礎價格特征
    df_feat["price_change"] = df_feat["Close"].pct_change()
    df_feat["high_low_diff"] = df_feat["High"] - df_feat["Low"]
    df_feat["open_close_diff"] = df_feat["Open"] - df_feat["Close"]
    
    # 技術指標特征（包含新增的MA線）
    df_feat["rsi_norm"] = df_feat["RSI"] / 100  # 歸一化RSI
    df_feat["macd_diff"] = df_feat["MACD"] - df_feat["MACD_Signal"]
    df_feat["ma5_ma20_diff"] = df_feat["MA5"] - df_feat["MA20"]
    df_feat["ma20_ma30_diff"] = df_feat["MA20"] - df_feat["MA30"]  # 新增
    df_feat["ma30_ma50_diff"] = df_feat["MA30"] - df_feat["MA50"]  # 新增
    df_feat["close_ma5_diff"] = df_feat["Close"] - df_feat["MA5"]
    
    # 成交量特征
    df_feat["volume_change"] = df_feat["Volume"].pct_change()
    
    # 時間特征
    df_feat["day_of_week"] = df_feat["Date"].dt.weekday
    df_feat["month"] = df_feat["Date"].dt.month
    
    # 填充缺失值（避免模型報錯）
    df_feat = df_feat.fillna(0)
    # 去除無窮值
    df_feat = df_feat.replace([np.inf, -np.inf], 0)
    
    # 特征列篩選（僅保留數值型特征）
    feature_cols = [
        "price_change", "high_low_diff", "open_close_diff",
        "rsi_norm", "macd_diff", "ma5_ma20_diff", "ma20_ma30_diff", "ma30_ma50_diff",
        "close_ma5_diff", "volume_change", "day_of_week", "month"
    ]
    # 確保特征列存在
    feature_cols = [col for col in feature_cols if col in df_feat.columns]
    
    return df_feat, feature_cols

def predict_price_optimized(df, days):
    """
    優化後的價格預測函數：
    1. 隨機森林（非線性模型）替代線性回歸
    2. 多特征融合（價格/技術指標/成交量/時間）
    3. 異常值處理
    4. 輸出預測值+置信區間（95%）
    """
    try:
        # 步驟1：數據清洗（去除異常值）
        df_clean = clean_outliers(df)
        if len(df_clean) < 20:  # 數據量不足時降級為線性回歸
            st.warning("⚠️ 有效數據量不足，降級為線性回歸預測")
            pred, slope = predict_price_linear(df, days)
            conf_interval = np.zeros(days)  # 無置信區間
            return pred, slope, conf_interval
        
        # 步驟2：構建多特征數據集
        df_feat, feature_cols = prepare_features(df_clean)
        if len(feature_cols) < 3:  # 特征不足時降級
            pred, slope = predict_price_linear(df, days)
            conf_interval = np.zeros(days)
            return pred, slope, conf_interval
        
        # 步驟3：特征工程（歸一化）
        X = df_feat[feature_cols].values
        y = df_feat["Close"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 步驟4：訓練隨機森林模型（調參優化）
        model = RandomForestRegressor(
            n_estimators=100,  # 決策樹數量
            max_depth=10,      # 樹深度（避免過擬合）
            min_samples_split=5,
            random_state=42    # 固定隨機種子（可復現）
        )
        # 劃分訓練集（用80%數據訓練）
        X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        # 步驟5：生成未來特征（基於最後一條數據的特征趨勢）
        last_feat = df_feat.iloc[-1][feature_cols].values.reshape(1, -1)
        future_X = []
        for i in range(days):
            # 基於時間遞增調整特征（模擬趨勢）
            temp_feat = last_feat.copy()
            if "day_of_week" in feature_cols:
                temp_feat[0, feature_cols.index("day_of_week")] = (df_feat["day_of_week"].iloc[-1] + i) % 5
            future_X.append(temp_feat[0])
        future_X_scaled = scaler.transform(future_X)
        
        # 步驟6：預測+計算95%置信區間（體現預測不確定性）
        # 用所有決策樹的預測值計算置信區間
        tree_predictions = [tree.predict(future_X_scaled) for tree in model.estimators_]
        pred = np.mean(tree_predictions, axis=0)  # 均值作為最終預測
        pred_std = np.std(tree_predictions, axis=0)  # 標準差
        # 95%置信區間（1.96倍標準差）
        conf_interval = 1.96 * pred_std
        
        # 步驟7：計算整體趨勢（基於預測值的斜率）
        slope, _, _, _, _ = stats.linregress(range(days), pred)
        
        return pred, slope, conf_interval
    
    except Exception as e:
        st.warning(f"⚠️ 優化預測失敗，降級為基礎線性回歸：{str(e)}")
        pred, slope = predict_price_linear(df, days)
        conf_interval = np.zeros(days)  # 無置信區間
        return pred, slope, conf_interval

def predict_price_linear(df, days):
    """保留原線性回歸作為兜底"""
    df["idx"] = np.arange(len(df))
    x = df["idx"].values.reshape(-1, 1)
    y = df["Close"].values
    model = LinearRegression()
    model.fit(x, y)
    future_idx = np.arange(len(df), len(df) + days).reshape(-1, 1)
    pred = model.predict(future_idx)
    slope = model.coef_[0]
    return pred, slope

def backtest_model(df):
    """簡單回測：用歷史數據驗證模型準確率"""
    try:
        df_clean = clean_outliers(df)
        if len(df_clean) < 50:
            return "數據量不足（<50條），無法回測"
        split_idx = int(len(df_clean) * 0.9)
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        pred_test, _, _ = predict_price_optimized(train_df, len(test_df))
        # 計算平均絕對誤差（MAE）
        mae = np.mean(np.abs(pred_test - test_df["Close"].values))
        return f"回測平均誤差：{mae:.2f} HK$（誤差越小越準確）"
    except Exception as e:
        return f"回測失敗：{str(e)[:50]}"

# ================== 主執行邏輯 ==================
if st.button("🚀 開始分析（增強版）", type="primary"):
    # 輸入驗證
    if user_code != "^HSI" and (not user_code.isdigit() or len(user_code) not in [4,5]):
        st.error("❌ 格式錯誤！港股代碼必須是4-5位數字（如0700），恆生指數請輸入^HSI")
    else:
        # 獲取數據
        df = get_hk_stock_data(user_code)
        if df is None:
            st.stop()
        
        # 計算技術指標
        df = calculate_indicators(df)
        if df is None:
            st.stop()
        
        # 獲取業績數據
        st.subheader("📋 去年財務業績")
        financial_data = get_stock_financials(user_code)
        if isinstance(financial_data, pd.DataFrame):
            st.dataframe(financial_data, use_container_width=True)
        else:
            st.info(financial_data)
        
        # 計算支撐壓力位
        sup, res = calculate_support_resistance(df)
        # 優化版預測（帶置信區間）
        pred, slope, conf_interval = predict_price_optimized(df, predict_days)
        last_close = df["Close"].iloc[-1]
        
        # ========== 展示數據 ==========
        # 最新10筆數據（包含新增MA線）
        st.subheader("📊 最新交易數據（前10筆）")
        show_cols = ["Date","Open","High","Low","Close","Volume","MA5","MA20","MA30","MA50","MA100"]
        show_cols = [col for col in show_cols if col in df.columns]
        show_df = df[show_cols].tail(10)
        # 格式化數據
        format_dict = {col: 2 for col in ["Open","High","Low","Close","MA5","MA20","MA30","MA50","MA100"] if col in show_df.columns}
        if "Volume" in show_df.columns:
            format_dict["Volume"] = 0
        show_df = show_df.round(format_dict)
        st.dataframe(show_df, use_container_width=True)
        
        # 價格走勢圖（包含新增MA線）
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 價格 & 多周期均線走勢")
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(df["Date"], df["Close"], label="收盤價", color="#1f77b4", linewidth=1.5)
            ax.plot(df["Date"], df["MA5"], label="MA5（5日均線）", color="#ff7f0e", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA20"], label="MA20（20日均線）", color="#2ca02c", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA30"], label="MA30（30日均線）", color="#d62728", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA50"], label="MA50（50日均線）", color="#9467bd", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA100"], label="MA100（100日均線）", color="#8c564b", linewidth=1, alpha=0.8)
            
            # 優化圖表樣式
            ax.set_title(f"{option if user_code in hot_stocks.values() else user_code} 價格走勢", fontsize=12)
            ax.set_xlabel("日期", fontsize=10)
            ax.set_ylabel("價格 (HK$)", fontsize=10)
            ax.legend(fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            plt.xticks(rotation=45)
            plt.tight_layout()  # 自動調整布局
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
        fig_r, ax_r = plt.subplots(figsize=(10,4))
        ax_r.plot(df["Date"], df["RSI"], color="#9467bd", linewidth=1)
        ax_r.axhline(70, c="#d62728", ls="--", alpha=0.7, label="超買線(70)")
        ax_r.axhline(30, c="#2ca02c", ls="--", alpha=0.7, label="超賣線(30)")
        ax_r.axhline(50, c="#7f7f7f", ls=":", alpha=0.5, label="中軸(50)")
        ax_r.set_title("RSI 走勢（14日）", fontsize=12)
        ax_r.set_xlabel("日期", fontsize=10)
        ax_r.set_ylabel("RSI 值", fontsize=10)
        ax_r.legend(fontsize=9)
        ax_r.tick_params(axis='both', labelsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_r)
        
        # 優化版價格預測（帶置信區間）
        st.subheader(f"🔮 未來 {predict_days} 天價格預測（隨機森林+多特征）")
        trend = "📈 上漲趨勢" if slope > 0 else "📉 下跌趨勢" if slope < 0 else "📊 平盤趨勢"
        st.success(f"整體趨勢：{trend} (斜率：{slope:.6f})")
        st.info(backtest_model(df))  # 展示回測結果
        
        # 生成交易日預測日期
        last_trading_day = df["Date"].iloc[-1]
        pred_dates = get_trading_dates(last_trading_day + timedelta(days=1), predict_days)
        pred_df = pd.DataFrame({
            "預測日期": [d.strftime("%Y-%m-%d") for d in pred_dates],
            "預測價格 (HK$)": [round(p, 2) for p in pred[:len(pred_dates)]],
            "95%置信下限 (HK$)": [round(p - ci, 2) for p, ci in zip(pred[:len(pred_dates)], conf_interval[:len(pred_dates)])],
            "95%置信上限 (HK$)": [round(p + ci, 2) for p, ci in zip(pred[:len(pred_dates)], conf_interval[:len(pred_dates)])]
        })
        st.dataframe(pred_df, use_container_width=True)
        st.info(f"當前價：{last_close:.2f} HK$ → 最後預測價：{pred[-1]:.2f} HK$")
        
        # 強化風險提示
        st.warning("⚠️ 預測風險提示：")
        st.warning("1. 股價/指數受政策、資金、消息等多因素影響，預測僅為技術面參考；")
        st.warning("2. 95%置信區間代表預測波動範圍，區間越寬，不確定性越高；")
        st.warning("3. 本模型未考慮停牌、分紅、除權等港股特殊事件，僅供學習使用。")
        
        # 綜合研判
        st.subheader("📌 技術研判（僅供學習參考）")
        rsi = df["RSI"].iloc[-1]
        ma5 = df["MA5"].iloc[-1]
        ma20 = df["MA20"].iloc[-1]
        ma30 = df["MA30"].iloc[-1]
        col_advice1, col_advice2 = st.columns(2)
        with col_advice1:
            st.markdown("### 指標狀態")
            st.write(f"RSI當前值：{rsi:.1f}")
            st.write(f"MA5：{ma5:.2f} | MA20：{ma20:.2f} | MA30：{ma30:.2f}")
            st.write(f"價格/MA5：{'↑ 站穩' if last_close > ma5 else '↓ 跌破'}")
            st.write(f"MA5/MA20：{'↑ 金叉' if ma5 > ma20 else '↓ 死叉'}")
            st.write(f"MA20/MA30：{'↑ 金叉' if ma20 > ma30 else '↓ 死叉'}")
        with col_advice2:
            st.markdown("### 操作建議")
            if ma5 > ma20 and ma20 > ma30 and rsi < 65:
                st.success("✅ 多周期均線向上，趨勢強勁，可適度關注")
            elif ma5 < ma20 and ma20 < ma30:
                st.warning("⚠️ 多周期均線向下，短期趨勢偏弱，謹慎操作")
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
st.caption("3. 預測模型已升級為隨機森林+多特征融合，相比線性回歸更貼近實際走勢")
st.caption("4. 新增MA30/50/100均線、去年業績查詢、恆生指數預測功能")
st.caption("5. 若仍失敗，請檢查網絡或稍後重試（數據源臨時維護）")