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
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import stats
import matplotlib as mpl

# ================== 全局配置（解決中文顯示核心） ==================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="港股專業頂級版", layout="wide")

# 徹底解決matplotlib中文顯示問題（兼容Windows/Mac/Linux/Streamlit Cloud）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
mpl.rcParams['font.family'] = 'sans-serif'

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
st.title("📈 港股分析預測系統｜高精度版")
st.markdown("### 支持：騰訊、美團、匯豐等主流港股（預測模型升級：隨機森林+多特征+MA60/MA120）")

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

# ================== 穩定的數據獲取函數 ==================
@st.cache_data(ttl=3600)  # 緩存1小時，減少請求次數
def get_hk_stock_data(symbol):
    """
    獲取港股數據（多層次兼容+兜底+請求優化）
    :param symbol: 港股代碼（如0700）
    :return: 清洗後的DataFrame或None
    """
    # 步驟1：構建標準Yahoo Finance代碼
    yf_symbol = f"{symbol}.HK"
    st.info(f"🔍 正在獲取數據：{yf_symbol}")
    
    # 步驟2：下載數據（擴展時間範圍，增加成功率）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=4*365)  # 拉長到4年，確保MA120有足夠數據
    try:
        # 核心優化：提升港股兼容性
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
            url = f"https://query1.finance.yahoo.com/v7/finance/chart/{yf_symbol}?range=4y&interval=1d&indicators=quote&includeTimestamps=true"
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
                    'Volume': quote['volume']
                })
                # 去除空值
                df = df.dropna(subset=['Close'])
            else:
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
        if len(df) < 120:
            st.warning(f"⚠️ 有效數據僅{len(df)}條（低於120條，MA120計算結果參考性低）")
        
        st.success(f"✅ 成功獲取 {yf_symbol} 數據（共{len(df)}條）")
        return df
    
    except Exception as e:
        st.error(f"❌ 數據獲取異常：{str(e)[:100]}")
        st.info("💡 解決方案：")
        st.info("1. 刷新頁面重試（網絡/數據源臨時波動）")
        st.info("2. 確認港股代碼格式（必須是4-5位數字，如0700而非700）")
        st.info("3. 更換熱門港股測試（如騰訊0700、小米1810）")
        return None

# ================== 技術指標計算（新增MA60/MA120） ==================
def calculate_indicators(df):
    """計算技術指標（兼容缺失字段+新增MA60/MA120）"""
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    try:
        # 移動平均線（擴展到MA60/MA120，最小週期1避免空值）
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["MA60"] = df["Close"].rolling(window=60, min_periods=1).mean()  # 新增60日均線
        df["MA120"] = df["Close"].rolling(window=120, min_periods=1).mean() # 新增120日均線
        
        # MACD（優化參數）
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]  # 新增MACD柱
        
        # RSI（優化計算，避免除零錯誤+兼容少數據）
        delta = df["Close"].pct_change()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)  # 替換0避免除零
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # 布林帶（新增，提升預測特徵）
        df["BB_Mid"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["BB_Std"] = df["Close"].rolling(window=20, min_periods=1).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"].replace(0, 0.0001)
        df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"].replace(0, 0.0001)
        
        return df
    except Exception as e:
        st.warning(f"⚠️ 技術指標計算部分失敗：{str(e)}")
        return df

# ================== 支撐壓力位計算（優化） ==================
def calculate_support_resistance(df, window=60):
    """優化支撐壓力位計算（使用60天窗口，更穩定）"""
    try:
        # 多窗口綜合計算，提升準確性
        support_short = df["Low"].rolling(window=20, min_periods=1).min().iloc[-1]
        support_long = df["Low"].rolling(window=60, min_periods=1).min().iloc[-1]
        resistance_short = df["High"].rolling(window=20, min_periods=1).max().iloc[-1]
        resistance_long = df["High"].rolling(window=60, min_periods=1).max().iloc[-1]
        
        # 加權平均（長窗口權重更高）
        support = (support_short * 0.3 + support_long * 0.7)
        resistance = (resistance_short * 0.3 + resistance_long * 0.7)
        
        return round(support, 2), round(resistance, 2)
    except:
        # 兜底：用最新價格計算
        return round(df["Low"].iloc[-1], 2), round(df["High"].iloc[-1], 2)

# ================== 高精度價格預測模塊（核心優化） ==================
def clean_outliers(df, column="Close"):
    """增強版異常值處理（雙重IQR+Z-score）"""
    # 第一步：IQR處理
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # 第二步：Z-score二次過濾（僅保留±2σ範圍）
    z_scores = stats.zscore(df_clean[column])
    df_clean = df_clean[(z_scores >= -2) & (z_scores <= 2)]
    
    return df_clean

def prepare_features(df):
    """增強版特征工程（加入MA60/MA120/布林帶等新特征）"""
    df_feat = df.copy()
    
    # 基礎價格特征
    df_feat["price_change"] = df_feat["Close"].pct_change()
    df_feat["high_low_diff"] = df_feat["High"] - df_feat["Low"]
    df_feat["open_close_diff"] = df_feat["Open"] - df_feat["Close"]
    df_feat["high_close_diff"] = df_feat["High"] - df_feat["Close"]
    df_feat["low_close_diff"] = df_feat["Close"] - df_feat["Low"]
    
    # 新增移動平均線特征（MA60/MA120）
    df_feat["ma5_ma60_diff"] = df_feat["MA5"] - df_feat["MA60"]
    df_feat["ma20_ma120_diff"] = df_feat["MA20"] - df_feat["MA120"]
    df_feat["close_ma60_diff"] = df_feat["Close"] - df_feat["MA60"]
    df_feat["close_ma120_diff"] = df_feat["Close"] - df_feat["MA120"]
    df_feat["ma60_ma120_diff"] = df_feat["MA60"] - df_feat["MA120"]
    
    # 技術指標特征（擴展）
    df_feat["rsi_norm"] = df_feat["RSI"] / 100  # 歸一化RSI
    df_feat["macd_diff"] = df_feat["MACD"] - df_feat["MACD_Signal"]
    df_feat["macd_hist_norm"] = df_feat["MACD_Hist"] / df_feat["Close"].std()  # 歸一化MACD柱
    df_feat["bb_position"] = (df_feat["Close"] - df_feat["BB_Lower"]) / (df_feat["BB_Upper"] - df_feat["BB_Lower"]).replace(0, 0.0001)  # 布林帶位置
    
    # 成交量特征（增強）
    df_feat["volume_change"] = df_feat["Volume"].pct_change()
    df_feat["volume_ma5"] = df_feat["Volume"].rolling(window=5, min_periods=1).mean()
    df_feat["volume_ratio"] = df_feat["Volume"] / df_feat["volume_ma5"].replace(0, 0.0001)
    
    # 時間特征
    df_feat["day_of_week"] = df_feat["Date"].dt.weekday
    df_feat["month"] = df_feat["Date"].dt.month
    df_feat["quarter"] = df_feat["Date"].dt.quarter
    
    # 趨勢特征
    df_feat["close_trend_5"] = df_feat["Close"].rolling(window=5, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df_feat["close_trend_20"] = df_feat["Close"].rolling(window=20, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # 填充缺失值（避免模型報錯）
    df_feat = df_feat.fillna(0)
    # 去除無窮值
    df_feat = df_feat.replace([np.inf, -np.inf], 0)
    
    # 特征列篩選（僅保留數值型特征）
    feature_cols = [
        # 基礎價格特征
        "price_change", "high_low_diff", "open_close_diff", "high_close_diff", "low_close_diff",
        # MA特征（新增60/120）
        "ma5_ma20_diff", "ma5_ma60_diff", "ma20_ma120_diff", "close_ma5_diff", "close_ma20_diff",
        "close_ma60_diff", "close_ma120_diff", "ma60_ma120_diff",
        # 技術指標特征
        "rsi_norm", "macd_diff", "macd_hist_norm", "bb_position",
        # 成交量特征
        "volume_change", "volume_ratio",
        # 時間/趨勢特征
        "day_of_week", "month", "quarter", "close_trend_5", "close_trend_20"
    ]
    
    # 兼容舊版計算（避免特征缺失）
    feature_cols = [col for col in feature_cols if col in df_feat.columns]
    
    return df_feat, feature_cols

def hyperparameter_tuning(X_train, y_train):
    """超參數調優（提升模型精度）"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12, 15],
        'min_samples_split': [4, 6],
        'min_samples_leaf': [2, 3]
    }
    
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # 3折交叉驗證
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def predict_price_optimized(df, days):
    """
    高精度價格預測函數：
    1. 隨機森林（超參數調優）
    2. 多特征融合（新增MA60/MA120/布林帶等）
    3. 雙重異常值處理
    4. 輸出預測值+置信區間（95%）
    5. 模型加權融合（隨機森林+線性回歸）
    """
    try:
        # 步驟1：數據清洗（雙重異常值處理）
        df_clean = clean_outliers(df)
        if len(df_clean) < 60:  # 數據量不足時降級
            st.warning("⚠️ 有效數據量不足（<60條），降級為增強版線性回歸預測")
            pred, slope = predict_price_linear_enhanced(df, days)
            conf_interval = np.zeros(days)
            return pred, slope, conf_interval
        
        # 步驟2：構建多特征數據集
        df_feat, feature_cols = prepare_features(df_clean)
        if len(feature_cols) < 5:  # 特征不足時降級
            pred, slope = predict_price_linear_enhanced(df, days)
            conf_interval = np.zeros(days)
            return pred, slope, conf_interval
        
        # 步驟3：特征工程（歸一化）
        X = df_feat[feature_cols].values
        y = df_feat["Close"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 步驟4：劃分訓練集（用85%數據訓練，提升精度）
        X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
        
        # 步驟5：超參數調優+訓練隨機森林模型
        best_model = hyperparameter_tuning(X_train, y_train)
        
        # 步驟6：生成未來特征（基於最後一條數據的特征趨勢）
        last_feat = df_feat.iloc[-1][feature_cols].values.reshape(1, -1)
        future_X = []
        for i in range(days):
            temp_feat = last_feat.copy()
            # 基於時間遞增調整特征（模擬真實趨勢）
            if "day_of_week" in feature_cols:
                temp_feat[0, feature_cols.index("day_of_week")] = (df_feat["day_of_week"].iloc[-1] + i) % 5
            if "month" in feature_cols and (df_feat["day_of_week"].iloc[-1] + i) % 30 == 0:
                temp_feat[0, feature_cols.index("month")] = (df_feat["month"].iloc[-1] + 1) % 12
            # 模擬趨勢特征遞增
            if "close_trend_5" in feature_cols:
                trend_5 = df_feat["close_trend_5"].iloc[-1]
                temp_feat[0, feature_cols.index("close_trend_5")] = trend_5 * (1 + 0.01 * i)
            future_X.append(temp_feat[0])
        
        future_X_scaled = scaler.transform(future_X)
        
        # 步驟7：預測+計算95%置信區間
        tree_predictions = [tree.predict(future_X_scaled) for tree in best_model.estimators_]
        rf_pred = np.mean(tree_predictions, axis=0)  # 隨機森林預測值
        rf_std = np.std(tree_predictions, axis=0)    # 標準差
        conf_interval = 1.96 * rf_std               # 95%置信區間
        
        # 步驟8：線性回歸輔助預測（融合提升精度）
        lr_pred, _ = predict_price_linear_enhanced(df_clean, days)
        
        # 步驟9：加權融合預測結果（隨機森林權重0.7，線性回歸0.3）
        final_pred = 0.7 * rf_pred + 0.3 * lr_pred
        
        # 步驟10：計算整體趨勢（基於融合預測值的斜率）
        slope, _, _, _, _ = stats.linregress(range(days), final_pred)
        
        return final_pred, slope, conf_interval
    
    except Exception as e:
        st.warning(f"⚠️ 高精度預測失敗，降級為增強版線性回歸：{str(e)}")
        pred, slope = predict_price_linear_enhanced(df, days)
        conf_interval = np.zeros(days)
        return pred, slope, conf_interval

def predict_price_linear_enhanced(df, days):
    """增強版線性回歸（加入MA60/MA120特征）"""
    # 構建多特征線性回歸
    df["idx"] = np.arange(len(df))
    df["idx2"] = df["idx"] ** 2  # 二次項，捕捉非線性趨勢
    df["ma60_norm"] = df["MA60"] / df["Close"].mean()
    df["ma120_norm"] = df["MA120"] / df["Close"].mean()
    
    # 多特征輸入
    X = df[["idx", "idx2", "ma60_norm", "ma120_norm"]].values
    y = df["Close"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 生成未來特征
    future_idx = np.arange(len(df), len(df) + days)
    future_X = np.column_stack([
        future_idx,
        future_idx ** 2,
        np.full(days, df["ma60_norm"].iloc[-1]),
        np.full(days, df["ma120_norm"].iloc[-1])
    ])
    
    pred = model.predict(future_X)
    slope = np.mean(np.diff(pred))  # 基於差分計算斜率，更準確
    
    return pred, slope

def backtest_model(df):
    """增強版回測（計算多維度評估指標）"""
    try:
        df_clean = clean_outliers(df)
        if len(df_clean) < 120:
            return "數據量不足（<120條），無法回測"
        
        # 時序劃分（避免未來數據泄露）
        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        # 預測測試集
        pred_test, _, _ = predict_price_optimized(train_df, len(test_df))
        actual = test_df["Close"].values
        
        # 計算多維度評估指標
        mae = np.mean(np.abs(pred_test - actual))  # 平均絕對誤差
        rmse = np.sqrt(np.mean((pred_test - actual) ** 2))  # 均方根誤差
        mape = np.mean(np.abs((pred_test - actual) / actual)) * 100  # 平均相對誤差
        r2 = stats.pearsonr(pred_test, actual)[0] ** 2  # 決定係數
        
        return (f"回測結果（越高越準）：\n"
                f"平均絕對誤差(MAE)：{mae:.2f} HK$\n"
                f"均方根誤差(RMSE)：{rmse:.2f} HK$\n"
                f"平均相對誤差(MAPE)：{mape:.2f}%\n"
                f"決定係數(R²)：{r2:.3f}（接近1更準）")
    except Exception as e:
        return f"回測失敗：{str(e)[:50]}"

# ================== 主執行邏輯 ==================
if st.button("🚀 開始分析（高精度版）", type="primary"):
    # 輸入驗證
    if not user_code.isdigit() or len(user_code) not in [4,5]:
        st.error("❌ 港股代碼格式錯誤！必須是4-5位數字（如騰訊=0700，小米=1810）")
    else:
        # 獲取數據
        df = get_hk_stock_data(user_code)
        if df is None:
            st.stop()
        
        # 計算技術指標（含MA60/MA120）
        df = calculate_indicators(df)
        if df is None:
            st.stop()
        
        # 計算支撐壓力位（優化版）
        sup, res = calculate_support_resistance(df)
        # 高精度預測（帶置信區間）
        pred, slope, conf_interval = predict_price_optimized(df, predict_days)
        last_close = df["Close"].iloc[-1]
        
        # ========== 展示數據 ==========
        # 最新10筆數據（含MA60/MA120）
        st.subheader("📊 最新交易數據（前10筆）")
        show_df = df[["Date","Open","High","Low","Close","Volume","MA5","MA20","MA60","MA120"]].tail(10)
        show_df = show_df.round({
            "Open":2, "High":2, "Low":2, "Close":2, 
            "Volume":0, "MA5":2, "MA20":2, "MA60":2, "MA120":2
        })
        st.dataframe(show_df, use_container_width=True)
        
        # 價格走勢圖（含MA60/MA120，解決中文顯示）
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 價格 & 多周期均線走勢")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df["Date"], df["Close"], label="收盤價", color="#1f77b4", linewidth=1.5)
            ax.plot(df["Date"], df["MA5"], label="MA5（5日均線）", color="#ff7f0e", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA20"], label="MA20（20日均線）", color="#2ca02c", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA60"], label="MA60（60日均線）", color="#d62728", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA120"], label="MA120（120日均線）", color="#9467bd", linewidth=1, alpha=0.8)
            
            ax.set_title(f"{option} ({user_code}.HK) 價格走勢", fontsize=12)
            ax.set_xlabel("日期", fontsize=10)
            ax.set_ylabel("價格 (HK$)", fontsize=10)
            ax.legend(fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            plt.xticks(rotation=45)
            plt.tight_layout()  # 防止標籤重疊
            st.pyplot(fig)
        
        with col2:
            st.subheader("🛡️ 支撐 / 壓力位（60天窗口）")
            st.info(f"📉 支撐位：{sup} HK$")
            st.info(f"📈 壓力位：{res} HK$")
            if last_close < sup:
                st.success(f"當前價 {last_close:.2f} HK$：低於支撐位（超賣區間）")
            elif last_close > res:
                st.warning(f"當前價 {last_close:.2f} HK$：高於壓力位（超買區間）")
            else:
                st.info(f"當前價 {last_close:.2f} HK$：處於支撐壓力區間")
            
            # 新增均線狀態
            st.subheader("📊 均線狀態")
            ma5 = df["MA5"].iloc[-1]
            ma20 = df["MA20"].iloc[-1]
            ma60 = df["MA60"].iloc[-1]
            ma120 = df["MA120"].iloc[-1]
            
            st.write(f"MA5: {ma5:.2f} | MA20: {ma20:.2f} | MA60: {ma60:.2f} | MA120: {ma120:.2f}")
            if ma5 > ma20 > ma60 > ma120:
                st.success("✅ 多頭排列（強勢上升趨勢）")
            elif ma5 < ma20 < ma60 < ma120:
                st.error("❌ 空頭排列（強勢下跌趨勢）")
            else:
                st.info("🔍 震盪排列（方向不明）")
        
        # RSI+布林帶組合圖
        st.subheader("📊 RSI 14日 + 布林帶指標")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # 布林帶
        ax1.plot(df["Date"], df["Close"], label="收盤價", color="#1f77b4", linewidth=1)
        ax1.plot(df["Date"], df["BB_Upper"], label="布林上軌", color="#d62728", linewidth=1, linestyle="--", alpha=0.7)
        ax1.plot(df["Date"], df["BB_Mid"], label="布林中軌", color="#2ca02c", linewidth=1, linestyle="--", alpha=0.7)
        ax1.plot(df["Date"], df["BB_Lower"], label="布林下軌", color="#ff7f0e", linewidth=1, linestyle="--", alpha=0.7)
        ax1.fill_between(df["Date"], df["BB_Lower"], df["BB_Upper"], color="#1f77b4", alpha=0.1)
        ax1.set_ylabel("價格 (HK$)", fontsize=10)
        ax1.legend(fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        ax1.set_title("布林帶（20日）", fontsize=10)
        
        # RSI
        ax2.plot(df["Date"], df["RSI"], color="#9467bd", linewidth=1)
        ax2.axhline(70, c="#d62728", ls="--", alpha=0.7, label="超買線(70)")
        ax2.axhline(30, c="#2ca02c", ls="--", alpha=0.7, label="超賣線(30)")
        ax2.axhline(50, c="#7f7f7f", ls=":", alpha=0.5, label="中軸(50)")
        ax2.set_ylabel("RSI 值", fontsize=10)
        ax2.set_xlabel("日期", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.set_title("RSI 走勢（14日）", fontsize=10)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # 高精度價格預測（帶置信區間）
        st.subheader(f"🔮 未來 {predict_days} 天價格預測（隨機森林+多特征+MA60/MA120）")
        trend = "📈 強勢上漲" if slope > 0.01 else "📉 強勢下跌" if slope < -0.01 else \
                "📗 弱勢上漲" if slope > 0 else "📘 弱勢下跌" if slope < 0 else "📊 平盤震盪"
        st.success(f"整體趨勢：{trend} (斜率：{slope:.6f})")
        st.info(backtest_model(df))  # 展示增強版回測結果
        
        # 生成交易日預測日期
        last_trading_day = df["Date"].iloc[-1]
        pred_dates = get_trading_dates(last_trading_day + timedelta(days=1), predict_days)
        pred_df = pd.DataFrame({
            "預測日期": [d.strftime("%Y-%m-%d") for d in pred_dates],
            "預測價格 (HK$)": [round(p, 2) for p in pred[:len(pred_dates)]],
            "95%置信下限 (HK$)": [round(p - ci, 2) for p, ci in zip(pred[:len(pred_dates)], conf_interval[:len(pred_dates)])],
            "95%置信上限 (HK$)": [round(p + ci, 2) for p, ci in zip(pred[:len(pred_dates)], conf_interval[:len(pred_dates)])],
            "漲跌幅 (%)": [round((p / last_close - 1) * 100, 2) for p in pred[:len(pred_dates)]]
        })
        st.dataframe(pred_df, use_container_width=True)
        st.info(f"當前價：{last_close:.2f} HK$ → 最後預測價：{pred[-1]:.2f} HK$ → 預測漲跌幅：{((pred[-1]/last_close -1)*100):.2f}%")
        
        # 強化風險提示
        st.warning("⚠️ 預測風險提示：")
        st.warning("1. 股價受政策、資金、消息等多因素影響，預測僅為技術面參考；")
        st.warning("2. 95%置信區間代表預測波動範圍，區間越寬，不確定性越高；")
        st.warning("3. 本模型未考慮停牌、分紅、除權等港股特殊事件，僅供學習使用；")
        st.warning("4. MA60/MA120反映中長期趨勢，短期預測仍存在較大波動風險。")
        
        # 綜合研判（增強版）
        st.subheader("📌 技術研判（僅供學習參考）")
        rsi = df["RSI"].iloc[-1]
        ma5 = df["MA5"].iloc[-1]
        ma20 = df["MA20"].iloc[-1]
        ma60 = df["MA60"].iloc[-1]
        ma120 = df["MA120"].iloc[-1]
        bb_position = df["bb_position"].iloc[-1] if "bb_position" in df.columns else 0.5
        
        col_advice1, col_advice2 = st.columns(2)
        with col_advice1:
            st.markdown("### 核心指標狀態")
            st.write(f"RSI當前值：{rsi:.1f}（正常區間：30-70）")
            st.write(f"布林帶位置：{bb_position:.2f}（0=下軌，1=上軌）")
            st.write(f"價格/MA5：{'↑ 站穩' if last_close > ma5 else '↓ 跌破'}")
            st.write(f"MA5/MA20：{'↑ 金叉' if ma5 > ma20 else '↓ 死叉'}")
            st.write(f"MA20/MA60：{'↑ 金叉' if ma20 > ma60 else '↓ 死叉'}")
            st.write(f"MA60/MA120：{'↑ 金叉' if ma60 > ma120 else '↓ 死叉'}")
        
        with col_advice2:
            st.markdown("### 綜合操作建議")
            # 多維度研判
            conditions = [
                ma5 > ma20 > ma60 > ma120 and rsi < 65 and bb_position < 0.8,
                ma5 < ma20 < ma60 < ma120,
                rsi > 75 or bb_position > 0.9,
                rsi < 25 or bb_position < 0.1,
                (ma5 > ma20 and ma20 < ma60) or (ma5 < ma20 and ma20 > ma60)
            ]
            advices = [
                "✅ 多頭趨勢確立，可適度跟進（中長期看好）",
                "❌ 空頭趨勢明顯，建議規避（中長期看空）",
                "⚠️ 超買嚴重，短期回調風險高，建議減倉",
                "✅ 超賣嚴重，短期反彈機會大，可輕倉布局",
                "🔍 多空分歧，震盪為主，建議觀察或波段操作"
            ]
            advice = next((adv for cond, adv in zip(conditions, advices) if cond), "🔍 趨勢不明，建議觀察為主")
            st.write(advice)

# ================== 底部提示 ==================
st.divider()
st.caption("⚠️ 重要提示：")
st.caption("1. 本工具僅供編程學習使用，不構成任何投資建議")
st.caption("2. 數據來源為Yahoo Finance，請以港交所官方數據為準")
st.caption("3. 預測模型升級為：隨機森林超參數調優+多特征融合+MA60/MA120+布林帶")
st.caption("4. 若仍失敗，請檢查網絡或稍後重試（數據源臨時維護）")
st.caption("5. 中文顯示已優化，兼容Windows/Mac/Linux/Streamlit Cloud環境")