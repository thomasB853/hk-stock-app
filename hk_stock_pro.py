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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import stats
import matplotlib as mpl
# LSTM时序模型依赖
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ================== 全局配置（彻底解决中文显示+TensorFlow优化） ==================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="港股專業頂級版", layout="wide")
# 彻底解决matplotlib中文显示（兼容所有系统/Streamlit Cloud）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['figure.autolayout'] = True  # 自动适配布局，防止标签截断
# TensorFlow显存优化（避免显存溢出）
tf.config.set_soft_device_placement(True)
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0] if tf.config.list_physical_devices('GPU') else tf.config.list_physical_devices('CPU')[0], True)

# ================== 依赖检查&强制升级（新增TensorFlow） ==================
def install_package(pkg_name, pkg_version=""):
    """统一安装/升级依赖函数"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if pkg_version:
        cmd.append(f"{pkg_name}>={pkg_version}")
    else:
        cmd.append(pkg_name)
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 检查yfinance
try:
    import yfinance as yf
    if hasattr(yf, '__version__') and yf.__version__ < "0.2.31":
        st.warning("⚠️ yfinance版本過舊，正在自動升級至最新版...")
        install_package("yfinance", "0.2.31")
        importlib.reload(yf)
except ImportError:
    st.error("❌ 缺少yfinance庫，正在自動安裝...")
    install_package("yfinance", "0.2.31")
    import yfinance as yf

# 检查scikit-learn
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.error("❌ 缺少scikit-learn庫，正在自動安裝...")
    install_package("scikit-learn", "1.3.0")
    from sklearn.linear_model import LinearRegression

# 检查TensorFlow（LSTM依赖）
try:
    import tensorflow as tf
except ImportError:
    st.warning("⚠️ 缺少TensorFlow庫，正在安裝（LSTM模型依赖）...")
    install_package("tensorflow", "2.15.0")
    import tensorflow as tf

# ================== 页面UI ==================
st.title("📈 港股分析預測系統｜超精準版")
st.markdown("### 多模型融合预测+全周期均線（MA5/20/30/50/60/120）｜支持騰訊/美團/匯豐等主流港股")
st.markdown("#### 核心模型：LSTM时序模型+随机森林+增强线性回归｜多特征融合+时序趋势挖掘")

# 热门港股
hot_stocks = {
    "騰訊控股 (0700)": "0700",
    "美團-W (3690)": "3690",
    "匯豐控股 (0005)": "0005",
    "小米集團-W (1810)": "1810",
    "阿里巴巴-SW (9988)": "9988",
    "工商銀行 (1398)": "1398",
    "京東集團-SW (9618)": "9618",
    "快手-W (1024)": "1024"
}
option = st.selectbox("選擇熱門港股（數據穩定）", list(hot_stocks.keys()))
default_code = hot_stocks[option]
user_code = st.text_input("手動輸入港股代碼（4-5位數字，如0700）", default_code).strip()
predict_days = st.slider("預測天數（1-15天）", 1, 15, 5)
# 新增模型选择（让用户可选单模型/融合模型）
model_choice = st.radio("選擇預測模型", ["多模型融合（最精準）", "LSTM時序模型（短期趨勢）", "隨機森林（多特征）"], index=0)

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
    """列名清洗：兼容yfinance所有格式"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    # 标准列名映射
    column_mapping = {
        'date': 'Date', 'datetime': 'Date', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'adj close': 'Adj Close', 'adj_close': 'Adj Close',
        'volume': 'Volume', 'vol': 'Volume'
    }
    final_cols = {}
    for col in df.columns:
        for key in column_mapping.keys():
            if key in col:
                final_cols[col] = column_mapping[key]
                break
    df.rename(columns=final_cols, inplace=True)
    return df

# ================== 穩定的數據獲取函數（拉長至5年，适配长周期均線） ==================
@st.cache_data(ttl=3600)
def get_hk_stock_data(symbol):
    """獲取港股數據：5年數據+雙接口兜底+數據清洗"""
    yf_symbol = f"{symbol}.HK"
    st.info(f"🔍 正在獲取{yf_symbol}5年交易數據...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5年數據，适配MA120+长周期特征
    
    try:
        # 主接口：yfinance下载
        df = yf.download(
            yf_symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"),
            progress=False, timeout=80, threads=False, auto_adjust=False, back_adjust=False, repair=True
        )
        # 空数据兜底：直接调用Yahoo Finance原生接口
        if df.empty or len(df) < 20:
            st.warning("⚠️ 默認接口獲取失敗，嘗試原生接口...")
            url = f"https://query1.finance.yahoo.com/v7/finance/chart/{yf_symbol}?range=5y&interval=1d&indicators=quote&includeTimestamps=true"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
            resp = requests.get(url, headers=headers, timeout=80)
            data = resp.json()
            if 'chart' in data and 'result' in data['chart'] and len(data['chart']['result'])>0:
                ts = data['chart']['result'][0]['timestamp']
                quote = data['chart']['result'][0]['indicators']['quote'][0]
                df = pd.DataFrame({
                    'Date': [datetime.fromtimestamp(t) for t in ts],
                    'Open': quote['open'], 'High': quote['high'], 'Low': quote['low'],
                    'Close': quote['close'], 'Volume': quote['volume']
                })
                df = df.dropna(subset=['Close'])
            else:
                st.error(f"❌ 未獲取到{yf_symbol}數據（代碼錯誤/停牌/未上市）")
                return None
        
        # 数据清洗核心步骤
        df.reset_index(inplace=True)
        df = clean_column_names(df)
        # 缺失列补全
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"⚠️ 部分字段缺失：{missing_cols}，正在補全...")
            if "Date" not in df.columns: st.error("❌ 核心字段Date缺失"); return None
            if "Close" in df.columns:
                for col in ["Open", "High", "Low"]:
                    if col not in df.columns: df[col] = df["Close"]
            else: st.error("❌ 核心字段Close缺失"); return None
            if "Volume" not in df.columns: df["Volume"] = 0
        # 最终清洗
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        # 数据量检查
        if len(df) < 150:
            st.warning(f"⚠️ 有效數據僅{len(df)}條（低於150條，長周期均線/模型預測參考性降低）")
        st.success(f"✅ 成功獲取{yf_symbol}數據（共{len(df)}條，時間範圍：{df['Date'].iloc[0].strftime('%Y-%m-%d')}至{df['Date'].iloc[-1].strftime('%Y-%m-%d')}）")
        return df
    except Exception as e:
        st.error(f"❌ 數據獲取異常：{str(e)[:120]}")
        st.info("💡 解決方案：1.刷新頁面 2.確認港股代碼（4-5位數字）3.更換熱門股測試")
        return None

# ================== 技術指標計算（新增MA30/MA50+全周期均線+增强技术指标） ==================
def calculate_indicators(df):
    """計算技術指標：MA5/20/30/50/60/120 + MACD/RSI/布林帶/成交量指標/均線交叉"""
    if df is None or len(df) == 0: return None
    df = df.copy()
    try:
        # 核心：全周期移動平均線（新增MA30/MA50）
        ma_windows = [5,20,30,50,60,120]
        for window in ma_windows:
            df[f"MA{window}"] = df["Close"].rolling(window=window, min_periods=1).mean()
        
        # MACD（增强：加入MACD柱归一化）
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        df["MACD_Hist_Norm"] = df["MACD_Hist"] / df["Close"].rolling(window=20, min_periods=1).std().replace(0, 0.0001)
        
        # RSI（14日，避免除零）
        delta = df["Close"].pct_change()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # 布林帶（20日，趋势判断）
        df["BB_Mid"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["BB_Std"] = df["Close"].rolling(window=20, min_periods=1).std().replace(0, 0.0001)
        df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
        df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
        df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
        
        # 成交量指标（成交量MA+量比）
        df["Vol_MA5"] = df["Volume"].rolling(window=5, min_periods=1).mean()
        df["Vol_MA20"] = df["Volume"].rolling(window=20, min_periods=1).mean()
        df["Vol_Ratio"] = df["Volume"] / df["Vol_MA5"].replace(0, 0.0001)
        
        # 关键：均線交叉特征（预测核心特征，反映趋势变化）
        df["MA5_MA20_Cross"] = (df["MA5"] > df["MA20"]).astype(int)  # 5/20金叉=1，死叉=0
        df["MA20_MA30_Cross"] = (df["MA20"] > df["MA30"]).astype(int)
        df["MA30_MA50_Cross"] = (df["MA30"] > df["MA50"]).astype(int)
        df["MA50_MA60_Cross"] = (df["MA50"] > df["MA60"]).astype(int)
        df["MA60_MA120_Cross"] = (df["MA60"] > df["MA120"]).astype(int)
        # 均線价差（归一化，反映趋势强度）
        df["MA5_MA20_Diff_Norm"] = (df["MA5"] - df["MA20"]) / df["Close"]
        df["MA30_MA50_Diff_Norm"] = (df["MA30"] - df["MA50"]) / df["Close"]
        df["MA60_MA120_Diff_Norm"] = (df["MA60"] - df["MA120"]) / df["Close"]
        
        # 价格趋势特征（斜率，反映短期涨跌幅）
        for window in [5,20,30,50]:
            df[f"Close_Trend_{window}"] = df["Close"].rolling(window=window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
            )
        
        # 基础价格特征
        df["Price_Change"] = df["Close"].pct_change()
        df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
        df["Open_Close_Diff"] = (df["Open"] - df["Close"]) / df["Close"]
        
        # 时间特征（时序模型核心）
        df["Day_Of_Week"] = df["Date"].dt.weekday
        df["Month"] = df["Date"].dt.month
        df["Quarter"] = df["Date"].dt.quarter
        df["Day_Of_Month"] = df["Date"].dt.day
        
        # 缺失值/无穷值处理
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        return df
    except Exception as e:
        st.warning(f"⚠️ 技術指標計算部分失敗：{str(e)[:80]}")
        return df

# ================== 支撐壓力位計算（多窗口融合+长周期均線辅助） ==================
def calculate_support_resistance(df):
    """多窗口融合計算支撐壓力位：結合短期20/30天+中期50/60天+長期120天"""
    try:
        # 不同周期高低点
        low_windows = [20,30,50,60,120]
        high_windows = [20,30,50,60,120]
        supports = [df["Low"].rolling(window=w, min_periods=1).min().iloc[-1] for w in low_windows]
        resistances = [df["High"].rolling(window=w, min_periods=1).max().iloc[-1] for w in high_windows]
        # 加权平均（长周期权重更高，更贴合实际趋势）
        weights = [0.1,0.15,0.2,0.25,0.3]  # 120天权重30%，20天10%
        support = round(np.average(supports, weights=weights), 2)
        resistance = round(np.average(resistances, weights=weights), 2)
        # 用MA60/MA120二次修正（中长周期趋势支撑）
        ma60 = df["MA60"].iloc[-1]
        ma120 = df["MA120"].iloc[-1]
        support = max(support, min(ma60, ma120) * 0.98)  # 不低于长周期均線的98%
        resistance = min(resistance, max(ma60, ma120) * 1.02)  # 不高于长周期均線的102%
        return support, resistance
    except:
        # 兜底：最新高低点+MA60辅助
        return round(df["Low"].iloc[-5:].min(),2), round(df["High"].iloc[-5:].max(),2)

# ================== 异常值处理（三重过滤：IQR+Z-Score+价格波动率） ==================
def clean_outliers(df):
    """三重异常值过滤：彻底去除极端价格对模型的干扰"""
    df_clean = df.copy()
    # 1. IQR过滤（价格）
    q1, q3 = df_clean["Close"].quantile([0.05, 0.95])  # 缩小区间，更严格
    iqr = q3 - q1
    df_clean = df_clean[(df_clean["Close"] >= q1 - 1.2*iqr) & (df_clean["Close"] <= q3 + 1.2*iqr)]
    # 2. Z-Score过滤（价格涨跌幅）
    df_clean["Price_Change_Abs"] = abs(df_clean["Price_Change"])
    z_scores = stats.zscore(df_clean["Price_Change_Abs"])
    df_clean = df_clean[(z_scores >= -2) & (z_scores <= 2)]
    # 3. 波动率过滤（去除单日涨跌幅超过15%的极端值）
    df_clean = df_clean[abs(df_clean["Price_Change"]) < 0.15]
    return df_clean.reset_index(drop=True)

# ================== 特征工程（全维度特征+时序特征提取） ==================
def prepare_features(df):
    """提取全维度特征：均線+技术指标+成交量+时序+趋势+交叉特征"""
    df_feat = df.copy()
    # 筛选核心数值特征（排除日期/非数值列）
    feature_cols = [
        # 价格基础特征
        "Price_Change", "High_Low_Range", "Open_Close_Diff",
        # 全周期均線归一化价差
        "MA5_MA20_Diff_Norm", "MA20_MA30_Diff_Norm", "MA30_MA50_Diff_Norm",
        "MA50_MA60_Diff_Norm", "MA60_MA120_Diff_Norm",
        # 均線交叉特征
        "MA5_MA20_Cross", "MA20_MA30_Cross", "MA30_MA50_Cross",
        "MA50_MA60_Cross", "MA60_MA120_Cross",
        # 技术指标
        "RSI", "MACD", "MACD_Signal", "MACD_Hist_Norm", "BB_Position",
        # 成交量指标
        "Vol_Ratio", "Volume", "Vol_MA5", "Vol_MA20",
        # 价格趋势斜率
        "Close_Trend_5", "Close_Trend_20", "Close_Trend_30", "Close_Trend_50",
        # 时间特征
        "Day_Of_Week", "Month", "Quarter", "Day_Of_Month"
    ]
    # 确保特征列存在
    feature_cols = [col for col in feature_cols if col in df_feat.columns]
    # 特征归一化（提升模型收敛性）
    scaler = StandardScaler()
    df_feat[feature_cols] = scaler.fit_transform(df_feat[feature_cols])
    return df_feat, feature_cols, scaler

# ================== LSTM时序模型（短期趋势预测核心，适配股价时序特性） ==================
def create_lstm_model(input_shape):
    """构建LSTM模型：适配股价时序预测，防止过拟合"""
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(units=32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def lstm_predict(df, predict_days, seq_len=60):
    """LSTM时序预测：基于历史价格序列预测未来价格"""
    # 数据准备：仅用收盘价（时序模型核心），归一化
    data = df[["Close"]].values
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    # 构建时序序列
    X = []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    # 训练LSTM模型
    model = create_lstm_model((X.shape[1], 1))
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, data_scaled[seq_len:], batch_size=32, epochs=20, callbacks=[early_stop], verbose=0)
    # 预测未来：基于最后seq_len个数据迭代预测
    last_seq = data_scaled[-seq_len:]
    lstm_pred = []
    for _ in range(predict_days):
        last_seq_reshaped = np.reshape(last_seq, (1, seq_len, 1))
        pred = model.predict(last_seq_reshaped, verbose=0)
        lstm_pred.append(pred[0,0])
        # 更新序列：滑动窗口
        last_seq = np.append(last_seq[1:], pred, axis=0)
    # 反归一化，还原真实价格
    lstm_pred = scaler.inverse_transform(np.array(lstm_pred).reshape(-1,1)).flatten()
    return lstm_pred

# ================== 随机森林模型（超参调优+多特征融合） ==================
def rf_predict(df, feature_cols, predict_days, scaler):
    """随机森林预测：超参调优+多特征融合，捕捉特征间非线性关系"""
    X = df[feature_cols].values
    y = df["Close"].values
    # 超参调优（网格搜索）
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [10,12,15],
        'min_samples_split': [3,4,5],
        'min_samples_leaf': [1,2]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                               param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=0)
    grid_search.fit(X, y)
    best_rf = grid_search.best_estimator_
    # 生成未来特征：基于最后一条数据的特征，模拟时序变化
    last_feat = df[feature_cols].iloc[-1].values.reshape(1, -1)
    future_feat = []
    for i in range(predict_days):
        temp_feat = last_feat.copy()
        # 时间特征时序更新
        temp_feat[0, feature_cols.index("Day_Of_Week")] = (temp_feat[0, feature_cols.index("Day_Of_Week")] + i) % 5
        future_feat.append(temp_feat[0])
    future_feat = scaler.transform(np.array(future_feat))
    # 预测
    rf_pred = best_rf.predict(future_feat)
    return rf_pred

# ================== 增强线性回归（多特征+二次项，兜底基础预测） ==================
def lr_predict(df, feature_cols, predict_days):
    """增强线性回归：多特征+二次项，捕捉线性趋势，作为融合模型兜底"""
    X = df[feature_cols].values
    y = df["Close"].values
    # 加入二次项，提升非线性拟合能力
    X = np.hstack([X, X**2])
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X, y)
    # 生成未来特征
    last_feat = df[feature_cols].iloc[-1].values.reshape(1, -1)
    future_feat = []
    for i in range(predict_days):
        temp_feat = last_feat.copy()
        temp_feat[0, feature_cols.index("Day_Of_Week")] = (temp_feat[0, feature_cols.index("Day_Of_Week")] + i) % 5
        future_feat.append(temp_feat[0])
    future_feat = np.hstack([np.array(future_feat), np.array(future_feat)**2])
    # 预测
    lr_pred = lr.predict(future_feat)
    return lr_pred

# ================== 多模型融合预测（核心：加权融合LSTM+RF+LR，最精準） ==================
def ensemble_predict(df, feature_cols, scaler, predict_days):
    """多模型加权融合：LSTM(0.5)+随机森林(0.3)+线性回归(0.2)，兼顾时序/特征/线性趋势"""
    try:
        # 分别获取各模型预测结果
        lstm_pred = lstm_predict(df, predict_days)
        rf_pred = rf_predict(df, feature_cols, predict_days, scaler)
        lr_pred = lr_predict(df, feature_cols, predict_days)
        # 加权融合（LSTM权重最高，因为股价是时序数据）
        ensemble_pred = 0.5 * lstm_pred + 0.3 * rf_pred + 0.2 * lr_pred
        # 趋势修正：基于均線趋势调整预测值（避免偏离实际趋势）
        ma60 = df["MA60"].iloc[-1]
        ma120 = df["MA120"].iloc[-1]
        trend = 1 if df["MA5"].iloc[-1] > df["MA120"].iloc[-1] else 0.98
        ensemble_pred = ensemble_pred * trend
        # 上下限修正：不低于支撑位，不高于压力位
        sup, res = calculate_support_resistance(df)
        ensemble_pred = np.clip(ensemble_pred, sup * 0.95, res * 1.05)
        return ensemble_pred, lstm_pred, rf_pred, lr_pred
    except Exception as e:
        st.warning(f"⚠️ 多模型融合失敗，切換為LSTM單模型：{str(e)[:80]}")
        lstm_pred = lstm_predict(df, predict_days)
        return lstm_pred, lstm_pred, lstm_pred, lstm_pred

# ================== 回测函数（多维度评估：MAE/MAPE/R²/胜率，精准判断模型效果） ==================
def backtest(df, feature_cols, scaler, predict_days=5):
    """模型回测：用历史数据验证预测效果，输出多维度评估指标"""
    try:
        df_clean = clean_outliers(df)
        if len(df_clean) < 300:
            return "📊 數據量不足（<300條），無法執行回測"
        # 时序划分：前80%训练，后20%测试（避免未来数据泄露）
        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        # 预测测试集
        if len(test_df) < predict_days:
            return f"📊 測試集數據不足（僅{len(test_df)}條），無法回測"
        # 融合模型预测
        pred, _, _, _ = ensemble_predict(train_df, feature_cols, scaler, len(test_df))
        actual = test_df["Close"].values
        # 计算多维度评估指标
        mae = round(np.mean(np.abs(pred - actual)), 2)  # 平均绝对误差
        mape = round(np.mean(np.abs((pred - actual)/actual)) * 100, 2)  # 平均相对误差
        r2 = round(stats.pearsonr(pred, actual)[0] ** 2, 3)  # 决定系数（越接近1越准）
        # 胜率：预测涨跌幅方向正确的比例
        pred_change = np.diff(pred)
        actual_change = np.diff(actual)
        win_rate = round(np.sum((pred_change * actual_change) > 0) / len(pred_change) * 100, 1) if len(pred_change) > 0 else 0
        # 输出结果
        return (
            f"📊 模型回測結果（測試集{len(test_df)}條）\n"
            f"✅ 平均絕對誤差(MAE)：{mae} HK$\n"
            f"✅ 平均相對誤差(MAPE)：{mape}%\n"
            f"✅ 決定係數(R²)：{r2}（接近1更精準）\n"
            f"✅ 漲跌方向預測勝率：{win_rate}%"
        )
    except Exception as e:
        return f"📊 回測失敗：{str(e)[:60]}"

# ================== 主執行邏輯 ==================
if st.button("🚀 開始分析（超精準版）", type="primary", use_container_width=True):
    # 输入验证
    if not user_code.isdigit() or len(user_code) not in [4,5]:
        st.error("❌ 港股代碼格式錯誤！必須是4-5位數字（如騰訊=0700，小米=1810）")
    else:
        # 1. 获取数据
        df = get_hk_stock_data(user_code)
        if df is None: st.stop()
        # 2. 计算技术指标（含MA30/MA50）
        df = calculate_indicators(df)
        if df is None: st.stop()
        # 3. 数据清洗（异常值过滤）
        df_clean = clean_outliers(df)
        # 4. 特征工程
        df_feat, feature_cols, scaler = prepare_features(df_clean)
        if len(feature_cols) < 10:
            st.warning("⚠️ 有效特征不足，模型预测精度降低")
        # 5. 计算支撑压力位
        sup, res = calculate_support_resistance(df)
        last_close = df["Close"].iloc[-1]
        # 6. 执行预测
        st.subheader("🔮 價格預測計算中...（多模型融合需數秒，請耐心等待）")
        if model_choice == "多模型融合（最精準）":
            pred, lstm_pred, rf_pred, lr_pred = ensemble_predict(df_clean, feature_cols, scaler, predict_days)
            pred_title = "多模型融合（LSTM+隨機森林+線性回歸）"
        elif model_choice == "LSTM時序模型（短期趨勢）":
            pred = lstm_predict(df_clean, predict_days)
            pred_title = "LSTM時序模型（專注短期趨勢）"
            lstm_pred = rf_pred = lr_pred = pred
        else:
            pred = rf_predict(df_clean, feature_cols, predict_days, scaler)
            pred_title = "隨機森林模型（多特征融合）"
            lstm_pred = rf_pred = lr_pred = pred
        # 计算趋势斜率（判断涨跌强度）
        slope = round(stats.linregress(range(predict_days), pred)[0], 6)
        # 7. 生成预测交易日
        last_trading_day = df["Date"].iloc[-1]
        pred_dates = get_trading_dates(last_trading_day + timedelta(days=1), predict_days)
        # 8. 计算涨跌幅
        pred_change = [round((p / last_close - 1) * 100, 2) for p in pred]
        lstm_change = [round((p / last_close - 1) * 100, 2) for p in lstm_pred]
        rf_change = [round((p / last_close - 1) * 100, 2) for p in rf_pred]

        # ========== 数据展示 ==========
        # 最新交易数据（含全周期均線）
        st.subheader("📊 最新交易數據（含全周期均線）")
        show_cols = ["Date","Open","High","Low","Close","Volume","MA5","MA20","MA30","MA50","MA60","MA120"]
        show_cols = [col for col in show_cols if col in df.columns]
        show_df = df[show_cols].tail(10)
        show_df = show_df.round({col:2 for col in show_df.columns if col not in ["Date","Volume"]} | {"Volume":0})
        st.dataframe(show_df, use_container_width=True)

        # 价格+全周期均線走势（中文正常显示）
        st.subheader("📈 價格 & 全周期均線走勢（MA5/20/30/50/60/120）")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df["Date"], df["Close"], label="收盤價", color="#1f77b4", linewidth=2, zorder=5)
        # 不同均線不同颜色/线型，区分短/中/长周期
        ma_style = {
            "MA5": ("#ff7f0e", 1.5, "-"), "MA20": ("#2ca02c", 1.5, "-"),
            "MA30": ("#d62728", 1.2, "--"), "MA50": ("#9467bd", 1.2, "--"),
            "MA60": ("#8c564b", 1.0, ":"), "MA120": ("#e377c2", 1.0, ":")
        }
        for ma, (color, lw, ls) in ma_style.items():
            if ma in df.columns:
                ax.plot(df["Date"], df[ma], label=ma, color=color, linewidth=lw, linestyle=ls, alpha=0.8)
        ax.set_title(f"{option} ({user_code}.HK) 價格&全周期均線走勢", fontsize=14, pad=20)
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("價格 (HK$)", fontsize=12)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3, zorder=0)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # 支撑压力位+均線状态
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🛡️ 支撐/壓力位（多窗口融合）")
            st.info(f"📉 支撐位：{sup} HK$")
            st.info(f"📈 壓力位：{res} HK$")
            # 价格位置判断
            if last_close < sup * 0.99:
                st.success(f"當前價 {last_close:.2f} HK$：超賣區間（低於支撐位）")
            elif last_close > res * 1.01:
                st.warning(f"當前價 {last_close:.2f} HK$：超買區間（高於壓力位）")
            else:
                st.info(f"當前價 {last_close:.2f} HK$：正常區間（支撐壓力之間）")
        with col2:
            st.subheader("📊 全周期均線狀態")
            ma5,ma20,ma30,ma50,ma60,ma120 = [df[f"MA{x}"].iloc[-1] for x in [5,20,30,50,60,120]]
            st.write(f"MA5:{ma5:.2f} | MA20:{ma20:.2f} | MA30:{ma30:.2f}")
            st.write(f"MA50:{ma50:.2f} | MA60:{ma60:.2f} | MA120:{ma120:.2f}")
            # 均線排列判断
            if ma5>ma20>ma30>ma50>ma60>ma120:
                st.success("✅ 強勢多頭排列（中長期上漲趨勢）")
            elif ma5<ma20<ma30<ma50<ma60<ma120:
                st.error("❌ 強勢空頭排列（中長期下跌趨勢）")
            elif ma30>ma50>ma60>ma120 and ma5>ma20:
                st.success("📗 弱勢多頭排列（短期偏多）")
            elif ma30<ma50<ma60<ma120 and ma5<ma20:
                st.error("📘 弱勢空頭排列（短期偏空）")
            else:
                st.info("🔍 震盪排列（多空分歧，方向不明）")

        # 技术指标组合图（RSI+MACD+布林帶）
        st.subheader("📊 核心技術指標組合（RSI+MACD+布林帶）")
        fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(14,9), sharex=True)
        # 布林帶
        ax1.plot(df["Date"], df["Close"], color="#1f77b4", linewidth=1, label="收盤價")
        ax1.plot(df["Date"], df["BB_Upper"], color="#d62728", linestyle="--", alpha=0.7, label="布林上軌")
        ax1.plot(df["Date"], df["BB_Mid"], color="#2ca02c", linestyle="--", alpha=0.7, label="布林中軌")
        ax1.plot(df["Date"], df["BB_Lower"], color="#ff7f0e", linestyle="--", alpha=0.7, label="布林下軌")
        ax1.fill_between(df["Date"], df["BB_Lower"], df["BB_Upper"], color="#1f77b4", alpha=0.1)
        ax1.set_ylabel("布林帶 (HK$)", fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        # RSI
        ax2.plot(df["Date"], df["RSI"], color="#9467bd", linewidth=1)
        ax2.axhline(70, color="#d62728", linestyle="--", alpha=0.7, label="超買線70")
        ax2.axhline(30, color="#2ca02c", linestyle="--", alpha=0.7, label="超賣線30")
        ax2.axhline(50, color="#7f7f7f", linestyle=":", alpha=0.5, label="中軸50")
        ax2.fill_between(df["Date"], 30, 70, color="#9467bd", alpha=0.1)
        ax2.set_ylabel("RSI (14日)", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        # MACD
        ax3.plot(df["Date"], df["MACD"], color="#1f77b4", linewidth=1, label="MACD")
        ax3.plot(df["Date"], df["MACD_Signal"], color="#d62728", linewidth=1, label="Signal")
        ax3.bar(df["Date"], df["MACD_Hist"], color="#2ca02c" if df["MACD_Hist"].iloc[-1]>0 else "#d62728", alpha=0.5, label="MACD柱")
        ax3.axhline(0, color="#7f7f7f", linestyle=":", alpha=0.5)
        ax3.set_ylabel("MACD", fontsize=10)
        ax3.set_xlabel("日期", fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # 核心：价格预测结果（含多模型对比）
        st.subheader(f"🔮 未來{predict_days}天價格預測｜{pred_title}")
        # 趋势判断
        if slope > 0.02:
            trend = "📈 強勢上漲"
        elif slope > 0:
            trend = "📗 弱勢上漲"
        elif slope < -0.02:
            trend = "📉 強勢下跌"
        elif slope < 0:
            trend = "📘 弱勢下跌"
        else:
            trend = "📊 平盤震盪"
        st.success(f"整體趨勢：{trend} | 趨勢斜率：{slope:.6f}")
        # 回测结果
        st.info(backtest(df, feature_cols, scaler))
        # 预测数据框（含多模型对比+涨跌幅）
        pred_df = pd.DataFrame({
            "預測交易日": [d.strftime("%Y-%m-%d") for d in pred_dates],
            "融合模型預測价(HK$)": [round(p,2) for p in pred],
            "漲跌幅(%)": pred_change,
            "LSTM預測价(HK$)": [round(p,2) for p in lstm_pred],
            "LSTM漲跌幅(%)": lstm_change,
            "隨機森林預測价(HK$)": [round(p,2) for p in rf_pred],
            "隨機森林漲跌幅(%)": rf_change
        })
        st.dataframe(pred_df, use_container_width=True)
        # 预测总结
        final_pred = pred[-1]
        final_change = round((final_pred / last_close - 1) * 100, 2)
        st.info(f"📌 預測總結：當前價{last_close:.2f} HK$ → 最後預測價{final_pred:.2f} HK$ → 整體預測漲跌幅{final_change}%")

        # 综合技术研判
        st.subheader("📌 綜合技術研判（僅供學習參考）")
        rsi = df["RSI"].iloc[-1]
        bb_pos = df["BB_Position"].iloc[-1]
        macd_cross = 1 if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1] else 0
        ma_cross = df["MA60_MA120_Cross"].iloc[-1]
        # 多维度研判
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            st.markdown("### 📋 核心指標狀態")
            st.write(f"RSI指標：{rsi:.1f}（30=超賣，70=超買）")
            st.write(f"布林帶位置：{bb_pos:.2f}（0=下軌，1=上軌）")
            st.write(f"MACD交叉：{'金叉（看多）' if macd_cross else '死叉（看空）'}")
            st.write(f"MA60/MA120交叉：{'金叉（中長期看多）' if ma_cross else '死叉（中長期看空）'}")
            st.write(f"當前價/MA60：{'站穩（偏多）' if last_close>ma60 else '跌破（偏空）'}")
            st.write(f"當前價/MA120：{'站穩（偏多）' if last_close>ma120 else '跌破（偏空）'}")
        with col_adv2:
            st.markdown("### 🎯 操作建議（僅供學習）")
            # 综合判断逻辑
            if ma_cross and macd_cross and rsi < 65 and bb_pos < 0.8:
                st.success("✅ 多維度看多：中長期趨勢向上+短期技術指標配合，可適度跟進")
            elif not ma_cross and not macd_cross and rsi > 35 and bb_pos > 0.2:
                st.error("❌ 多維度看空：中長期趨勢向下+短期技術指標配合，建議規避")
            elif rsi > 75 or bb_pos > 0.95:
                st.warning("⚠️ 短期超買：RSI/布林帶進入超買區，注意回調風險，建議減倉")
            elif rsi < 25 or bb_pos < 0.05:
                st.success("✅ 短期超賣：RSI/布林帶進入超賣區，存在反彈機會，可輕倉布局")
            else:
                st.info("🔍 震盪整理：多空指標分歧，趨勢不明，建議觀察為主，不宜追漲殺跌")

        # 强风险提示
        st.warning("⚠️ 極重要風險提示", icon="❗")
        st.warning("1. 本工具僅供編程/量化學習使用，**不構成任何投資建議/操作依據**；")
        st.warning("2. 數據來源為Yahoo Finance，請以港交所官方發布的數據為準；")
        st.warning("3. 模型預測基於技術指標/歷史數據，未考慮政策/消息/資金等市場突發因素；")
        st.warning("4. 港股實行T+0+無漲跌幅限制，交易風險極高，請謹慎參與；")
        st.warning("5. 預測結果存在誤差，隨預測天數增加，精度會逐漸降低。")

# ================== 底部信息 ==================
st.divider()
st.caption("📌 港股分析預測系統｜超精準版")
st.caption("✅ 核心特性：LSTM+随机森林+线性回归多模型融合｜全周期均線MA5/20/30/50/60/120｜多特征时序挖掘｜多窗口支撑压力位")
st.caption("✅ 兼容環境：Windows/Mac/Linux/Streamlit Cloud｜中文顯示完美解決｜數據自動補全/兜底")
st.caption("⚠️ 本工具僅供學習，不構成任何投資建議，投資有風險，入市需謹慎！")