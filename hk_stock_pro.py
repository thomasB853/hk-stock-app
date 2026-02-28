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

# ================== Global Configuration ==================
warnings.filterwarnings('ignore')
st.set_page_config(page_title="HK Stock Analysis System", layout="wide")
# Set font for English display (compatible with Streamlit Cloud)
plt.rcParams["font.family"] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams["axes.unicode_minus"] = False

# ================== Dependency Check & Force Upgrade ==================
try:
    import yfinance as yf
    if hasattr(yf, '__version__') and yf.__version__ < "0.2.31":
        st.warning("⚠️ yfinance version is outdated, upgrading to latest version...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yfinance>=0.2.31"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        importlib.reload(yf)
except ImportError:
    st.error("❌ Missing yfinance library, installing automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance>=0.2.31"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import yfinance as yf

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.error("❌ Missing scikit-learn library, installing automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn>=1.3.0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from sklearn.linear_model import LinearRegression

# ================== Page UI ==================
st.title("📈 HK Stock Analysis & Prediction System | Enhanced Version")
st.markdown("### Supported: Tencent, Meituan, HSBC and other major HK stocks (Prediction Model: Random Forest + Multi-Feature)")

# Hot HK Stocks (stable data from Yahoo Finance)
hot_stocks = {
    "Tencent Holdings (0700)": "0700",
    "Meituan-W (3690)": "3690",
    "HSBC Holdings (0005)": "0005",
    "Xiaomi Group-W (1810)": "1810",
    "Alibaba-SW (9988)": "9988",
    "ICBC (1398)": "1398"
}
option = st.selectbox("Select Popular HK Stocks (Stable Data)", list(hot_stocks.keys()))
default_code = hot_stocks[option]
user_code = st.text_input("Manual Input HK Stock Code (4-5 digits, e.g., 0700)", default_code).strip()
predict_days = st.slider("Prediction Days (1-15 days)", 1, 15, 5)

# ================== Core Utility Functions ==================
def is_trading_day(date):
    """Check if date is HK stock trading day (exclude Saturday/Sunday)"""
    return date.weekday() not in [5, 6]

def get_trading_dates(start_date, days):
    """Get future HK trading dates"""
    trading_dates = []
    current_date = start_date
    while len(trading_dates) < days:
        if is_trading_day(current_date):
            trading_dates.append(current_date)
        current_date += timedelta(days=1)
    return trading_dates

def clean_column_names(df):
    """Clean column names for Yahoo Finance data"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    
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
    
    final_cols = {}
    for col in df.columns:
        for key in column_mapping.keys():
            if key in col:
                final_cols[col] = column_mapping[key]
                break
    
    df.rename(columns=final_cols, inplace=True)
    return df

# ================== Stable Data Fetching Functions ==================
@st.cache_data(ttl=3600)
def get_hk_stock_data(symbol):
    """Fetch HK stock data with multi-level compatibility"""
    yf_symbol = f"{symbol}.HK"
    st.info(f"🔍 Fetching data: {yf_symbol}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=4*365)  # Extend to 4 years for quarterly data
    
    try:
        df = yf.download(
            yf_symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            timeout=60,
            threads=False,
            auto_adjust=False,
            back_adjust=False,
            repair=True
        )
        
        if df.empty or len(df) < 5:
            st.warning("⚠️ Default method failed, trying backup API...")
            url = f"https://query1.finance.yahoo.com/v7/finance/chart/{yf_symbol}?range=4y&interval=1d&indicators=quote&includeTimestamps=true"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=60)
            data = resp.json()
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
                df = df.dropna(subset=['Close'])
            else:
                st.error(f"❌ Failed to fetch data for {yf_symbol}")
                return None
        
        df.reset_index(inplace=True)
        df = clean_column_names(df)
        
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"⚠️ Missing columns: {missing_cols}, attempting to fill...")
            if "Date" not in df.columns:
                st.error("❌ Core field Date missing, cannot continue")
                return None
            if "Close" in df.columns:
                for col in ["Open", "High", "Low"]:
                    if col not in df.columns:
                        df[col] = df["Close"]
            else:
                st.error("❌ Core field Close missing, cannot continue")
                return None
            if "Volume" not in df.columns:
                df["Volume"] = 0
        
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        
        if len(df) < 10:
            st.warning(f"⚠️ Only {len(df)} valid data points (low reference value)")
        
        st.success(f"✅ Successfully fetched {len(df)} records for {yf_symbol}")
        return df
    
    except Exception as e:
        st.error(f"❌ Data fetch error: {str(e)[:100]}")
        st.info("💡 Solutions:")
        st.info("1. Refresh page (network/data source temporary fluctuation)")
        st.info("2. Verify HK stock code format (4-5 digits, e.g., 0700 not 700)")
        st.info("3. Try popular stocks (e.g., Tencent 0700, Xiaomi 1810)")
        return None

@st.cache_data(ttl=3600)
def get_hsi_data():
    """Fetch Hang Seng Index (HSI) data for prediction"""
    try:
        hsi_symbol = "^HSI"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        
        df = yf.download(
            hsi_symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            timeout=60
        )
        
        if df.empty:
            st.warning("⚠️ Failed to fetch HSI data")
            return None
        
        df.reset_index(inplace=True)
        df = clean_column_names(df)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        
        return df
    except Exception as e:
        st.warning(f"⚠️ HSI data fetch error: {str(e)}")
        return None

# ================== Technical Indicators Calculation ==================
def calculate_indicators(df):
    """Calculate technical indicators including MA30/50/100"""
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    try:
        # Moving Averages (extend to MA30/50/100)
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["MA30"] = df["Close"].rolling(window=30, min_periods=1).mean()
        df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
        df["MA100"] = df["Close"].rolling(window=100, min_periods=1).mean()
        
        # MACD
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
        
        # RSI
        delta = df["Close"].pct_change()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)
        df["RSI"] = 100 - (100 / (1 + rs))
        
        return df
    except Exception as e:
        st.warning(f"⚠️ Partial failure in technical indicator calculation: {str(e)}")
        return df

# ================== Quarterly Performance Data ==================
def generate_quarterly_performance(df):
    """Generate quarterly performance data (last 3 years + current year)"""
    if df is None or len(df) == 0:
        return None
    
    df_q = df.copy()
    df_q['Year'] = df_q['Date'].dt.year
    df_q['Quarter'] = df_q['Date'].dt.quarter
    
    # Calculate quarterly metrics (based on price movement and volume)
    quarterly_metrics = df_q.groupby(['Year', 'Quarter']).agg({
        'Close': ['first', 'last', 'min', 'max'],
        'Volume': 'sum',
        'Date': 'count'
    }).reset_index()
    
    # Simplify column names
    quarterly_metrics.columns = ['Year', 'Quarter', 'Open_Price', 'Close_Price', 'Low_Price', 'High_Price', 'Total_Volume', 'Trading_Days']
    
    # Calculate quarterly return
    quarterly_metrics['Quarterly_Return'] = ((quarterly_metrics['Close_Price'] - quarterly_metrics['Open_Price']) / quarterly_metrics['Open_Price']) * 100
    quarterly_metrics['Average_Price'] = (quarterly_metrics['Open_Price'] + quarterly_metrics['Close_Price'] + quarterly_metrics['Low_Price'] + quarterly_metrics['High_Price']) / 4
    
    # Filter to last 3 years + current year
    current_year = datetime.now().year
    target_years = [current_year - 3, current_year - 2, current_year - 1, current_year]
    quarterly_metrics = quarterly_metrics[quarterly_metrics['Year'].isin(target_years)]
    
    # Add formatted quarter name
    quarterly_metrics['Quarter_Name'] = quarterly_metrics['Year'].astype(str) + ' Q' + quarterly_metrics['Quarter'].astype(str)
    
    return quarterly_metrics

# ================== Support/Resistance Calculation ==================
def calculate_support_resistance(df, window=20):
    """Calculate support and resistance levels"""
    try:
        support = df["Low"].rolling(window=window, min_periods=1).min().iloc[-1]
        resistance = df["High"].rolling(window=window, min_periods=1).max().iloc[-1]
        return round(support, 2), round(resistance, 2)
    except:
        return round(df["Low"].iloc[-1], 2), round(df["High"].iloc[-1], 2)

# ================== Price Prediction Modules ==================
def clean_outliers(df, column="Close"):
    """Handle outliers using IQR method"""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_clean

def prepare_features(df):
    """Prepare multi-feature dataset"""
    df_feat = df.copy()
    
    df_feat["price_change"] = df_feat["Close"].pct_change()
    df_feat["high_low_diff"] = df_feat["High"] - df_feat["Low"]
    df_feat["open_close_diff"] = df_feat["Open"] - df_feat["Close"]
    
    if "RSI" in df_feat.columns:
        df_feat["rsi_norm"] = df_feat["RSI"] / 100
    if "MACD" in df_feat.columns and "MACD_Signal" in df_feat.columns:
        df_feat["macd_diff"] = df_feat["MACD"] - df_feat["MACD_Signal"]
    if "MA5" in df_feat.columns and "MA20" in df_feat.columns:
        df_feat["ma5_ma20_diff"] = df_feat["MA5"] - df_feat["MA20"]
        df_feat["close_ma5_diff"] = df_feat["Close"] - df_feat["MA5"]
    
    df_feat["volume_change"] = df_feat["Volume"].pct_change()
    df_feat["day_of_week"] = df_feat["Date"].dt.weekday
    df_feat["month"] = df_feat["Date"].dt.month
    
    df_feat = df_feat.fillna(0)
    df_feat = df_feat.replace([np.inf, -np.inf], 0)
    
    feature_cols = [
        "price_change", "high_low_diff", "open_close_diff",
        "volume_change", "day_of_week", "month"
    ]
    
    # Add optional features if available
    optional_features = ["rsi_norm", "macd_diff", "ma5_ma20_diff", "close_ma5_diff"]
    for feat in optional_features:
        if feat in df_feat.columns:
            feature_cols.append(feat)
    
    feature_cols = [col for col in feature_cols if col in df_feat.columns]
    return df_feat, feature_cols

def predict_price_optimized(df, days):
    """Optimized price prediction with Random Forest"""
    try:
        df_clean = clean_outliers(df)
        if len(df_clean) < 20:
            st.warning("⚠️ Insufficient valid data, downgrading to linear regression")
            pred, slope = predict_price_linear(df, days)
            conf_interval = np.zeros(days)
            return pred, slope, conf_interval
        
        df_feat, feature_cols = prepare_features(df_clean)
        if len(feature_cols) < 3:
            pred, slope = predict_price_linear(df, days)
            conf_interval = np.zeros(days)
            return pred, slope, conf_interval
        
        X = df_feat[feature_cols].values
        y = df_feat["Close"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        last_feat = df_feat.iloc[-1][feature_cols].values.reshape(1, -1)
        future_X = []
        for i in range(days):
            temp_feat = last_feat.copy()
            if "day_of_week" in feature_cols:
                temp_feat[0, feature_cols.index("day_of_week")] = (df_feat["day_of_week"].iloc[-1] + i) % 5
            future_X.append(temp_feat[0])
        future_X_scaled = scaler.transform(future_X)
        
        tree_predictions = [tree.predict(future_X_scaled) for tree in model.estimators_]
        pred = np.mean(tree_predictions, axis=0)
        pred_std = np.std(tree_predictions, axis=0)
        conf_interval = 1.96 * pred_std
        
        slope, _, _, _, _ = stats.linregress(range(days), pred)
        
        return pred, slope, conf_interval
    
    except Exception as e:
        st.warning(f"⚠️ Optimized prediction failed, downgrading to linear regression: {str(e)}")
        pred, slope = predict_price_linear(df, days)
        conf_interval = np.zeros(days)
        return pred, slope, conf_interval

def predict_price_linear(df, days):
    """Fallback linear regression prediction"""
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
    """Simple backtesting to verify model accuracy"""
    try:
        df_clean = clean_outliers(df)
        if len(df_clean) < 50:
            return "Insufficient data (<50 records), cannot backtest"
        split_idx = int(len(df_clean) * 0.9)
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        pred_test, _, _ = predict_price_optimized(train_df, len(test_df))
        mae = np.mean(np.abs(pred_test - test_df["Close"].values))
        return f"Backtest MAE: {mae:.2f} HK$ (lower = more accurate)"
    except Exception as e:
        return f"Backtest failed: {str(e)[:50]}"

# ================== Main Execution Logic ==================
if st.button("🚀 Start Analysis (Enhanced Version)", type="primary"):
    if not user_code.isdigit() or len(user_code) not in [4,5]:
        st.error("❌ Invalid HK stock code format! Must be 4-5 digits (e.g., Tencent=0700, Xiaomi=1810)")
    else:
        # Fetch stock data
        df = get_hk_stock_data(user_code)
        if df is None:
            st.stop()
        
        # Calculate indicators
        df = calculate_indicators(df)
        if df is None:
            st.stop()
        
        # Generate quarterly performance data
        quarterly_data = generate_quarterly_performance(df)
        
        # Fetch HSI data and predict
        hsi_df = get_hsi_data()
        hsi_pred = None
        hsi_slope = None
        if hsi_df is not None:
            hsi_df = calculate_indicators(hsi_df)
            hsi_pred, hsi_slope, _ = predict_price_optimized(hsi_df, predict_days)
        
        # Calculate support/resistance
        sup, res = calculate_support_resistance(df)
        # Predict stock price
        pred, slope, conf_interval = predict_price_optimized(df, predict_days)
        last_close = df["Close"].iloc[-1]
        
        # ========== Display Data ==========
        # Latest 10 records
        st.subheader("📊 Latest Trading Data (Top 10 Records)")
        show_cols = ["Date","Open","High","Low","Close","Volume","MA5","MA20","MA30","MA50","MA100"]
        show_cols = [col for col in show_cols if col in df.columns]
        show_df = df[show_cols].tail(10)
        float_cols = [col for col in show_cols if col not in ["Date", "Volume"]]
        round_dict = {col:2 for col in float_cols}
        if "Volume" in show_cols:
            round_dict["Volume"] = 0
        show_df = show_df.round(round_dict)
        st.dataframe(show_df, use_container_width=True)
        
        # Price & MA chart
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Price & Moving Averages Trend")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df["Date"], df["Close"], label="Close Price", color="#1f77b4", linewidth=1.5)
            ax.plot(df["Date"], df["MA5"], label="MA5 (5-day)", color="#ff7f0e", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA20"], label="MA20 (20-day)", color="#2ca02c", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA30"], label="MA30 (30-day)", color="#d62728", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA50"], label="MA50 (50-day)", color="#9467bd", linewidth=1, alpha=0.8)
            ax.plot(df["Date"], df["MA100"], label="MA100 (100-day)", color="#8c564b", linewidth=1, alpha=0.8)
            
            ax.set_title(f"{option} ({user_code}.HK) Price Trend", fontsize=12)
            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel("Price (HK$)", fontsize=10)
            ax.legend(fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("🛡️ Support / Resistance Levels")
            st.info(f"📉 Support Level: {sup} HK$")
            st.info(f"📈 Resistance Level: {res} HK$")
            st.info(f"📊 Last Close Price: {last_close:.2f} HK$")
            
            # Trend analysis
            if slope > 0:
                trend = "UPWARD 📈"
                color = "green"
            elif slope < 0:
                trend = "DOWNWARD 📉"
                color = "red"
            else:
                trend = "SIDEWAYS ➡️"
                color = "gray"
            
            st.markdown(f"### 📊 Prediction Trend: <span style='color:{color}'>{trend}</span>", unsafe_allow_html=True)
            st.info(f"Prediction Slope: {slope:.4f} HK$/day")
            
            # Backtest result
            backtest_result = backtest_model(df)
            st.info(f"🔍 Model Backtest: {backtest_result}")
        
        # Quarterly performance comparison
        st.subheader("📊 Quarterly Performance (Last 3 Years + Current Year)")
        if quarterly_data is not None and len(quarterly_data) > 0:
            # Display quarterly data table
            show_q_cols = ["Quarter_Name", "Open_Price", "Close_Price", "Quarterly_Return", "Average_Price", "Total_Volume"]
            q_show_df = quarterly_data[show_q_cols].round({
                "Open_Price":2, "Close_Price":2, "Quarterly_Return":2, "Average_Price":2, "Total_Volume":0
            })
            st.dataframe(q_show_df, use_container_width=True)
            
            # Quarterly return comparison chart
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("📈 Quarterly Return Comparison")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                
                # Prepare data for plotting
                q_years = quarterly_data['Year'].unique()
                q_returns = []
                q_labels = []
                
                for year in q_years:
                    year_data = quarterly_data[quarterly_data['Year'] == year]
                    for q in sorted(year_data['Quarter'].unique()):
                        q_data = year_data[year_data['Quarter'] == q]
                        if not q_data.empty:
                            q_labels.append(f"{year} Q{q}")
                            q_returns.append(q_data['Quarterly_Return'].iloc[0])
                
                # Plot bar chart
                colors = ['green' if x > 0 else 'red' for x in q_returns]
                ax2.bar(q_labels, q_returns, color=colors, alpha=0.7)
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax2.set_title("Quarterly Return (%) Comparison", fontsize=12)
                ax2.set_xlabel("Quarter", fontsize=10)
                ax2.set_ylabel("Return (%)", fontsize=10)
                ax2.tick_params(axis='x', rotation=45, labelsize=8)
                ax2.tick_params(axis='y', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig2)
            
            with col4:
                st.subheader("📊 Average Price by Quarter")
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                
                # Prepare data
                avg_prices = []
                for year in q_years:
                    year_data = quarterly_data[quarterly_data['Year'] == year]
                    for q in sorted(year_data['Quarter'].unique()):
                        q_data = year_data[year_data['Quarter'] == q]
                        if not q_data.empty:
                            avg_prices.append(q_data['Average_Price'].iloc[0])
                
                # Plot line chart
                ax3.plot(q_labels, avg_prices, marker='o', linewidth=2, markersize=4, color='#1f77b4')
                ax3.set_title("Average Quarterly Price (HK$)", fontsize=12)
                ax3.set_xlabel("Quarter", fontsize=10)
                ax3.set_ylabel("Average Price (HK$)", fontsize=10)
                ax3.tick_params(axis='x', rotation=45, labelsize=8)
                ax3.tick_params(axis='y', labelsize=8)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig3)
        else:
            st.warning("⚠️ Insufficient data to generate quarterly performance")
        
        # Price prediction chart
        st.subheader("🔮 Price Prediction (Next {} Trading Days)".format(predict_days))
        # Get future trading dates
        last_date = df["Date"].iloc[-1]
        future_dates = get_trading_dates(last_date + timedelta(days=1), predict_days)
        
        # Plot prediction
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        # Historical data
        ax4.plot(df["Date"].tail(60), df["Close"].tail(60), label="Historical Price", color="#1f77b4", linewidth=1.5)
        # Prediction with confidence interval
        ax4.plot(future_dates, pred, label="Predicted Price", color="#ff7f0e", linewidth=2, marker='o', markersize=4)
        ax4.fill_between(future_dates, pred - conf_interval, pred + conf_interval, alpha=0.2, color="#ff7f0e", label="95% Confidence Interval")
        
        ax4.set_title(f"{option} ({user_code}.HK) Price Prediction", fontsize=12)
        ax4.set_xlabel("Date", fontsize=10)
        ax4.set_ylabel("Price (HK$)", fontsize=10)
        ax4.legend(fontsize=10)
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
        ax4.tick_params(axis='y', labelsize=8)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        
        # HSI Prediction
        st.subheader("🌐 Hang Seng Index (HSI) Future Trend Prediction")
        if hsi_df is not None and hsi_pred is not None:
            col5, col6 = st.columns(2)
            
            with col5:
                # HSI historical + prediction chart
                fig5, ax5 = plt.subplots(figsize=(10, 5))
                ax5.plot(hsi_df["Date"].tail(60), hsi_df["Close"].tail(60), label="HSI Historical", color="#2ca02c", linewidth=1.5)
                
                # HSI future dates
                hsi_last_date = hsi_df["Date"].iloc[-1]
                hsi_future_dates = get_trading_dates(hsi_last_date + timedelta(days=1), predict_days)
                ax5.plot(hsi_future_dates, hsi_pred, label="HSI Predicted", color="#d62728", linewidth=2, marker='o', markersize=4)
                
                ax5.set_title("HSI Price Trend & Prediction", fontsize=12)
                ax5.set_xlabel("Date", fontsize=10)
                ax5.set_ylabel("Index Value", fontsize=10)
                ax5.legend(fontsize=9)
                ax5.tick_params(axis='x', rotation=45, labelsize=8)
                ax5.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig5)
            
            with col6:
                # HSI trend analysis
                st.info(f"📊 HSI Last Close: {hsi_df['Close'].iloc[-1]:.2f}")
                
                if hsi_slope > 0:
                    hsi_trend = "UPWARD 📈"
                    hsi_color = "green"
                elif hsi_slope < 0:
                    hsi_trend = "DOWNWARD 📉"
                    hsi_color = "red"
                else:
                    hsi_trend = "SIDEWAYS ➡️"
                    hsi_color = "gray"
                
                st.markdown(f"### 📊 HSI Prediction Trend: <span style='color:{hsi_color}'>{hsi_trend}</span>", unsafe_allow_html=True)
                st.info(f"HSI Prediction Slope: {hsi_slope:.2f} points/day")
                
                # Display HSI prediction table
                hsi_pred_df = pd.DataFrame({
                    "Prediction Date": [d.strftime("%Y-%m-%d") for d in hsi_future_dates],
                    "Predicted HSI Value": [round(p, 2) for p in hsi_pred]
                })
                st.dataframe(hsi_pred_df, use_container_width=True)
        else:
            st.warning("⚠️ Failed to generate HSI prediction (data unavailable)")
        
        # Prediction summary
        st.subheader("📋 Prediction Summary")
        pred_summary = pd.DataFrame({
            "Prediction Date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "Predicted Price (HK$)": [round(p, 2) for p in pred],
            "Lower Bound (95%)": [round(p - ci, 2) for p, ci in zip(pred, conf_interval)],
            "Upper Bound (95%)": [round(p + ci, 2) for p, ci in zip(pred, conf_interval)]
        })
        st.dataframe(pred_summary, use_container_width=True)
        
        # Risk warning
        st.markdown("---")
        st.warning("⚠️ RISK WARNING: This prediction is for reference only based on historical data and machine learning models. Stock market investment involves risks, please make decisions cautiously.")