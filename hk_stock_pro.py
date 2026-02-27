# ================== 优化后的价格预测模块（替换原predict_price函数） ==================
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

def clean_outliers(df, column="Close"):
    """处理股价异常值（IQR方法）"""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_clean

def prepare_features(df):
    """构建多特征数据集（替代单一时间索引）"""
    df_feat = df.copy()
    
    # 基础价格特征
    df_feat["price_change"] = df_feat["Close"].pct_change()
    df_feat["high_low_diff"] = df_feat["High"] - df_feat["Low"]
    df_feat["open_close_diff"] = df_feat["Open"] - df_feat["Close"]
    
    # 技术指标特征（复用已计算的MA/RSI/MACD）
    df_feat["rsi_norm"] = df_feat["RSI"] / 100  # 归一化RSI
    df_feat["macd_diff"] = df_feat["MACD"] - df_feat["MACD_Signal"]
    df_feat["ma5_ma20_diff"] = df_feat["MA5"] - df_feat["MA20"]
    df_feat["close_ma5_diff"] = df_feat["Close"] - df_feat["MA5"]
    
    # 成交量特征
    df_feat["volume_change"] = df_feat["Volume"].pct_change()
    
    # 时间特征
    df_feat["day_of_week"] = df_feat["Date"].dt.weekday
    df_feat["month"] = df_feat["Date"].dt.month
    
    # 填充缺失值（避免模型报错）
    df_feat = df_feat.fillna(0)
    # 去除无穷值
    df_feat = df_feat.replace([np.inf, -np.inf], 0)
    
    # 特征列筛选（仅保留数值型特征）
    feature_cols = [
        "price_change", "high_low_diff", "open_close_diff",
        "rsi_norm", "macd_diff", "ma5_ma20_diff", "close_ma5_diff",
        "volume_change", "day_of_week", "month"
    ]
    # 确保特征列存在
    feature_cols = [col for col in feature_cols if col in df_feat.columns]
    
    return df_feat, feature_cols

def predict_price_optimized(df, days):
    """
    优化后的价格预测函数：
    1. 随机森林（非线性模型）替代线性回归
    2. 多特征融合（价格/技术指标/成交量/时间）
    3. 异常值处理
    4. 输出预测值+置信区间（95%）
    """
    try:
        # 步骤1：数据清洗（去除异常值）
        df_clean = clean_outliers(df)
        if len(df_clean) < 20:  # 数据量不足时降级为线性回归
            st.warning("⚠️ 有效数据量不足，降级为线性回归预测")
            return predict_price_linear(df, days)
        
        # 步骤2：构建多特征数据集
        df_feat, feature_cols = prepare_features(df_clean)
        if len(feature_cols) < 3:  # 特征不足时降级
            return predict_price_linear(df, days)
        
        # 步骤3：特征工程（归一化）
        X = df_feat[feature_cols].values
        y = df_feat["Close"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 步骤4：训练随机森林模型（调参优化）
        model = RandomForestRegressor(
            n_estimators=100,  # 决策树数量
            max_depth=10,      # 树深度（避免过拟合）
            min_samples_split=5,
            random_state=42    # 固定随机种子（可复现）
        )
        # 划分训练集（用80%数据训练）
        X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        # 步骤5：生成未来特征（基于最后一条数据的特征趋势）
        last_feat = df_feat.iloc[-1][feature_cols].values.reshape(1, -1)
        future_X = []
        for i in range(days):
            # 基于时间递增调整特征（模拟趋势）
            temp_feat = last_feat.copy()
            temp_feat[0, feature_cols.index("day_of_week")] = (df_feat["day_of_week"].iloc[-1] + i) % 5
            future_X.append(temp_feat[0])
        future_X_scaled = scaler.transform(future_X)
        
        # 步骤6：预测+计算95%置信区间（体现预测不确定性）
        # 用所有决策树的预测值计算置信区间
        tree_predictions = [tree.predict(future_X_scaled) for tree in model.estimators_]
        pred = np.mean(tree_predictions, axis=0)  # 均值作为最终预测
        pred_std = np.std(tree_predictions, axis=0)  # 标准差
        # 95%置信区间（1.96倍标准差）
        conf_interval = 1.96 * pred_std
        
        # 步骤7：计算整体趋势（基于预测值的斜率）
        slope, _, _, _, _ = stats.linregress(range(days), pred)
        
        return pred, slope, conf_interval
    
    except Exception as e:
        st.warning(f"⚠️ 优化预测失败，降级为基础线性回归：{str(e)}")
        pred, slope = predict_price_linear(df, days)
        conf_interval = np.zeros(days)  # 无置信区间
        return pred, slope, conf_interval

def predict_price_linear(df, days):
    """保留原线性回归作为兜底"""
    df["idx"] = np.arange(len(df))
    x = df["idx"].values.reshape(-1, 1)
    y = df["Close"].values
    model = LinearRegression()
    model.fit(x, y)
    future_idx = np.arange(len(df), len(df) + days).reshape(-1, 1)
    pred = model.predict(future_idx)
    slope = model.coef_[0]
    return pred, slope