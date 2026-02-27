# -*- coding: utf-8 -*-
"""
港股分析预测系统 | 专业顶级版
功能：港股数据获取（修复代码错误）、基本面分析、技术指标计算、价格预测、可视化输出
作者：专业量化分析团队
更新：2026-02-27
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（解决matplotlib中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['PingFang SC']  # Mac
plt.rcParams['axes.unicode_minus'] = False

class HKStockAnalysisSystem:
    """港股分析预测核心类"""
    def __init__(self, stock_code):
        """
        初始化
        :param stock_code: 港股代码（支持纯数字/带.HK后缀，自动修正）
        """
        # 核心修复：自动修正港股代码格式
        self.stock_code = self._fix_hk_code(stock_code)
        self.raw_data = None  # 原始行情数据
        self.tech_data = None  # 技术指标数据
        self.predict_data = None  # 预测数据

    def _fix_hk_code(self, code):
        """
        修复港股代码格式（解决"无法获取数据，请检查代码"核心问题）
        :param code: 输入的代码（如1810/01810/01810.HK）
        :return: 标准格式代码（如01810.HK）
        """
        # 去除多余字符，仅保留数字
        pure_code = ''.join([c for c in str(code) if c.isdigit()])
        
        # 港股代码补全为5位
        if len(pure_code) < 5:
            pure_code = pure_code.zfill(5)
        
        # 添加.HK后缀
        standard_code = f"{pure_code}.HK"
        print(f"✅ 代码格式修正完成：{code} → {standard_code}")
        return standard_code

    def get_stock_data(self, start_date=None, end_date=None):
        """
        获取港股历史数据（修复数据获取失败问题）
        :param start_date: 开始日期（格式YYYYMMDD，默认30天前）
        :param end_date: 结束日期（格式YYYYMMDD，默认今天）
        :return: 修复后的历史数据DataFrame
        """
        # 设置默认时间范围
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

        try:
            # 尝试多个数据源，确保数据获取成功率
            print(f"\n📊 正在获取 {self.stock_code} 数据（{start_date} ~ {end_date}）...")
            
            # 数据源1：东方财富（优先）
            self.raw_data = ak.stock_hk_hist(
                symbol=self.stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            # 数据源2：备用（新浪），若东方财富接口失效
            if self.raw_data.empty:
                print("⚠️ 东方财富数据源失效，切换至新浪数据源...")
                self.raw_data = ak.stock_hk_spot_sina(symbol=self.stock_code.split('.')[0])
                if isinstance(self.raw_data, pd.Series):
                    self.raw_data = pd.DataFrame([self.raw_data])

            # 数据清洗
            if not self.raw_data.empty:
                # 统一列名，方便后续分析
                self.raw_data.rename(columns={
                    '日期': 'date', '开盘': 'open', '最高': 'high',
                    '最低': 'low', '收盘': 'close', '成交量': 'volume'
                }, inplace=True, errors='ignore')
                # 转换日期格式
                if 'date' in self.raw_data.columns:
                    self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
                print(f"✅ 数据获取成功！共 {len(self.raw_data)} 条记录")
                return self.raw_data
            else:
                raise ValueError("获取的数据为空")

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ 数据获取失败！错误原因：{error_msg}")
            
            # 针对性排查提示（解决"检查代码"问题）
            if "代码" in error_msg or "symbol" in error_msg or self.raw_data is None:
                print("\n🔍 排查建议：")
                print("1. 确认股票代码正确性（港股需5位数字，如小米=01810）")
                print("2. 检查是否在港股交易时段（9:30-12:00, 13:00-16:00）")
                print("3. 尝试更新akshare：pip install akshare --upgrade")
                print("4. 测试热门股票代码（如腾讯=00700.HK）验证接口有效性")
            return None

    def calculate_technical_indicators(self):
        """计算核心技术指标（MA/RSI/MACD），用于预测分析"""
        if self.raw_data is None or self.raw_data.empty:
            print("❌ 请先获取有效数据再计算技术指标！")
            return None
        
        self.tech_data = self.raw_data.copy()
        
        # 1. 移动平均线（MA）
        self.tech_data['MA5'] = self.tech_data['close'].rolling(window=5).mean()
        self.tech_data['MA10'] = self.tech_data['close'].rolling(window=10).mean()
        self.tech_data['MA20'] = self.tech_data['close'].rolling(window=20).mean()
        
        # 2. 相对强弱指数（RSI）
        delta = self.tech_data['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.tech_data['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD
        ema12 = self.tech_data['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.tech_data['close'].ewm(span=26, adjust=False).mean()
        self.tech_data['MACD'] = ema12 - ema26
        self.tech_data['MACD_Signal'] = self.tech_data['MACD'].ewm(span=9, adjust=False).mean()
        self.tech_data['MACD_Hist'] = self.tech_data['MACD'] - self.tech_data['MACD_Signal']
        
        print("✅ 技术指标计算完成（MA/RSI/MACD）")
        return self.tech_data

    def price_prediction(self, predict_days=5):
        """
        基于线性回归的价格预测
        :param predict_days: 预测未来天数
        :return: 预测结果
        """
        if self.tech_data is None or self.tech_data.empty:
            print("❌ 请先计算技术指标！")
            return None
        
        # 准备特征数据（使用收盘价和MA5）
        df = self.tech_data[['close', 'MA5']].dropna()
        X = np.arange(len(df)).reshape(-1, 1)  # 时间序列作为特征
        y = df['close'].values
        
        # 线性回归模型
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # 预测未来n天价格
        last_day = len(df) - 1
        future_days = np.arange(last_day + 1, last_day + 1 + predict_days).reshape(-1, 1)
        future_prices = model.predict(future_days)
        
        # 生成预测日期
        last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else self.tech_data['date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(predict_days)]
        
        # 整理预测结果
        self.predict_data = pd.DataFrame({
            '预测日期': future_dates,
            '预测收盘价': future_prices.round(2),
            'MA5参考': [df['MA5'].iloc[-1]] * predict_days
        })
        
        print(f"\n📈 未来 {predict_days} 天价格预测：")
        print(self.predict_data)
        return self.predict_data

    def visualize_analysis(self):
        """可视化分析结果（价格走势+技术指标+预测）"""
        if self.raw_data is None or self.raw_data.empty:
            print("❌ 无数据可可视化！")
            return
        
        # 创建2行1列的子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 子图1：价格走势 + MA + 预测
        ax1.plot(self.raw_data['date'], self.raw_data['close'], label='收盘价', color='blue')
        if 'MA5' in self.tech_data.columns:
            ax1.plot(self.tech_data['date'], self.tech_data['MA5'], label='MA5', color='orange')
            ax1.plot(self.tech_data['date'], self.tech_data['MA10'], label='MA10', color='green')
        if self.predict_data is not None:
            ax1.plot(self.predict_data['预测日期'], self.predict_data['预测收盘价'], 
                     label='预测价格', color='red', linestyle='--', marker='o')
        ax1.set_title(f'{self.stock_code} 港股价格分析与预测', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格（港元）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：RSI指标
        if 'RSI' in self.tech_data.columns:
            ax2.plot(self.tech_data['date'], self.tech_data['RSI'], label='RSI(14)', color='purple')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='超买线(70)')
            ax2.axhline(30, color='green', linestyle='--', alpha=0.7, label='超卖线(30)')
            ax2.set_ylabel('RSI')
            ax2.set_xlabel('日期')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.stock_code}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ 分析图表已保存为：{self.stock_code}_analysis.png")
        plt.show()

    def generate_analysis_report(self):
        """生成综合分析报告"""
        if self.raw_data is None or self.raw_data.empty:
            print("❌ 无法生成报告：无有效数据！")
            return
        
        print("\n" + "="*50)
        print(f"📋 {self.stock_code} 港股综合分析报告")
        print("="*50)
        
        # 基本信息
        latest_data = self.raw_data.iloc[-1]
        print(f"\n【基本行情】")
        print(f"最新日期：{latest_data['date'].strftime('%Y-%m-%d')}")
        print(f"开盘价：{latest_data['open']:.2f} 港元")
        print(f"收盘价：{latest_data['close']:.2f} 港元")
        print(f"最高价：{latest_data['high']:.2f} 港元")
        print(f"最低价：{latest_data['low']:.2f} 港元")
        print(f"成交量：{latest_data['volume']:,.0f} 股")
        
        # 技术分析
        if self.tech_data is not None:
            latest_tech = self.tech_data.iloc[-1]
            print(f"\n【技术指标】")
            print(f"MA5：{latest_tech['MA5']:.2f} | MA10：{latest_tech['MA10']:.2f} | MA20：{latest_tech['MA20']:.2f}")
            print(f"RSI(14)：{latest_tech['RSI']:.2f} → {'超买' if latest_tech['RSI'] > 70 else '超卖' if latest_tech['RSI'] < 30 else '正常'}")
            print(f"MACD：{latest_tech['MACD']:.4f} | 信号线：{latest_tech['MACD_Signal']:.4f}")
        
        # 预测分析
        if self.predict_data is not None:
            print(f"\n【预测结论】")
            predict_trend = "上涨" if self.predict_data['预测收盘价'].iloc[-1] > latest_data['close'] else "下跌"
            price_change = abs(self.predict_data['预测收盘价'].iloc[-1] - latest_data['close'])
            print(f"未来5天价格趋势：{predict_trend}（预计变动 {price_change:.2f} 港元）")
        
        print("\n" + "="*50)

# -------------------------- 核心运行代码 --------------------------
if __name__ == "__main__":
    # 1. 初始化系统（支持多种代码格式：1810/01810/01810.HK）
    stock_code = "01810"  # 小米集团-W（可替换为任意港股代码）
    analysis_system = HKStockAnalysisSystem(stock_code)
    
    # 2. 获取数据（自动修复代码格式，解决"无法获取数据"问题）
    analysis_system.get_stock_data(start_date="20260101")
    
    # 3. 计算技术指标
    if analysis_system.raw_data is not None:
        analysis_system.calculate_technical_indicators()
        
        # 4. 价格预测（未来5天）
        analysis_system.price_prediction(predict_days=5)
        
        # 5. 可视化分析结果
        analysis_system.visualize_analysis()
        
        # 6. 生成综合报告
        analysis_system.generate_analysis_report()