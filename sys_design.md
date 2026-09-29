# 股票当日走势智能预测系统 — 完整技术方案

> **版本**：v1.0  
> **日期**：2025年1月  
> **定位**：基于多源数据融合 + 统计/机器学习模型 + LLM辅助研判的股票日内走势预测系统

---

## 目录

- [一、项目概述](#一项目概述)
- [二、系统架构](#二系统架构)
- [三、数据采集层](#三数据采集层)
- [四、特征工程层](#四特征工程层)
- [五、政治经济形势量化](#五政治经济形势量化)
- [六、模型预测层](#六模型预测层)
- [七、回测与评估框架](#七回测与评估框架)
- [八、风控与极端情况处理](#八风控与极端情况处理)
- [九、工程部署与监控](#九工程部署与监控)
- [十、用户体验设计](#十用户体验设计)
- [十一、合规与伦理](#十一合规与伦理)
- [十二、实施路线图](#十二实施路线图)
- [十三、技术栈总览](#十三技术栈总览)
- [十四、风险与局限性](#十四风险与局限性)

---

## 一、项目概述

### 1.1 项目目标

用户输入股票代码后，系统自动完成以下工作：

1. 获取该股票一年内的历史升跌数据
2. 采集当前政治形势、经济形势、行业动态
3. 识别有利因素与不利因素
4. 结合统计模型与机器学习模型预测当日走势
5. 输出预测结果、置信度、可解释性分析报告

### 1.2 核心设计理念

| 原则             | 说明                                                  |
| ---------------- | ----------------------------------------------------- |
| **多源融合**     | 不依赖单一数据源，技术面+基本面+政策面+情绪面多维融合 |
| **状态感知**     | 识别市场所处状态（牛/熊/震荡），动态切换策略          |
| **不确定性量化** | 不仅给出预测，还给出"模型有多不确定"                  |
| **可解释性**     | 用户能理解"为什么这样预测"                            |
| **合规优先**     | 内嵌免责声明，不输出具体买卖建议                      |

### 1.3 准确率预期

> ⚠️ **重要提示**：日频方向预测准确率在 **55%~62%** 已属优秀水平。  
> 若回测准确率 > 70%，大概率存在过拟合或数据泄露。  
> 系统核心价值在于 **风险识别 + 不确定性量化 + 可解释性**。

---

## 二、系统架构

### 2.1 总体架构图

┌─────────────────────────────────────────────────────────────────────┐ │ 用户输入股票代码 │ └────────────────────────────────┬────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────────┐ │ Layer 1: 数据采集层 │ │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │ │ │ K线 │ │资金流│ │北向 │ │龙虎榜│ │板块 │ │宏观 │ │新闻 │ │ │ │ 行情 │ │ 向 │ │资金 │ │ │ │联动 │ │指标 │ │政策 │ │ │ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │ └────────────────────────────────┬────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────────┐ │ Layer 2: 数据预处理 & 异常检测 │ │ • 缺失值填充 • 异常值检测 • 极端行情识别 • 数据源健康检查 │ └────────────────────────────────┬────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────────┐ │ Layer 3: 特征工程层 │ │ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────────┐ │ │ │技术指标│ │资金特征│ │情感得分│ │宏观指标│ │事件衰减+板块动量 │ │ │ └────────┘ └────────┘ └────────┘ └────────┘ └──────────────────┘ │ └────────────────────────────────┬────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────────┐ │ Layer 4: 市场状态识别 │ │ HMM / 规则引擎 → 牛市 / 熊市 / 震荡市 → 选择对应模型策略 │ └────────────────────────────────┬────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────────┐ │ Layer 5: 多模型集成预测 │ │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │ │ │ LightGBM │ │ LSTM │ │ XGBoost │ │ LLM 综合研判 │ │ │ └──────────┘ └──────────┘ └──────────┘ └────────────────────┘ │ │ ↓ 加权融合 + 不确定性量化 + 多时间尺度融合 ↓ │ └────────────────────────────────┬────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────────┐ │ Layer 6: 后处理 & 风控 │ │ • 极端行情降级 • 置信度调整 • 合规检查 • 免责声明注入 │ └────────────────────────────────┬────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────────┐ │ Layer 7: 输出层（用户自适应） │ │ • 预测结果 + 置信度 + 不确定性区间 │ │ • 有利/不利因素（可解释） │ │ • 风险提示 + 模型历史准确率参考 │ └────────────────────────────────┬────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────────┐ │ Layer 8: 监控 & 反馈闭环 │ │ • 预测准确率追踪 • 特征漂移监控 • 模型自动重训练 • 用户反馈收集 │ └─────────────────────────────────────────────────────────────────────┘

### 2.2 数据流向

原始数据 → 清洗/对齐 → 特征计算 → 状态识别 → 模型推理 → 后处理 → 输出 ↑ | └──────────── 反馈闭环（准确率监控、重训练）─────┘

---

## 三、数据采集层

### 3.1 数据源全景

| 数据类别 | 具体内容                         | 数据源                     | 更新频率   |
| -------- | -------------------------------- | -------------------------- | ---------- |
| 历史行情 | 日K线（OHLCV）、涨跌幅           | akshare / tushare          | 每日收盘后 |
| 资金流向 | 主力/超大单/大单/中单/小单净流入 | 东方财富 / akshare         | 实时/日频  |
| 北向资金 | 沪股通/深股通净买入              | 交易所 / akshare           | 日频       |
| 融资融券 | 融资余额、融券余额、变化率       | 交易所 / tushare           | 日频       |
| 龙虎榜   | 机构/游资席位买卖明细            | 东方财富 / akshare         | 日频       |
| 板块数据 | 所属行业/概念板块涨跌幅、排名    | 同花顺 / akshare           | 日频       |
| 宏观经济 | PMI、CPI、M2、SHIBOR、国债收益率 | 国家统计局 / Wind          | 月频/日频  |
| 政策动态 | 央行公告、国务院文件、行业政策   | 政府网站 / 新闻            | 事件驱动   |
| 新闻舆情 | 个股新闻、研报、社交媒体         | 新浪财经 / 东方财富 / 雪球 | 实时       |
| 股东行为 | 增减持、解禁、大宗交易           | 巨潮资讯 / akshare         | 事件驱动   |
| 国际市场 | 美股、港股、汇率、大宗商品       | yfinance / akshare         | 日频       |

### 3.2 数据采集代码

```python
import akshare as ak
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

class DataCollector:
    """统一数据采集器"""

    def __init__(self, tushare_token: str = None):
        if tushare_token:
            self.pro = ts.pro_api(tushare_token)

    def collect_all(self, stock_code: str) -> dict:
        """采集所有数据源"""
        data = {}

        # ① 历史K线（一年）
        data['kline'] = self.get_kline(stock_code)

        # ② 资金流向
        data['money_flow'] = self.get_money_flow(stock_code)

        # ③ 北向资金
        data['northbound'] = self.get_northbound()

        # ④ 融资融券
        data['margin'] = self.get_margin(stock_code)

        # ⑤ 龙虎榜
        data['dragon_tiger'] = self.get_dragon_tiger(stock_code)

        # ⑥ 板块数据
        data['sector'] = self.get_sector_data(stock_code)

        # ⑦ 宏观经济
        data['macro'] = self.get_macro_indicators()

        # ⑧ 新闻舆情
        data['news'] = self.get_news(stock_code)

        # ⑨ 股东行为
        data['shareholder'] = self.get_shareholder_actions(stock_code)

        # ⑩ 解禁日历
        data['unlock'] = self.get_unlock_schedule(stock_code)

        # ⑪ 国际市场
        data['international'] = self.get_international_market()

        return data

    def get_kline(self, stock_code: str) -> pd.DataFrame:
        """获取一年日K线数据（前复权）"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        return df

    def get_money_flow(self, stock_code: str) -> pd.DataFrame:
        """获取个股资金流向"""
        market = "sh" if stock_code.startswith('6') else "sz"
        df = ak.stock_individual_fund_flow(stock=stock_code, market=market)
        return df

    def get_northbound(self) -> pd.DataFrame:
        """获取北向资金净流入"""
        df = ak.stock_hsgt_north_net_flow_in_em()
        return df

    def get_margin(self, stock_code: str) -> pd.DataFrame:
        """获取融资融券数据"""
        df = ak.stock_margin_detail_szse(date=datetime.now().strftime('%Y%m%d'))
        return df[df['证券代码'] == stock_code]

    def get_dragon_tiger(self, stock_code: str) -> pd.DataFrame:
        """获取龙虎榜数据"""
        start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        end = datetime.now().strftime('%Y%m%d')
        df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
        return df[df['代码'] == stock_code]

    def get_sector_data(self, stock_code: str) -> dict:
        """获取所属板块及板块近期表现"""
        # 获取个股所属行业
        industry = ak.stock_board_industry_name_em()
        # 获取板块近20日涨跌幅排名
        sector_perf = ak.stock_board_industry_hist_em(
            symbol="行业名称", period="日k",
            start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        )
        return {'industry': industry, 'performance': sector_perf}

    def get_macro_indicators(self) -> dict:
        """获取宏观经济指标"""
        return {
            'pmi': self._get_pmi(),
            'cpi': self._get_cpi(),
            'm2_growth': self._get_m2(),
            'shibor': self._get_shibor(),
            'treasury_yield_10y': self._get_treasury_yield(),
            'usd_cny': self._get_exchange_rate(),
        }

    def get_news(self, stock_code: str) -> list:
        """获取近7天相关新闻"""
        # 可接入东方财富、新浪财经、雪球等
        news_list = []
        # ... 具体爬取逻辑
        return news_list

    def get_shareholder_actions(self, stock_code: str) -> pd.DataFrame:
        """获取股东增减持"""
        df = ak.stock_share_change_cninfo(symbol=stock_code)
        return df

    def get_unlock_schedule(self, stock_code: str) -> pd.DataFrame:
        """获取限售股解禁日历"""
        df = ak.stock_restricted_release_queue_em(symbol=stock_code)
        return df

    def get_international_market(self) -> dict:
        """获取国际市场关键指标"""
        return {
            'sp500_change': self._get_index_change('^GSPC'),
            'nasdaq_change': self._get_index_change('^IXIC'),
            'hsi_change': self._get_index_change('^HSI'),
            'vix': self._get_index_change('^VIX'),
            'crude_oil': self._get_index_change('CL=F'),
            'gold': self._get_index_change('GC=F'),
        }
```

## 四、特征工程层

### 4.1 技术指标特征

```python
import talib
import numpy as np

class TechnicalFeatureEngineer:
    """技术指标特征计算"""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['收盘'].values.astype(float)
        high = df['最高'].values.astype(float)
        low = df['最低'].values.astype(float)
        volume = df['成交量'].values.astype(float)

        features = pd.DataFrame(index=df.index)

        # ===== 趋势类 =====
        features['MA5'] = talib.SMA(close, timeperiod=5)
        features['MA10'] = talib.SMA(close, timeperiod=10)
        features['MA20'] = talib.SMA(close, timeperiod=20)
        features['MA60'] = talib.SMA(close, timeperiod=60)
        features['EMA12'] = talib.EMA(close, timeperiod=12)
        features['EMA26'] = talib.EMA(close, timeperiod=26)

        # MACD
        macd, signal, hist = talib.MACD(close)
        features['MACD'] = macd
        features['MACD_signal'] = signal
        features['MACD_hist'] = hist

        # ===== 动量类 =====
        features['RSI_6'] = talib.RSI(close, timeperiod=6)
        features['RSI_14'] = talib.RSI(close, timeperiod=14)
        features['RSI_24'] = talib.RSI(close, timeperiod=24)

        k, d = talib.STOCH(high, low, close)
        features['KDJ_K'] = k
        features['KDJ_D'] = d

        features['CCI'] = talib.CCI(high, low, close, timeperiod=20)
        features['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        features['MOM'] = talib.MOM(close, timeperiod=10)

        # ===== 波动类 =====
        upper, mid, lower = talib.BBANDS(close, timeperiod=20)
        features['BOLL_upper'] = upper
        features['BOLL_mid'] = mid
        features['BOLL_lower'] = lower
        features['BOLL_width'] = (upper - lower) / mid  # 布林带宽度
        features['BOLL_position'] = (close - lower) / (upper - lower)  # 价格在带中位置

        features['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        features['ATR_ratio'] = features['ATR_14'] / close  # 归一化ATR

        # ===== 量价类 =====
        features['OBV'] = talib.OBV(close, volume)
        features['volume_MA5'] = talib.SMA(volume, timeperiod=5)
        features['volume_MA20'] = talib.SMA(volume, timeperiod=20)
        features['volume_ratio'] = volume / features['volume_MA20']  # 量比

        features['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        features['AD'] = talib.AD(high, low, close, volume)

        # ===== 收益率特征 =====
        features['return_1d'] = pd.Series(close).pct_change(1)
        features['return_3d'] = pd.Series(close).pct_change(3)
        features['return_5d'] = pd.Series(close).pct_change(5)
        features['return_10d'] = pd.Series(close).pct_change(10)
        features['return_20d'] = pd.Series(close).pct_change(20)

        # ===== 波动率特征 =====
        features['volatility_5d'] = features['return_1d'].rolling(5).std()
        features['volatility_20d'] = features['return_1d'].rolling(20).std()

        # ===== 形态特征 =====
        features['high_low_ratio'] = (high - low) / close  # 振幅
        features['close_open_ratio'] = (close - df['开盘'].values) / df['开盘'].values
        features['upper_shadow'] = (high - np.maximum(close, df['开盘'].values)) / close
        features['lower_shadow'] = (np.minimum(close, df['开盘'].values) - low) / close

        return features
```

## 4.3 情感与舆情特征

```python
from openai import OpenAI
import json

class SentimentFeatures:
    """新闻舆情情感特征"""

    def __init__(self):
        self.client = OpenAI()

    def compute(self, news_list: list, stock_name: str) -> dict:
        """
        对新闻列表进行多维度情感打分
        """
        prompt = f"""
        请分析以下关于"{stock_name}"的近期新闻，从以下维度打分（-5到+5，0为中性）：

        1. policy_score: 政策面影响（利好政策为正，利空政策为负）
        2. industry_score: 行业景气度（行业向好为正，恶化
```
