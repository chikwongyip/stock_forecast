"""akshare 适配层: 所有外部数据请求的唯一出口.

职责: 重试、超时控制、列名归一化(中文列名 -> 英文 snake_case)、明确报错.
akshare 接口随版本变动, 全部收敛到本文件, 修复时只改这里.
"""

import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

RETRIES = 3
RETRY_BACKOFF = 2.0  # 秒, 指数退避基数


class AkshareError(RuntimeError):
    """akshare 调用最终失败."""


def _retry(func, *args, **kwargs) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            df = func(*args, **kwargs)
            if df is None:
                raise AkshareError(f"{func.__name__} 返回 None")
            return df
        except AkshareError:
            raise
        except Exception as e:  # noqa: BLE001 - akshare 异常类型杂, 统一捕获
            last_err = e
            if attempt < RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2**attempt))
    raise AkshareError(f"akshare 调用失败 (重试{RETRIES}次): {last_err}") from last_err


def _col_map(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """按映射重命名列; 允许部分列缺失(akshare 版本差异), 缺失列记 None."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for cn, en in mapping.items():
        if cn in df.columns and en not in df.columns:
            df = df.rename(columns={cn: en})
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def market_of(symbol: str) -> str:
    """个股所属交易所: 6 开头沪市, 其余按深市处理(0/3 开头)."""
    return "sh" if symbol.startswith("6") else "sz"


def get_kline(symbol: str, days: int = 365) -> pd.DataFrame:
    """日K线(前复权), 英文列: date/open/close/high/low/volume/amount/pct_change/turnover."""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    df = _retry(
        ak.stock_zh_a_hist,
        symbol=symbol, period="daily",
        start_date=start, end_date=end, adjust="qfq",
    )
    df = _col_map(df, {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount",
        "涨跌幅": "pct_change", "换手率": "turnover",
    })
    keep = ["date", "open", "close", "high", "low", "volume", "amount",
            "pct_change", "turnover"]
    return df[[c for c in keep if c in df.columns]]


def get_index_kline(symbol: str = "000001", days: int = 365) -> pd.DataFrame:
    """指数日K线, 默认上证指数. 列同 get_kline(无 turnover)."""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    df = _retry(
        ak.index_zh_a_hist,
        symbol=symbol, period="daily",
        start_date=start, end_date=end,
    )
    df = _col_map(df, {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change",
    })
    keep = ["date", "open", "close", "high", "low", "volume", "amount", "pct_change"]
    return df[[c for c in keep if c in df.columns]]


def get_fund_flow(symbol: str) -> pd.DataFrame:
    """个股资金流向(收盘后数据). 关键列: main_net / main_net_pct."""
    df = _retry(ak.stock_individual_fund_flow, stock=symbol, market=market_of(symbol))
    df = _col_map(df, {
        "日期": "date", "收盘价": "close", "涨跌幅": "pct_change",
        "主力净流入-净额": "main_net", "主力净流入-净占比": "main_net_pct",
        "超大单净流入-净额": "super_net", "超大单净流入-净占比": "super_net_pct",
        "大单净流入-净额": "big_net", "大单净流入-净占比": "big_net_pct",
        "中单净流入-净额": "mid_net", "中单净流入-净占比": "mid_net_pct",
        "小单净流入-净额": "small_net", "小单净流入-净占比": "small_net_pct",
    })
    drop = [c for c in ("Unnamed: 0",) if c in df.columns]
    df = df.drop(columns=drop)
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_news_raw(symbol: str) -> pd.DataFrame:
    """个股新闻(东方财富). 列: keyword/title/content/publish_time/source_url.

    注意: 接口按关键词匹配, 噪音较多, M4 的 llm/news.py 负责过滤.
    """
    df = _retry(ak.stock_news_em, symbol=symbol)
    if "发布时间" not in df.columns:
        # 列名变动的兜底: 尝试常见等价名
        ren = {"时间": "发布时间", "publish_time": "发布时间"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    return df


# ---------------------------------------------------------------------------
# Phase 2 扩展数据源 (develop_plan.md 批次 A/B)
#
# 约定: 全部使用 _map_cols 做多候选列名归一化(接口字段随 akshare 版本漂移,
# 首个命中候选生效); 无法归一出 date/关键值列时抛 AkshareError, 由 collector
# 降级为 None。真实字段映射需联网联调确认(见 docs/experiments.md)。
# ---------------------------------------------------------------------------


def _map_cols(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """按候选映射重命名列; 多个候选可映射同一目标, 先命中者生效."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for cn, en in mapping.items():
        if cn in df.columns and en not in df.columns:
            df = df.rename(columns={cn: en})
    return df


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AkshareError(f"{name}: 接口缺少关键列 {missing} (实际列: {list(df.columns)[:8]})")
    return df


def get_stock_industry(symbol: str) -> str | None:
    """个股所属行业名称(东财个股信息, item/value 两列表). 失败/未找到返回 None."""
    df = _retry(ak.stock_individual_info_em, symbol=symbol)
    df = _map_cols(df, {"item": "item", "项目": "item", "value": "value", "值": "value"})
    if "item" not in df.columns or "value" not in df.columns:
        return None
    row = df[df["item"].astype(str).str.strip() == "行业"]
    if row.empty:
        return None
    val = str(row["value"].iloc[0]).strip()
    return val or None


def get_stock_name(symbol: str) -> str | None:
    """股票简称(东财个股信息, 用于 ST 识别, Phase 5 / 5.1). 失败/未找到返回 None."""
    df = _retry(ak.stock_individual_info_em, symbol=symbol)
    df = _map_cols(df, {"item": "item", "项目": "item", "value": "value", "值": "value"})
    if "item" not in df.columns or "value" not in df.columns:
        return None
    row = df[df["item"].astype(str).str.strip().isin(["股票简称", "名称", "股票名称"])]
    if row.empty:
        return None
    val = str(row["value"].iloc[0]).strip()
    return val or None


def get_industry_hist(industry: str, days: int = 365) -> pd.DataFrame:
    """行业板块日K线(东财). 列同 get_index_kline 口径."""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    df = _retry(
        ak.stock_board_industry_hist_em,
        symbol=industry, period="日k", start_date=start, end_date=end,
    )
    df = _map_cols(df, {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change",
    })
    df = _require_cols(df, ["date", "close"], "get_industry_hist")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_northbound() -> pd.DataFrame:
    """北向资金逐日净流入(亿元). 列: date/value.

    ⚠ 数据可得性风险: 交易所自 2024-08 起停止北向资金逐日净买入披露,
    本接口可能仅返回历史数据(特征在近期样本上为 NaN, 自动降级)。
    """
    df = _retry(ak.stock_hsgt_north_net_flow_in_em)
    df = _map_cols(df, {
        "日期": "date", "date": "date",
        "value": "value", "当日资金流入": "value", "当日成交净买额": "value",
        "净流入": "value",
    })
    df = _require_cols(df, ["date", "value"], "get_northbound")
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)


def get_margin_sse(days: int = 400) -> pd.DataFrame:
    """沪市融资融券总量(市场级融资余额历史, 杠杆情绪代理). 列: date/margin_balance(亿元).

    说明: akshare 无低成本的个股两融历史接口(按日全市场快照需逐日调用),
    故取市场级总量作弱特征(见 docs/experiments.md 范围决策)。
    """
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    df = _retry(ak.stock_margin_sse, start_date=start, end_date=end)
    df = _map_cols(df, {
        "信用交易日期": "date", "日期": "date",
        "融资余额(元)": "margin_balance", "融资余额": "margin_balance",
        "rzye": "margin_balance",
    })
    df = _require_cols(df, ["date", "margin_balance"], "get_margin_sse")
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["margin_balance"] = pd.to_numeric(df["margin_balance"], errors="coerce") / 1e8
    return df.dropna().sort_values("date").reset_index(drop=True)


def get_lhb(days: int = 120) -> pd.DataFrame:
    """龙虎榜全市场明细(近 N 天). 列: date/symbol/net_buy(元).

    返回全市场表(缓存共享, 全票池一次拉取); 按个股过滤在 collector 层。
    """
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    df = _retry(ak.stock_lhb_detail_em, start_date=start, end_date=end)
    df = _map_cols(df, {
        "上榜日期": "date", "日期": "date",
        "代码": "symbol",
        "龙虎榜净买额": "net_buy", "龙虎榜净买入": "net_buy",
    })
    df = _require_cols(df, ["date", "symbol", "net_buy"], "get_lhb")
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.strip().str.zfill(6)
    df["net_buy"] = pd.to_numeric(df["net_buy"], errors="coerce")
    return df.dropna(subset=["date", "net_buy"])[["date", "symbol", "net_buy"]]


def _parse_month(v: object) -> pd.Timestamp | None:
    """解析 '2024年01月份'/'2024-01'/'202401'/datetime 等月份表示 -> 月初 Timestamp."""
    import re

    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    ts = pd.to_datetime(v, errors="coerce")
    if ts is not None and not pd.isna(ts):
        return ts.to_period("M").to_timestamp()
    m = re.search(r"(\d{4})\D?(\d{1,2})", str(v))
    if not m:
        return None
    year, mon = int(m.group(1)), int(m.group(2))
    if not 1 <= mon <= 12:
        return None
    return pd.Timestamp(year=year, month=mon, day=1)


def _monthly_long(df: pd.DataFrame, value_cands: list[str]) -> pd.Series | None:
    """长表(月份列 + 数值列) -> 以月初日期为索引的月度 Series."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in ("月份", "日期", "统计时间", "date") if c in df.columns), None)
    value_col = next((c for c in value_cands if c in df.columns), None)
    if date_col is None or value_col is None:
        return None
    dates = df[date_col].map(_parse_month)
    vals = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=pd.DatetimeIndex(dates)).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if len(s) else None


def get_macro_china() -> pd.DataFrame:
    """宏观月度弱特征. 列: date(月初)/pmi/cpi_yoy/m2_yoy, 单指标失败降级为缺列.

    ⚠ 字段映射不确定性最高的接口组(见 docs/experiments.md); 解析失败时
    对应列为空, 特征层自动 NaN 降级, 不影响主链路。
    """
    out: dict[str, pd.Series] = {}
    try:
        pmi = _monthly_long(_retry(ak.macro_china_pmi_yearly),
                            ["制造业", "制造业PMI", "综合PMI", "PMI"])
        if pmi is not None:
            out["pmi"] = pmi
    except AkshareError:
        pass
    try:
        cpi = _monthly_long(_retry(ak.macro_china_cpi_yearly),
                            ["当月同比", "全国同比", "同比", "CPI同比"])
        if cpi is not None:
            out["cpi_yoy"] = cpi
    except AkshareError:
        pass
    try:
        m2 = _monthly_long(_retry(ak.macro_china_money_supply),
                           ["货币和准货币(M2)同比增长", "M2同比增长", "同比增长"])
        if m2 is not None:
            out["m2_yoy"] = m2
    except AkshareError:
        pass
    if not out:
        raise AkshareError("get_macro_china: 所有宏观指标解析失败")
    df = pd.DataFrame(out)  # 索引 = 月初日期, 缺失指标列为 NaN
    df.index.name = "date"
    return df.reset_index()


def get_holder_events(symbol: str) -> pd.DataFrame:
    """股东增减持事件(巨潮). 列: date(公告日)/amount(带符号股数, 增持为正).

    字段映射不确定, 失败抛 AkshareError 由上层降级。
    """
    df = _retry(ak.stock_share_change_cninfo, symbol=symbol)
    df = _map_cols(df, {
        "公告日期": "date", "公告日": "date", "变动日期": "date", "日期": "date",
        "变动方向": "direction", "增减": "direction",
        "变动数量": "amount", "变动股数": "amount", "数量": "amount",
    })
    df = _require_cols(df, ["date", "direction", "amount"], "get_holder_events")
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    sign = df["direction"].astype(str).map(
        lambda s: 1.0 if "增" in s else (-1.0 if "减" in s else 0.0)
    )
    df["amount"] = sign * pd.to_numeric(df["amount"], errors="coerce").abs()
    return (
        df.dropna(subset=["date", "amount"])[["date", "amount"]]
        .query("amount != 0").sort_values("date").reset_index(drop=True)
    )


def get_unlock_schedule(symbol: str) -> pd.DataFrame:
    """限售股解禁日历. 列: date(解禁日, 可含未来)/ratio(占总股本比例, %).

    特征层只用"已发生"的解禁事件做后顾衰减(防前视契约可机械判定),
    未来日历行不参与任何特征计算。
    """
    df = _retry(ak.stock_restricted_release_queue_em, symbol=symbol)
    df = _map_cols(df, {
        "解禁日期": "date", "解禁时间": "date", "日期": "date",
        "占总股本比例": "ratio", "本次解禁占比": "ratio", "解禁比例": "ratio",
    })
    df = _require_cols(df, ["date", "ratio"], "get_unlock_schedule")
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["ratio"] = (
        df["ratio"].astype(str).str.rstrip("%").pipe(pd.to_numeric, errors="coerce")
    )
    return df.dropna(subset=["date", "ratio"])[["date", "ratio"]].sort_values("date")


def get_global_index(symbol: str) -> pd.DataFrame:
    """海外指数日线(新浪). 列: date/close. 符号如 .IXIC/.INX/.VIX."""
    df = _retry(ak.index_us_stock_sina, symbol=symbol)
    df = _map_cols(df, {"date": "date", "日期": "date", "close": "close", "收盘": "close"})
    df = _require_cols(df, ["date", "close"], f"get_global_index({symbol})")
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)
