"""交易状态识别 (Phase 5 / L6 / 5.1): 停牌 / 涨跌停 / ST → 适用性降级.

纯函数模块(无 IO, 离线可单测)。判定分两级:
- **不适用**(applicable=False): 当前停牌(个股K线最新日落后于大盘最新交易日)——
  价格不产生, 预测无意义, 融合层强制"震荡/观望 + 低置信";
- **仅供参考**(reference_only=True): 当日涨跌停、ST/*ST、近期有停牌断档——
  信号可能失真, 融合层把置信度封顶到"低"。

涨跌停判定口径: 缓存为**前复权**价, 无法用"收盘价 = round(前收×(1±限幅),2)"的
精确涨停价匹配(复权因子会破坏分位舍入), 故改用当日收益 vs 板别限幅(容差
`limit_move_tol`); 优先用 akshare 的官方 pct_change 列, 缺失时退回 close-to-close。
ST 判定依赖股票简称(需联网, 见 collector.fetch_stock_name), 名称缺失时跳过该项。
"""

from __future__ import annotations

import pandas as pd

from stock_forecast.config import get_settings

# 交易状态标签严重度(用于挑主标签, 数字越小越严重)
_SEVERITY = {"停牌": 0, "涨停": 1, "跌停": 2, "ST": 3, "近期停牌": 4, "正常": 9}


def price_limit_for(symbol: str, is_st: bool = False) -> float:
    """板别涨跌停限幅(小数). 创业板/科创板 20%, 北交所 30%, 主板 ST 5%, 主板 10%.

    注: 创业板/科创板注册制后 ST 不改限幅(仍 20%), 故先判板块再判 ST。
    """
    s = str(symbol)
    if s.startswith(("688", "689", "300", "301", "302")):
        return 0.20
    if s.startswith(("8", "4", "92")):  # 北交所(83/87/88/43/92 等)
        return 0.30
    if is_st:
        return 0.05
    return 0.10


def detect_st(name: str | None) -> bool:
    """股票简称是否含 ST 标记(覆盖 ST/*ST/退市整理). 名称缺失返回 False(跳过)."""
    if not name:
        return False
    return "ST" in str(name).upper()


def detect_limit_move(
    kline: pd.DataFrame | None, symbol: str, is_st: bool = False, settings=None
) -> str | None:
    """最新交易日是否涨/跌停: 返回 "涨停"/"跌停"/None."""
    settings = settings or get_settings()
    if kline is None or len(kline) < 2 or "close" not in kline.columns:
        return None
    limit = price_limit_for(symbol, is_st)
    tol = settings.limit_move_tol
    last = kline.iloc[-1]
    if "pct_change" in kline.columns and pd.notna(last.get("pct_change")):
        r = float(last["pct_change"]) / 100.0
    else:
        prev = float(kline["close"].iloc[-2])
        if prev <= 0 or pd.isna(prev):
            return None
        r = float(last["close"]) / prev - 1.0
    if pd.isna(r):
        return None
    if r >= limit - tol:
        return "涨停"
    if r <= -(limit - tol):
        return "跌停"
    return None


def detect_suspension(
    kline: pd.DataFrame | None, index: pd.DataFrame | None, settings=None
) -> tuple[bool, bool, int, object, object]:
    """停牌识别(K线断档). 返回 (当前停牌, 近期停牌, 断档天数, 个股最新日, 大盘最新日).

    - 当前停牌: 个股K线最新交易日 < 大盘最新交易日(个股最近一日无行情);
    - 近期停牌: 回看窗口内(stock_last 之前)个股缺失 >= suspend_gap_threshold 个
      大盘交易日(停牌后复牌, 历史模式连续性受损);
    - index 不可用时无法判定, 全部返回 False(优雅降级)。
    """
    settings = settings or get_settings()
    if kline is None or len(kline) == 0 or "date" not in kline.columns:
        return False, False, 0, None, None
    stock_dates = set(pd.to_datetime(kline["date"]).dt.date)
    stock_last = max(stock_dates)
    if index is None or len(index) == 0 or "date" not in index.columns:
        return False, False, 0, stock_last, None
    idx_dates = sorted(set(pd.to_datetime(index["date"]).dt.date))
    idx_last = idx_dates[-1]
    current = stock_last < idx_last
    window = idx_dates[-settings.suspend_gap_lookback:]
    gap_days = sum(1 for d in window if d not in stock_dates and d < stock_last)
    recent = (not current) and gap_days >= settings.suspend_gap_threshold
    return current, recent, gap_days, stock_last, idx_last


def assess_trading_status(
    symbol: str,
    kline: pd.DataFrame | None,
    index: pd.DataFrame | None = None,
    name: str | None = None,
    settings=None,
) -> dict:
    """综合交易状态评估(5.1 主入口).

    返回 dict: applicable(bool) / reference_only(bool) / label(str 主状态) /
    statuses(命中列表) / reason(中文说明) / warnings(需追加到融合层警告的条目)。
    任何数据缺失都降级为"正常/适用", 不 crash。
    """
    settings = settings or get_settings()
    statuses: list[str] = []
    warnings: list[str] = []
    applicable = True
    reference_only = False

    # ST/*ST(需股票简称; 缺失则跳过)
    is_st = detect_st(name)
    if is_st:
        statuses.append("ST")
        reference_only = True
        warnings.append(
            f"{name or symbol} 为 ST/*ST 股票(退市风险, 主板限幅 5%), 预测仅供参考"
        )

    # 停牌(当前 → 不适用; 近期断档 → 仅供参考)
    current, recent, gap_days, stock_last, idx_last = detect_suspension(kline, index, settings)
    if current:
        applicable = False
        statuses.append("停牌")
        warnings.append(
            f"个股当前停牌(K线止于 {stock_last}, 大盘已到 {idx_last}), 预测不适用"
        )
    elif recent:
        reference_only = True
        statuses.append("近期停牌")
        warnings.append(
            f"近 {settings.suspend_gap_lookback} 个交易日内有 {gap_days} 日停牌断档, "
            "历史模式连续性受损, 预测仅供参考"
        )

    # 涨跌停(当前停牌时无意义, 跳过)
    if not current:
        limit = detect_limit_move(kline, symbol, is_st, settings)
        if limit:
            statuses.append(limit)
            reference_only = True
            warnings.append(
                f"当日{limit}, 单边行情情绪极端, 次日价格发现可能失真, 预测仅供参考"
            )

    if not statuses:
        statuses.append("正常")
    label = min(statuses, key=lambda s: _SEVERITY.get(s, 5))
    return {
        "applicable": applicable,
        "reference_only": reference_only,
        "label": label,
        "statuses": statuses,
        "reason": "; ".join(warnings) if warnings else "无异常交易状态",
        "warnings": warnings,
    }
