"""Phase 2 扩展数据源特征: 行业/北向/两融/龙虎榜/增减持/解禁/外盘/宏观.

时间语义契约(与 build.py 一致): 返回帧第 T 行 = "截至第 T 日收盘可得的信息",
本模块一律不做时间平移(统一 shift(1) 由 build.py 执行)。

防前视的"公布时点"显式建模(保证契约可被机械判定 —— 原始帧中日期 > T 的行
绝不影响第 T 行特征):
- 事件类(龙虎榜/增减持/解禁): 第 T 行只聚合 date <= T 且窗口内的事件, 指数衰减加权;
  解禁"未来日历行"属公开信息但会破坏机械可判定性, 明确不使用(只用后顾事件);
- 宏观月频: M 月的值自 M+1 月 15 日起生效(覆盖官方PMI(1日)/CPI(9-10日)/
  M2(10-15日)的实际公布时点, 取保守统一滞后);
- 外盘: 美股 T-1 场次于北京时间 T 日凌晨收盘, 经 build 统一 shift(1) 后
  X[T] 恰好用到"隔夜外盘", 语义天然正确; 日历缺口 ffill 只向后填充(不引入未来)。

所有函数: 输入源帧为 None -> 全 NaN 列(源不可用); 空帧 -> 按源语义给 0 或 NaN;
输出恒为对齐 kline 交易日、列集合固定的 DataFrame(保证跨股票 schema 稳定,
optional 列不进 dropna, NaN 由 LightGBM 原生处理)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 事件衰减半衰期(日历天): ~10 个交易日
EVENT_HALFLIFE_DAYS = 14.0
# 事件回看窗口(日历天): 更久的事件不再计入
EVENT_WINDOW_DAYS = 60
# 宏观公布滞后: M 月值自 M+1 月 15 日生效
MACRO_EFFECTIVE_MONTHS = 1
MACRO_EFFECTIVE_LAG = pd.Timedelta(days=14)


def _macro_effective(month_starts: pd.Series) -> pd.Series:
    """月初日期 -> 该月数据的生效日(M+1 月 15 日)."""
    return month_starts + pd.DateOffset(months=MACRO_EFFECTIVE_MONTHS) + MACRO_EFFECTIVE_LAG


def _nan_frame(n: int, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=pd.RangeIndex(n), columns=cols)


def _kd(kline: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(kline["date"]))


def _daily(df: pd.DataFrame, col: str, kd: pd.DatetimeIndex) -> pd.Series:
    """源帧的 col 列按日期对齐到 kline 交易日(缺失日 NaN, 重复日取最后).

    返回 RangeIndex 位置索引(与特征帧一致), 避免日期索引对齐产生整列 NaN。
    """
    d = df.dropna(subset=["date", col]).drop_duplicates(subset="date", keep="last")
    return d[col].set_axis(pd.DatetimeIndex(d["date"])).reindex(kd).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 批次 A
# ---------------------------------------------------------------------------

INDUSTRY_COLS = ["sec_ret1", "sec_ret5", "sec_ret20", "sec_vol20",
                 "excess_ret1", "excess_ret5"]


def industry_features(ind_hist: pd.DataFrame | None, kline: pd.DataFrame) -> pd.DataFrame:
    """行业板块动量 + 个股相对行业超额收益(替代计划中的"行业内排名", 见 experiments)."""
    kd = _kd(kline)
    n = len(kd)
    f = _nan_frame(n, INDUSTRY_COLS)
    if ind_hist is None or len(ind_hist) == 0 or "close" not in ind_hist.columns:
        return f
    sec = _daily(ind_hist, "close", kd).ffill(limit=3)
    sec_ret = sec.pct_change()
    f["sec_ret1"] = sec_ret
    f["sec_ret5"] = sec.pct_change(5)
    f["sec_ret20"] = sec.pct_change(20)
    f["sec_vol20"] = sec_ret.rolling(20).std()
    c = pd.Series(kline["close"].to_numpy(dtype=float))
    f["excess_ret1"] = c.pct_change().to_numpy() - f["sec_ret1"].to_numpy()
    f["excess_ret5"] = c.pct_change(5).to_numpy() - f["sec_ret5"].to_numpy()
    return f


NORTH_COLS = ["north_net1", "north_net3", "north_net5", "north_div5"]


def north_features(north: pd.DataFrame | None, kline: pd.DataFrame,
                   index: pd.DataFrame | None = None) -> pd.DataFrame:
    """北向净流入 1/3/5 日累计(亿元) + 与指数的 5 日方向背离度.

    north_div5 = sign(北向5日累计) * sign(指数5日收益): -1=背离, +1=同向, 0=中性。
    流量数据不做 ffill(缺失日不累计, min_periods 允许部分求和)。
    """
    kd = _kd(kline)
    f = _nan_frame(len(kd), NORTH_COLS)
    if north is None or len(north) == 0 or "value" not in north.columns:
        return f
    v = _daily(north, "value", kd)
    f["north_net1"] = v
    f["north_net3"] = v.rolling(3, min_periods=2).sum()
    f["north_net5"] = v.rolling(5, min_periods=3).sum()
    if index is not None and len(index) and "close" in index.columns:
        idx5 = _daily(index, "close", kd).ffill(limit=5).pct_change(5)
        f["north_div5"] = np.sign(f["north_net5"]) * np.sign(idx5)
    return f


MARGIN_COLS = ["margin_chg5", "margin_chg20"]


def margin_features(margin: pd.DataFrame | None, kline: pd.DataFrame) -> pd.DataFrame:
    """市场级融资余额变化率(存量概念, 短缺口向后填充不引入未来)."""
    kd = _kd(kline)
    f = _nan_frame(len(kd), MARGIN_COLS)
    if margin is None or len(margin) == 0 or "margin_balance" not in margin.columns:
        return f
    b = _daily(margin, "margin_balance", kd).ffill(limit=10)
    f["margin_chg5"] = b.pct_change(5)
    f["margin_chg20"] = b.pct_change(20)
    return f


# ---------------------------------------------------------------------------
# 批次 B: 事件衰减(通用) / 外盘 / 宏观
# ---------------------------------------------------------------------------

def event_decay_features(events: pd.DataFrame | None, kline: pd.DataFrame,
                         value_col: str, prefix: str, scale: float = 1.0,
                         halflife: float = EVENT_HALFLIFE_DAYS,
                         window: float = EVENT_WINDOW_DAYS) -> pd.DataFrame:
    """通用后顾事件衰减: 第 T 行 = Σ value_i * scale * 0.5^((T-date_i)/halflife).

    只聚合 date_i <= T 且 T-date_i <= window 的事件(严格无前视);
    count 列 = 窗口内事件条数(不衰减)。
    events=None(源不可用) -> NaN; 空帧(源可用但无事件) -> 0(有信息的真值)。
    """
    cols = [f"{prefix}_decay", f"{prefix}_count"]
    kd = _kd(kline)
    n = len(kd)
    if events is None:
        return _nan_frame(n, cols)
    if len(events) == 0 or "date" not in events.columns or value_col not in events.columns:
        return pd.DataFrame(0.0, index=pd.RangeIndex(n), columns=cols)

    ev = events.assign(date=pd.to_datetime(events["date"])).dropna(subset=["date", value_col])
    if ev.empty:
        return pd.DataFrame(0.0, index=pd.RangeIndex(n), columns=cols)
    # 不做同日聚合: count 需要原始事件条数(同日多次上榜分别计数)
    ed = pd.DatetimeIndex(ev["date"]).to_numpy()
    vals = ev[value_col].to_numpy(dtype=float) * scale
    days = (kd.to_numpy()[:, None] - ed[None, :]) / np.timedelta64(1, "D")
    in_win = (days >= 0) & (days <= window)
    # 窗口外/未来事件权重为 0 (0.5**inf = 0, 无溢出告警)
    w = 0.5 ** np.where(in_win, days / halflife, np.inf)
    return pd.DataFrame({
        f"{prefix}_decay": w @ vals,
        f"{prefix}_count": in_win.sum(axis=1).astype(float),
    })


GLOBAL_COLS = ["us_nasdaq_ret1", "us_spx_ret1", "vix_level", "vix_chg5"]
_GLOBAL_MAP = {"us_nasdaq_ret1": "nasdaq", "us_spx_ret1": "spx", "vix_level": "vix"}


def global_features(glob: pd.DataFrame | None, kline: pd.DataFrame) -> pd.DataFrame:
    """隔夜外盘: 纳指/标普单日收益、VIX 水平与 5 日变化.

    glob 列: date/nasdaq/spx/vix(收盘价, collector 合并)。美股日历缺口
    ffill(limit=7) 只向后填充。行 T 值 = 美股 T 场次收盘(北京 T+1 凌晨),
    经 build 统一 shift(1) 后 X[T] 用到 T-1 场次 = T 日开盘前的隔夜信息。
    """
    kd = _kd(kline)
    f = _nan_frame(len(kd), GLOBAL_COLS)
    if glob is None or len(glob) == 0:
        return f
    for col, src in _GLOBAL_MAP.items():
        if src not in glob.columns:
            continue
        s = _daily(glob, src, kd).ffill(limit=7)
        if col == "vix_level":
            f[col] = s
            f["vix_chg5"] = s.pct_change(5)
        else:
            f[col] = s.pct_change()
    return f


MACRO_COLS = ["macro_pmi", "macro_pmi_mom", "macro_cpi_yoy", "macro_m2_yoy"]


def macro_features(macro: pd.DataFrame | None, kline: pd.DataFrame) -> pd.DataFrame:
    """宏观月度弱特征: 月频值 + PMI 环比, 按公布时点滞后生效再日频 ffill.

    macro 列: date(月初)/pmi/cpi_yoy/m2_yoy(可缺列)。M 月行自 M+1 月 15 日
    起生效 —— 生效日之前的任何特征行都取不到该月的值(防前视)。
    """
    kd = _kd(kline)
    f = _nan_frame(len(kd), MACRO_COLS)
    if macro is None or len(macro) == 0 or "date" not in macro.columns:
        return f
    m = macro.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.dropna(subset=["date"]).drop_duplicates(subset="date", keep="last").sort_values("date")
    if "pmi" in m.columns:
        m["pmi_mom"] = pd.to_numeric(m["pmi"], errors="coerce").diff()
    eff = _macro_effective(m["date"])
    src_cols = {"macro_pmi": "pmi", "macro_pmi_mom": "pmi_mom",
                "macro_cpi_yoy": "cpi_yoy", "macro_m2_yoy": "m2_yoy"}
    for target, src in src_cols.items():
        if src not in m.columns:
            continue
        s = pd.Series(pd.to_numeric(m[src], errors="coerce").to_numpy(), index=eff)
        s = s.dropna()
        s = s[~s.index.duplicated(keep="last")]
        if s.empty:
            continue
        combined = s.index.union(kd).sort_values()
        f[target] = s.reindex(combined).ffill(limit=45).reindex(kd).to_numpy()
    return f
