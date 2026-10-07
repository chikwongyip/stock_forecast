"""L4 市场状态识别 (Phase 3 规则引擎 v1): 牛/熊/震荡三状态.

规则(参数见 config.Settings.regime_*, 全部可通过环境变量调整):
- 多头排列(收盘 > MA20 > MA60)且 20 日波动率分位未达高位 -> 牛市
- 空头排列(收盘 < MA20 < MA60) -> 熊市
- 其余(排列混杂; 或多头排列但波动率分位 >= regime_high_vol_q 的高波动市) -> 震荡
- 暖机期(MA 慢线尚无值) -> 未知(融合层视为不可用, 不做状态调整)

时间语义(与 build.py 防前视契约一致): classify_regimes 第 T 行 = 截至 T 日
收盘(含)的信息, 本模块不做时间平移。**对"预测第 T 日走势"的场景必须取 T-1 行**
(regime_for_prediction / shifted_regimes 已内置该对齐)—— 与 X[T] 只含 ≤T-1
数据的口径一致; 若直接用 T 行, 状态会携带当日收盘信息, 经融合调整泄露进对
y[T] 的预测(track 结算出的方向准确率将失真)。

本模块不产出模型特征列(X schema 不变, 旧模型兼容, 无需重训)。消费方:
fusion.combine(震荡收紧方向阈值 / 熊市下调偏涨置信)、output.render(报告展示)、
model.backtest(分状态方向准确率报表)、monitor.track(埋点 regime 字段分桶)。

HMM v2(develop_plan 3.3, 可选项)本版未实施: 需新增 hmmlearn 依赖并重建镜像,
且规则引擎的真实分状态收益尚未验证, 先交付纯 pandas 规则版
(决策记录见 docs/experiments.md)。
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_forecast.config import get_settings

console = Console()

REGIME_BULL = "牛市"
REGIME_BEAR = "熊市"
REGIME_RANGE = "震荡"
REGIME_UNKNOWN = "未知"

_REGIME_COLOR = {"牛市": "green", "熊市": "red", "震荡": "yellow", "未知": "white"}


def classify_regimes(index: pd.DataFrame, settings=None) -> pd.DataFrame:
    """向量化三状态识别. 返回与 index 等长、date 为 DatetimeIndex 的帧.

    列: regime/close/ma_fast/ma_slow/vol20/vol_quantile。
    第 T 行 = 截至 T 日收盘的状态(预测场景请用 regime_for_prediction 取 T-1 行)。
    波动率分位样本不足(min_periods 未满)时不触发高波动覆盖, 仅按均线排列分类;
    MA 慢线暖机期内的行标记为"未知"。
    """
    settings = settings or get_settings()
    idx = index.reset_index(drop=True)
    close = pd.to_numeric(idx["close"], errors="coerce")
    dates = pd.DatetimeIndex(pd.to_datetime(idx["date"]))

    ma_fast = close.rolling(settings.regime_ma_fast).mean()
    ma_slow = close.rolling(settings.regime_ma_slow).mean()
    vol20 = close.pct_change().rolling(settings.regime_ma_fast).std()
    # 波动率分位 = 当前 vol20 在回看窗口内的百分位秩。
    # 先 dropna 再 rank(避免暖机期 NaN 干扰窗口计数), 完成后对齐回原索引。
    v = vol20.dropna()
    vol_q = (
        v.rolling(settings.regime_vol_window, min_periods=settings.regime_vol_min_periods)
        .rank(pct=True)
        .reindex(vol20.index)
    )

    warm = ma_slow.notna() & close.notna()
    high_vol = vol_q >= settings.regime_high_vol_q  # NaN -> False: 分位样本不足不触发覆盖
    bull = warm & (close > ma_fast) & (ma_fast > ma_slow) & ~high_vol
    bear = warm & (close < ma_fast) & (ma_fast < ma_slow)

    regime = pd.Series(REGIME_RANGE, index=close.index, dtype=object)
    regime[bull] = REGIME_BULL
    regime[bear] = REGIME_BEAR
    regime[~warm] = REGIME_UNKNOWN

    out = pd.DataFrame({
        "regime": regime, "close": close, "ma_fast": ma_fast,
        "ma_slow": ma_slow, "vol20": vol20, "vol_quantile": vol_q,
    })
    out.index = dates
    return out


def shifted_regimes(index: pd.DataFrame, settings=None) -> pd.Series:
    """行 T = T-1 日收盘时的状态(与 X[T] 信息时点对齐), 供回测/评估按日期映射."""
    return classify_regimes(index, settings)["regime"].shift(1)


def _describe(row: pd.Series, settings) -> str:
    """单行判定依据(中文, 供报告 / --json 展示)."""
    fq, sq = settings.regime_ma_fast, settings.regime_ma_slow
    c, mf, ms = row["close"], row["ma_fast"], row["ma_slow"]
    vq = row["vol_quantile"]
    vq_txt = f"20日波动率分位 {vq:.0%}" if pd.notna(vq) else "波动率分位样本不足"

    if row["regime"] == REGIME_BULL:
        tail = (f"{vq_txt}, 未达 {settings.regime_high_vol_q:.0%} 高位线"
                if pd.notna(vq) else vq_txt)
        return f"收盘 {c:.0f} > MA{fq} {mf:.0f} > MA{sq} {ms:.0f} 多头排列; {tail}"
    if row["regime"] == REGIME_BEAR:
        return f"收盘 {c:.0f} < MA{fq} {mf:.0f} < MA{sq} {ms:.0f} 空头排列; {vq_txt}"
    if row["regime"] == REGIME_UNKNOWN:
        return f"指数历史不足(需 ≥ {sq} 个交易日暖机), 状态无法识别"
    # 震荡: 区分"高波动覆盖"与"排列混杂"两种成因
    if pd.notna(vq) and vq >= settings.regime_high_vol_q and c > mf > ms:
        return f"虽多头排列, 但 {vq_txt} ≥ {settings.regime_high_vol_q:.0%}, 高波动市按震荡处理"
    return (f"MA{fq}/MA{sq} 排列混杂(收盘 {c:.0f}, MA{fq} {mf:.0f}, MA{sq} {ms:.0f}), "
            f"趋势不明; {vq_txt}")


def _f(v) -> float | None:
    """numpy/pandas 标量 -> 原生 float(json 可序列化); NaN -> None."""
    return None if v is None or pd.isna(v) else round(float(v), 4)


def regime_for_prediction(index: pd.DataFrame | None, settings=None) -> dict:
    """predict 场景的状态判定: 取倒数第 2 行(T-1 收盘), 与特征 shift(1) 同口径.

    返回 dict(全部原生类型, 可直接 json 序列化 / 写入埋点):
    regime/available/reason/as_of/close/ma_fast/ma_slow/vol20/vol_quantile。
    index 不可用、行数不足或尚在暖机期 -> available=False(展示用 regime=震荡,
    融合层应传 None 不做任何状态调整, 优雅降级)。
    """
    settings = settings or get_settings()

    def _unavailable(reason: str) -> dict:
        return {"regime": REGIME_RANGE, "available": False, "reason": reason,
                "as_of": None, "close": None, "ma_fast": None, "ma_slow": None,
                "vol20": None, "vol_quantile": None}

    if index is None or len(index) < 2 or "close" not in index.columns:
        return _unavailable("指数数据不可用, 市场状态未识别(不做状态调整)")
    r = classify_regimes(index, settings)
    row = r.iloc[-2]
    if row["regime"] == REGIME_UNKNOWN:
        return _unavailable(_describe(row, settings))
    return {
        "regime": row["regime"], "available": True,
        "reason": _describe(row, settings),
        "as_of": str(r.index[-2].date()),
        "close": _f(row["close"]), "ma_fast": _f(row["ma_fast"]),
        "ma_slow": _f(row["ma_slow"]), "vol20": _f(row["vol20"]),
        "vol_quantile": _f(row["vol_quantile"]),
    }


def cmd_regime(args: argparse.Namespace) -> int:
    """regime 子命令: 展示当前市场状态、判定依据与近期分布(规则引擎 v1)."""
    from stock_forecast.data import collector  # 延迟 import: 避免模块级拖入 akshare

    settings = get_settings()
    try:
        index = collector.fetch_index()
    except Exception as e:  # noqa: BLE001 - 数据不可用明确报告后退出, 不 crash
        console.print(f"[red]指数数据不可用: {type(e).__name__}: {e}[/red]")
        return 1
    if index is None or len(index) == 0:
        console.print("[red]指数缓存为空, 请先运行 fetch.[/red]")
        return 1

    r = classify_regimes(index, settings)
    cur = r.iloc[-1]
    dist = r["regime"].tail(60).value_counts()
    payload = {
        "current": {
            "as_of": str(r.index[-1].date()), "regime": cur["regime"],
            "reason": _describe(cur, settings),
            "close": _f(cur["close"]), "ma_fast": _f(cur["ma_fast"]),
            "ma_slow": _f(cur["ma_slow"]), "vol20": _f(cur["vol20"]),
            "vol_quantile": _f(cur["vol_quantile"]),
        },
        # 预测口径 = T-1 行(与特征信息时点对齐), predict 报告展示的就是它
        "for_prediction": regime_for_prediction(index, settings),
        "last60_distribution": {str(k): int(v) for k, v in dist.items()},
        "last10_sequence": " → ".join(r["regime"].tail(10)),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    c = payload["current"]
    pred = payload["for_prediction"]
    color = _REGIME_COLOR.get(c["regime"], "white")
    t = Table(title="市场状态识别 (上证指数 · 规则引擎 v1)", show_header=False)
    t.add_column("项目")
    t.add_column("值")
    t.add_row("当前状态(最新收盘)",
              f"[{color} bold]{c['regime']}[/{color} bold] @ {c['as_of']}")
    t.add_row("判定依据", c["reason"])
    pred_state = pred["regime"] if pred["available"] else "未识别"
    t.add_row("预测口径状态(T-1 对齐)",
              f"{pred_state} @ {pred['as_of'] or '-'} — {pred['reason']}")
    t.add_row("近60交易日分布",
              " | ".join(f"{k} {v}天" for k, v in payload["last60_distribution"].items()))
    t.add_row("近10日状态", payload["last10_sequence"])
    console.print(t)
    console.print(
        "[dim]规则: 收盘>MA20>MA60 且波动率分位<85% -> 牛市; 收盘<MA20<MA60 -> 熊市; "
        "其余 -> 震荡(参数 regime_* 可调)。predict 报告采用 T-1 口径状态: "
        "震荡市收紧方向阈值, 熊市下调偏涨置信。[/dim]"
    )
    return 0
