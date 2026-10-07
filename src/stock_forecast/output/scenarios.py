"""预测分布区间 (Phase 6 / 6.1): 三分类概率 × 历史条件幅度 → 情景区间描述.

模型输出的是三分类概率(涨/平/跌), 不是收益分布; 本模块把概率转成可展示的
p25/p50/p75 情景区间:
- 条件幅度: 回看窗口内按标签分组的历史收益均值(涨日均值 / 平日均值 / 跌日均值);
- 分布近似: 三点混合 {跌: p_down, 平: p_flat, 涨: p_up}, 分位数 = 按概率累计
  加权取原子(3 个原子, 分位数可能落在同一原子, 属如实呈现);
- **防前视(与全项目同一契约)**: 预测目标是最后一行日期 D 的 close-to-close
  收益, 而该收益已含在 kline 中(系统收盘后运行)——条件幅度只用 **≤ D-1 的收益**
  (`pct_change().iloc[:-1]`), 展示口径与模型信息时点(X[D] 只含 ≤ D-1)一致;
  `test_scenarios.py` 有硬断言(篡改最后一行, 输出必须一字不变)。
- 数据不足(< 31 行有效收益)返回 None, 展示端降级隐藏该段。
"""

from __future__ import annotations

import pandas as pd

MIN_SAMPLES = 30


def _weighted_quantile(atoms: list[tuple[str, float, float]], q: float) -> tuple[float, str]:
    """atoms = [(情景名, 幅度, 概率)](幅度升序), 返回 q 分位对应的 (幅度, 情景名)."""
    cum = 0.0
    for name, mu, w in atoms:
        cum += w
        if cum >= q - 1e-12:
            return mu, name
    return atoms[-1][1], atoms[-1][0]


def scenario_quantiles(
    probs: dict[str, float],
    kline: pd.DataFrame | None,
    label_threshold: float,
    lookback: int = 120,
) -> dict | None:
    """三分类概率 + 历史条件幅度 → {q25/q50/q75, 情景表, 口径说明}; 数据不足返回 None."""
    if kline is None or "close" not in getattr(kline, "columns", []) or len(kline) < 2:
        return None
    # 防前视: 剔除最后一行收益(= 本次预测的标签本身)
    rets = kline["close"].pct_change().iloc[:-1].tail(lookback).dropna()
    if len(rets) < MIN_SAMPLES:
        return None

    up = rets[rets > label_threshold]
    down = rets[rets < -label_threshold]
    flat = rets[(rets >= -label_threshold) & (rets <= label_threshold)]
    # 某组历史为空时用阈值外沿兜底(保守估计), 保证三点恒可构造
    up_mu = float(up.mean()) if len(up) else label_threshold * 2
    down_mu = float(down.mean()) if len(down) else -label_threshold * 2
    flat_mu = float(flat.mean()) if len(flat) else 0.0

    named = [("悲观", down_mu, float(probs["p_down"])),
             ("中性", flat_mu, float(probs["p_flat"])),
             ("乐观", up_mu, float(probs["p_up"]))]
    atoms = sorted(named, key=lambda a: a[1])  # 按幅度升序做加权分位

    q25, q50, q75 = (_weighted_quantile(atoms, q) for q in (0.25, 0.50, 0.75))
    return {
        "q25": round(q25[0], 4), "q25_scenario": q25[1],
        "q50": round(q50[0], 4), "q50_scenario": q50[1],
        "q75": round(q75[0], 4), "q75_scenario": q75[1],
        "scenarios": [
            {"name": n, "prob": round(w, 4), "typical_move": round(mu, 4)}
            for n, mu, w in named  # 固定 悲观/中性/乐观 顺序展示
        ],
        "lookback_days": int(len(rets)),
        "note": "幅度为历史分组条件均值(截至 T-1 收盘), 仅情景示意, 不是收益承诺",
    }


def format_scenario_lines(sq: dict) -> list[str]:
    """终端展示用的紧凑文本行(2 行: 分位数 + 情景表)."""
    if not sq:
        return []
    line1 = (
        f"p25 ≈ {sq['q25']:+.1%}({sq['q25_scenario']})   "
        f"p50 ≈ {sq['q50']:+.1%}({sq['q50_scenario']})   "
        f"p75 ≈ {sq['q75']:+.1%}({sq['q75_scenario']})"
    )
    parts = [f"{s['name']} 概率 {s['prob']:.0%} 典型 {s['typical_move']:+.1%}"
             for s in sq["scenarios"]]
    return [line1, " | ".join(parts), sq["note"]]
