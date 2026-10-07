"""极端场景扩充 (Phase 5 / L6 / 5.3): 连续一字板 / 近20日振幅异常 / 量能枯竭.

在 `fusion.combine.detect_extreme`(近5日累计涨跌 + 单日一字板)之外补充三类
"历史模式参考价值低"的极端形态; 纯函数, 输入个股K线尾段(需覆盖最长窗口,
默认 60 日量能分位 → 调用方传 kline.tail(>=61)), 输出中文风险提示列表。
参数全在 config(`extreme_*` / `volume_dry_*`), 数据不足时静默跳过(不误报)。
"""

from __future__ import annotations

import pandas as pd

from stock_forecast.config import get_settings


def _yizi_streak(df: pd.DataFrame) -> int:
    """从最新一日往回数连续一字板(high==low)天数."""
    if not {"high", "low"} <= set(df.columns):
        return 0
    streak = 0
    for i in range(len(df) - 1, -1, -1):
        hi, lo = df["high"].iloc[i], df["low"].iloc[i]
        if pd.notna(hi) and pd.notna(lo) and hi == lo:
            streak += 1
        else:
            break
    return streak


def detect_extreme_scenarios(kline_tail: pd.DataFrame | None, settings=None) -> list[str]:
    """识别 5.3 三类极端场景, 返回风险提示(空列表 = 无异常/数据不足)."""
    settings = settings or get_settings()
    notes: list[str] = []
    if kline_tail is None or kline_tail.empty:
        return notes
    df = kline_tail

    # ① 连续一字板(流动性枯竭, 价格无法成交)
    streak = _yizi_streak(df)
    if streak >= settings.extreme_yizi_streak:
        notes.append(
            f"连续 {streak} 日一字板, 流动性严重受限, 历史模式参考价值极低"
        )

    # ② 近 N 日振幅异常(区间高低差 / 中位收盘)
    w = settings.extreme_amp_window
    if {"high", "low", "close"} <= set(df.columns) and len(df) >= w:
        seg = df.tail(w)
        med = seg["close"].median()
        if pd.notna(med) and med > 0:
            amp = (seg["high"].max() - seg["low"].min()) / float(med)
            if amp >= settings.extreme_amplitude:
                notes.append(
                    f"近 {w} 日振幅达 {amp:.0%}(区间高低差/中位收盘), 属异常波动行情"
                )

    # ③ 量能枯竭(当日成交量在回看窗口内的分位 <= volume_dry_quantile)
    if "volume" in df.columns and len(df) >= settings.volume_dry_min_samples + 1:
        vol = df["volume"].tail(settings.volume_dry_window)
        latest = vol.iloc[-1]
        hist = vol.iloc[:-1]
        if len(hist) >= settings.volume_dry_min_samples and pd.notna(latest):
            pct = float((hist < latest).mean())
            # 中位数守卫: 等量序列(分位=0 但量并不低)不误报
            if pct <= settings.volume_dry_quantile and latest < float(hist.median()):
                notes.append(
                    f"当日成交量处于近 {len(hist)} 日最低 "
                    f"{settings.volume_dry_quantile:.0%} 分位(量能枯竭), "
                    "市场承接不足, 信号易失效"
                )
    return notes
