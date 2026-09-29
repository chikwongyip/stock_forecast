"""标签定义: close-to-close 三分类 (涨=1 / 平=0 / 跌=-1).

标签 y[T] 描述第 T 日的走势方向: close_T 相对 close_{T-1} 的涨跌幅.
±threshold 内记"平"(噪音带), 该阈值滤掉约 20~30% 无方向意义样本.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UP, FLAT, DOWN = 1, 0, -1


def pct_to_label(pct: float, threshold: float) -> int:
    """涨跌幅(小数) -> 三分类标签."""
    if pct > threshold:
        return UP
    if pct < -threshold:
        return DOWN
    return FLAT


def make_labels(close: pd.Series, threshold: float = 0.005) -> pd.Series:
    """由收盘价序列生成对齐的标签序列.

    y[T] = label(close_T / close_{T-1} - 1); 首行为 NaN(无前收).
    """
    pct = close.pct_change()
    return pct.apply(lambda x: pct_to_label(x, threshold) if pd.notna(x) else np.nan)


LABEL_NAMES = {UP: "涨", FLAT: "平", DOWN: "跌"}
