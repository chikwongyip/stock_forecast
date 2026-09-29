"""回测评估指标.

主指标: 方向准确率 —— 剔除真实标签为"平"的样本后, 预测涨/跌的命中率.
判据: 52%~65% 可信; >70% 大概率泄露, 直接报警.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEAK_WARNING_THRESHOLD = 0.70


def direction_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float((y_pred[mask] == y_true[mask]).mean())


def overall_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_pred == y_true).mean())


def class_recall(y_true: pd.Series, y_pred: pd.Series) -> dict[int, float]:
    out = {}
    for lab in (-1, 0, 1):
        mask = y_true == lab
        out[lab] = float((y_pred[mask] == lab).mean()) if mask.any() else float("nan")
    return out


def summarize(results: pd.DataFrame) -> dict:
    """results 需含列: y_true, y_pred (还有 date/symbol 供分组)."""
    y_true, y_pred = results["y_true"], results["y_pred"]
    summary = {
        "n_samples": len(results),
        "overall_accuracy": overall_accuracy(y_true, y_pred),
        "direction_accuracy": direction_accuracy(y_true, y_pred),
        "class_recall": class_recall(y_true, y_pred),
        "label_distribution": y_true.value_counts(normalize=True).to_dict(),
    }
    if "fold" in results.columns:
        per_fold = results.groupby("fold").apply(
            lambda g: direction_accuracy(g["y_true"], g["y_pred"]), include_groups=False
        )
        summary["per_fold_direction_accuracy"] = per_fold.tolist()
        summary["fold_accuracy_std"] = float(np.nanstd(per_fold)) if len(per_fold) else float("nan")
    summary["possible_leakage"] = (
        not np.isnan(summary["direction_accuracy"])
        and summary["direction_accuracy"] > LEAK_WARNING_THRESHOLD
    )
    return summary
