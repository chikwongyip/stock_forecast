"""特征漂移监控 (Phase 7 / 7.1, L8): PSI 对比近端特征分布 vs 训练期基线.

设计:
- 基线在 `backtest --pool` / `retrain` 落盘最终模型时一并生成
  (`build_baseline` -> `data/models/drift_baseline.json`), **随模型代际归档**:
  retrain/rollback 时与模型工件同进退(旧模型的基线只对旧模型有意义);
- PSI 口径: **只在非 NaN 样本上计算**(两侧比例各自归一), 每特征按训练期
  分位数分 `drift_bins` 箱; 缺失率单独做**单边上升**检查(近端 NaN 比例 -
  训练期 >= drift_nan_delta -> alert): 覆盖率恶化(数据源断供)是漂移,
  覆盖率改善(如资金流仅覆盖近 ~120 日, 训练期前段天然 NaN)不是 ——
  若把 NaN 并进 PSI 分箱, 后者会造成结构性误报;
- 边界去重后不足时精度降级: 常量列只剩 1 条边界(只监控单边漂移),
  非 NaN 样本 < 2 记 degenerate(数值漂移不监控, 只剩缺失率检查);
- 分级: PSI >= drift_psi_warn -> warn; >= drift_psi_alert -> alert;
  缺失率跳升直接 alert。warn/alert 均追加 `data/alerts.csvl`;
  **cmd_drift 退出码 2 = 存在 alert**(约定给 scripts/maintain.sh 等
  自动化脚本分支触发重训, 非错误);
- 已知局限(真实数据校准观察项, 见 docs/experiments.md): 水平/平滑类特征
  (ma20 等)强自相关, 近端窗口的**有效样本量 ≈ 票数**(每票一段近邻值挤在
  少数箱), PSI 噪声底随票数下降: 全票池 20 只可用默认阈值, 只查 1~2 只票时
  alert 多为噪声, 判读要保守; 空箱比例做半样本平滑(0.5/n 为底), 惩罚随
  样本量缩放而非固定 eps;
- 基线中特征在近端矩阵缺失(schema 漂移/缓存过旧) -> 单独 alert(kind=feature_missing);
- 优雅降级(不变量 5): 无基线 -> 黄色提示返回 0; 单票数据失败跳过;
  全票池无可用样本 -> 返回 1。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_forecast.config import (
    ALERTS_LOG,
    DRIFT_BASELINE_FILE,
    get_settings,
)

console = Console()

BASELINE_VERSION = 1
_PSI_EPS = 1e-6  # 防止 log(0): 比例下限

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_ALERT = "alert"


# ---------------------------------------------------------------------------
# PSI 纯函数
# ---------------------------------------------------------------------------

def psi_values(expected: np.ndarray, actual: np.ndarray, eps: float = _PSI_EPS) -> float:
    """PSI = Σ (a_i - e_i) * ln(a_i / e_i); 两侧比例 clip 到 eps 下限防 log(0)."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(((a - e) * np.log(a / e)).sum())


def _bin_props(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """非 NaN 值按边界分箱后的比例向量(长度 len(edges)+1); 空输入 -> 全 0."""
    if values.size == 0:
        return np.zeros(len(edges) + 1) if edges.size else np.zeros(1)
    idx = np.searchsorted(edges, values, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)[: len(edges) + 1]
    return counts / values.size


def build_baseline(X: pd.DataFrame, n_bins: int = 10) -> dict:
    """训练期特征矩阵 -> PSI 基线: 每特征分位数边界 + 非 NaN 期望比例 + 缺失率.

    expected 为**非 NaN 样本内**的分箱比例(归一, 长度 len(edges)+1), 与
    `feature_psi` 的近端口径严格一致; 边界取非 NaN 值的等频分位点(去重),
    去重后不足 1 个边界(常量列/样本 <2)记 degenerate=True。
    """
    features: dict[str, dict] = {}
    for col in X.columns:
        v = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        n = v.size
        nan_ratio = float(np.isnan(v).mean()) if n else 1.0
        nn = v[~np.isnan(v)]
        if nn.size >= 2:
            qs = np.quantile(nn, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
            edges = np.unique(qs[np.isfinite(qs)])
        else:
            edges = np.array([], dtype=float)
        props = _bin_props(nn, edges) if nn.size else np.ones(edges.size + 1 or 1)
        features[str(col)] = {
            "edges": [float(e) for e in edges],
            "expected": [float(p) for p in props],
            "nan_ratio": nan_ratio,
            "degenerate": bool(edges.size == 0),
        }
    return {
        "baseline_version": BASELINE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_bins": int(n_bins),
        "n_samples": int(len(X)),
        "features": features,
    }


def feature_psi(values: pd.Series, spec: dict) -> tuple[float | None, float] | None:
    """近端样本 vs 单特征基线 -> (psi, 近端 NaN 比例).

    PSI 只在非 NaN 样本上计算(两侧各自归一); 近端全 NaN 时 psi=None
    (由缺失率跳升检查兜底)。结构不一致/空输入返回 None。
    """
    edges = np.asarray(spec.get("edges", []), dtype=float)
    expected = np.asarray(spec.get("expected", []), dtype=float)
    if expected.size != edges.size + 1:
        return None
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if v.size == 0:
        return None
    nn = v[~np.isnan(v)]
    nan_actual = float(1.0 - nn.size / v.size)
    if nn.size == 0:
        return None, nan_actual
    # 半样本平滑: 空箱比例以 0.5/n 为底(而非固定 eps)——空箱惩罚随样本量缩放,
    # 抑制小样本+自相关(近端窗口每票仅 ~1 个独立簇)造成的结构性噪声底
    props = np.maximum(_bin_props(nn, edges), 0.5 / nn.size)
    return psi_values(expected, props), nan_actual


def classify_level(psi: float, warn: float, alert: float) -> str:
    if psi >= alert:
        return LEVEL_ALERT
    if psi >= warn:
        return LEVEL_WARN
    return LEVEL_OK


def evaluate_feature(
    name: str, values: pd.Series, spec: dict,
    warn: float, alert: float, nan_delta: float,
) -> dict | None:
    """单特征漂移评估(纯函数): PSI 分级 + 缺失率单边上升检查.

    返回行 dict {feature, psi(可 None), level, nan_expected, nan_actual,
    nan_jump, degenerate}; 无法评估(结构不一致/近端无样本)返回 None。
    缺失率**改善**不告警(覆盖期变长是好事), 只有恶化 >= nan_delta 才 alert。
    """
    out = feature_psi(values, spec)
    if out is None:
        return None
    psi, nan_actual = out
    nan_expected = float(spec.get("nan_ratio", 0.0))
    nan_jump = (nan_actual - nan_expected) >= nan_delta
    level = classify_level(psi, warn, alert) if psi is not None else LEVEL_OK
    if nan_jump:
        level = LEVEL_ALERT
    return {
        "feature": name, "psi": None if psi is None else round(psi, 4),
        "level": level, "nan_expected": round(nan_expected, 4),
        "nan_actual": round(nan_actual, 4), "nan_jump": bool(nan_jump),
        "degenerate": bool(spec.get("degenerate", False)),
    }


# ---------------------------------------------------------------------------
# 基线与告警 IO
# ---------------------------------------------------------------------------

def save_baseline(baseline: dict, path: Path = DRIFT_BASELINE_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, ensure_ascii=False), encoding="utf-8")
    return path


def load_baseline(path: Path = DRIFT_BASELINE_FILE) -> dict | None:
    """基线缺失/损坏 -> None(调用方降级, 不 crash)."""
    path = Path(path)
    try:
        b = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return b if isinstance(b, dict) and isinstance(b.get("features"), dict) else None


def append_alert(kind: str, level: str, path: Path = ALERTS_LOG, **detail) -> dict:
    """运维告警/审计一行 jsonl(7.1 漂移 + 7.2 重训/回滚共用); IO 失败静默."""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": kind, "level": level, **detail,
    }
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return record


def load_alerts(path: Path = ALERTS_LOG) -> list[dict]:
    """读取告警文件(坏行/空行静默跳过, 与 track.load_predictions 同口径)."""
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# 漂移检查编排
# ---------------------------------------------------------------------------

def collect_recent_features(
    symbols: list[str], window: int, settings=None
) -> tuple[pd.DataFrame, list[str]]:
    """票池各股近 window 日特征帧(不 dropna, NaN 本身是监控对象); 单票失败跳过.

    返回 (近端矩阵[含 symbol 列], 跳过的票列表)。缓存优先, 过期缓存会触网重拉
    (与 predict 同口径, 建议接在 daily.sh 之后跑)。
    """
    from stock_forecast.data import collector
    from stock_forecast.features import build as fb

    settings = settings or get_settings()
    frames, skipped = [], []
    for sym in symbols:
        try:
            data = collector.fetch_all(sym)
            kline = data.get("kline")
            if kline is None or kline.empty:
                skipped.append(sym)
                continue
            X, _y, _opt = fb.build_feature_matrix(
                kline, data["index"], data.get("fund_flow"),
                settings.label_threshold, extras=fb.make_extras(data),
            )
            frames.append(X.tail(window).assign(symbol=sym))
        except Exception:  # noqa: BLE001 - 单票失败不阻塞整体(不变量 5)
            skipped.append(sym)
    if not frames:
        return pd.DataFrame(), skipped
    return pd.concat(frames).sort_index(), skipped


def run_drift_check(
    symbols: list[str] | None = None,
    window: int | None = None,
    settings=None,
    baseline_path: Path = DRIFT_BASELINE_FILE,
) -> dict:
    """漂移检查主流程. 返回结构化结果(不写告警文件, 由 cmd 层决定落盘).

    result: {ok, error?, baseline_missing?, as_of, n_symbols, n_samples,
             skipped_symbols, results: [{feature, psi, level, nan_expected,
             nan_actual, degenerate}], missing_features, alerts: [...]}
    """
    settings = settings or get_settings()
    window = window or settings.drift_window
    symbols = symbols or list(settings.stock_pool)

    baseline = load_baseline(baseline_path)
    if baseline is None:
        return {"ok": False, "baseline_missing": True,
                "error": "未找到漂移基线(需 Phase 7 之后训练的模型). "
                         "请先运行: backtest --pool 或 retrain"}
    if baseline.get("baseline_version") != BASELINE_VERSION:
        # 版本闸门: 旧格式基线整体拒绝, 避免"全部特征评估失败 -> 假绿"
        return {"ok": False, "baseline_missing": True,
                "error": f"漂移基线版本不兼容(文件 v{baseline.get('baseline_version')}, "
                         f"当前 v{BASELINE_VERSION}). 请重跑 backtest --pool 或 retrain 重建基线"}

    recent, skipped = collect_recent_features(symbols, window, settings)
    if recent.empty or len(recent) < settings.drift_min_samples:
        return {"ok": False, "baseline_missing": False,
                "error": f"近端特征样本不足({len(recent)} 行 < {settings.drift_min_samples}), "
                         "无法做漂移检查(缓存过旧或数据源不可用?)",
                "skipped_symbols": skipped}

    as_of = str(pd.DatetimeIndex(recent.index).max().date())
    results: list[dict] = []
    missing_features: list[str] = []
    base_feats = baseline.get("features", {})
    for name, spec in base_feats.items():
        if name not in recent.columns:
            missing_features.append(name)
            continue
        row = evaluate_feature(
            name, recent[name], spec,
            settings.drift_psi_warn, settings.drift_psi_alert,
            settings.drift_nan_delta,
        )
        if row is not None:
            results.append(row)
    # PSI 降序, psi=None(近端全 NaN, 由缺失率检查兜底)排最后
    results.sort(key=lambda r: (r["psi"] is None, -(r["psi"] or 0.0)))

    if not results and not missing_features:
        return {"ok": False, "baseline_missing": False,
                "error": "基线特征全部无法评估(基线与近端矩阵结构不一致?), "
                         "拒绝给出'分布稳定'结论",
                "skipped_symbols": skipped}

    alerts: list[dict] = [
        {"kind": "feature_drift", **r} for r in results if r["level"] != LEVEL_OK
    ]
    for name in missing_features:
        alerts.append({"kind": "feature_missing", "feature": name, "psi": None,
                       "level": LEVEL_ALERT,
                       "reason": "基线特征在近端矩阵缺失(schema 漂移/缓存过旧)"})
    return {
        "ok": True, "baseline_missing": False, "as_of": as_of,
        "n_symbols": recent["symbol"].nunique() if "symbol" in recent else 0,
        "n_samples": int(len(recent)), "window": int(window),
        "n_features_evaluated": len(results),
        "n_features_baseline": len(base_feats),
        "n_samples_baseline": baseline.get("n_samples"),
        "skipped_symbols": skipped, "results": results,
        "missing_features": missing_features, "alerts": alerts,
        "has_alert_level": any(
            a["level"] == LEVEL_ALERT for a in alerts
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_drift(args: argparse.Namespace) -> int:
    """drift 子命令: 检查 -> 表格报告 -> warn/alert 落 alerts.csvl.

    退出码: 0=正常(含仅 warn); **2=存在 alert 级漂移**(供 maintain.sh 触发重训);
    1=错误(无基线以外的失败, 如样本不足)。无基线为降级场景, 提示后返回 0。
    """
    settings = get_settings()
    symbols = list(args.symbols) if args.symbols else None
    result = run_drift_check(symbols=symbols, window=args.window, settings=settings)

    if not result["ok"]:
        if result.get("baseline_missing"):
            console.print(f"[yellow]{result['error']}[/yellow]")
            return 0
        console.print(f"[red]{result['error']}[/red]")
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        _render(result, settings)

    # warn/alert 落盘(每特征一行, 便于事后按时间/特征聚合复盘)
    for a in result["alerts"]:
        detail = {k: v for k, v in a.items() if k not in ("kind", "level")}
        append_alert(a["kind"], a["level"], window=result["window"],
                     as_of=result["as_of"], **detail)
    # 退出码 2 = alert 级(约定, 非错误)
    return 2 if result["has_alert_level"] else 0


def _render(result: dict, settings) -> None:
    console.print(
        f"漂移检查(近 {result['window']} 日 × {result['n_symbols']} 只, "
        f"{result['n_samples']} 样本, 截至 {result['as_of']}); "
        f"基线样本 {result.get('n_samples_baseline') or '-'}, "
        f"特征评估 {result.get('n_features_evaluated')}/{result.get('n_features_baseline')}"
    )
    if result["skipped_symbols"]:
        console.print(f"[yellow]数据不可用跳过的票: {result['skipped_symbols']}[/yellow]")

    t = Table(title=f"特征 PSI (warn >= {settings.drift_psi_warn}, "
                    f"alert >= {settings.drift_psi_alert}, "
                    f"缺失率跳升 >= {settings.drift_nan_delta} 直接 alert)")
    t.add_column("特征")
    t.add_column("PSI", justify="right")
    t.add_column("级别")
    t.add_column("NaN 期望→近端", justify="right")
    shown = result["results"][:20]  # 终端只展示 PSI 最高的前 20 个
    for r in shown:
        color = {"ok": "green", "warn": "yellow", "alert": "red"}[r["level"]]
        psi_txt = "-" if r["psi"] is None else f"{r['psi']:.4f}"
        nan_txt = f"{r['nan_expected']:.1%} → {r['nan_actual']:.1%}"
        if r["nan_jump"]:
            nan_txt += " [red]⚠跳升[/red]"
        if r["degenerate"]:
            nan_txt += " [dim](退化列)[/dim]"
        t.add_row(r["feature"], psi_txt, f"[{color}]{r['level']}[/{color}]", nan_txt)
    console.print(t)
    if len(result["results"]) > len(shown):
        console.print(f"[dim]其余 {len(result['results']) - len(shown)} 个特征 PSI 均在展示行之下.[/dim]")
    for name in result["missing_features"]:
        console.print(f"[red]✗ 基线特征缺失: {name}[/red]")

    n_alert = sum(1 for a in result["alerts"] if a["level"] == LEVEL_ALERT)
    n_warn = sum(1 for a in result["alerts"] if a["level"] == LEVEL_WARN)
    if result["alerts"]:
        console.print(
            f"[yellow]告警 {len(result['alerts'])} 条(alert {n_alert} / warn {n_warn}), "
            f"已写入 {ALERTS_LOG}[/yellow]"
        )
        if n_alert:
            console.print("[red]存在 alert 级漂移: 建议重训(maintain.sh 会依退出码 2 自动触发).[/red]")
    else:
        console.print("[green]全部特征 PSI 低于告警阈值, 分布稳定.[/green]")
