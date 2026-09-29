"""CLI 入口: 子命令 fetch / features / backtest / news / predict."""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock_forecast",
        description="A股当日走势预测系统 (MVP): 技术面 ML + LLM 舆情双线融合",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch: 拉取并缓存数据
    p_fetch = sub.add_parser("fetch", help="拉取股票数据并写入本地缓存")
    p_fetch.add_argument("symbol", help="A股代码, 如 600519")
    p_fetch.add_argument("--refresh", action="store_true", help="忽略缓存强制重新拉取")

    # features: 构建特征矩阵
    p_feat = sub.add_parser("features", help="构建特征矩阵与标签")
    p_feat.add_argument("symbol", help="A股代码")

    # backtest: walk-forward 回测
    p_bt = sub.add_parser("backtest", help="walk-forward 回测")
    p_bt.add_argument("symbol", nargs="?", default=None, help="可选: 单票视图")
    p_bt.add_argument("--pool", action="store_true", help="对默认票池整体回测")

    # news: LLM 舆情线独立调试
    p_news = sub.add_parser("news", help="采集新闻并输出 LLM 情感打分")
    p_news.add_argument("symbol", help="A股代码")

    # predict: 端到端预测
    p_pred = sub.add_parser("predict", help="预测当日走势 (端到端)")
    p_pred.add_argument("symbol", help="A股代码")
    p_pred.add_argument("--no-llm", action="store_true", help="跳过 LLM 舆情线")
    p_pred.add_argument("--json", action="store_true", help="输出 JSON 而非终端报告")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch":
        from stock_forecast.data.collector import cmd_fetch

        return cmd_fetch(args)
    if args.command == "features":
        from stock_forecast.features.build import cmd_features

        return cmd_features(args)
    if args.command == "backtest":
        from stock_forecast.model.backtest import cmd_backtest

        return cmd_backtest(args)
    if args.command == "news":
        from stock_forecast.llm.sentiment import cmd_news

        return cmd_news(args)
    if args.command == "predict":
        from stock_forecast.cli import run_predict

        return run_predict(args)
    parser.print_help()
    return 1


def run_predict(args: argparse.Namespace) -> int:
    """predict 主编排: fetch -> features -> ML 推理 -> LLM 舆情 -> 融合 -> 渲染."""
    import json
    from datetime import datetime

    from rich.console import Console

    from stock_forecast.config import DATA_DIR, PREDICTIONS_LOG, get_settings
    from stock_forecast.data import collector
    from stock_forecast.features import build as fb
    from stock_forecast.fusion import combine
    from stock_forecast.llm import report as rpt
    from stock_forecast.llm.sentiment import analyze_sentiment
    from stock_forecast.model import train
    from stock_forecast.output import render

    console = Console()
    settings = get_settings()

    booster = train.load_booster()
    if booster is None:
        console.print(
            "[red]未找到已训练模型. 请先运行: docker compose run --rm sf backtest --pool[/red]"
        )
        return 1

    data = collector.fetch_all(args.symbol)
    kline = data.get("kline")
    if kline is None or kline.empty:
        console.print(f"[red]{args.symbol} K线数据不可用[/red]")
        return 1

    X, _y, _optional = fb.build_feature_matrix(
        kline, data["index"], data["fund_flow"], settings.label_threshold
    )
    X_latest = X.iloc[[-1]]
    feats = booster.feature_name()
    missing = [c for c in feats if c not in X_latest.columns]
    if missing:
        console.print(f"[red]特征缺失(缓存过旧?): {missing[:5]} ...[/red]")
        return 1
    probs = train.predict_proba_df(booster, X_latest[feats]).iloc[0].to_dict()

    sentiment = (
        {"available": False, "reasoning": "--no-llm: 已跳过 LLM 舆情线"}
        if args.no_llm
        else analyze_sentiment(args.symbol)
    )

    fused = combine.fuse(probs, sentiment, kline_tail=kline.tail(6), settings=settings)

    gains = booster.feature_importance(importance_type="gain")
    top_names = [n for n, _ in sorted(zip(feats, gains), key=lambda t: -t[1])[:5]]

    date_str = str(X.index[-1].date())
    narrative = rpt.build_narrative(
        fused["direction"], fused["confidence"],
        rpt.ml_narrative(probs["p_up"], probs["p_flat"], probs["p_down"], top_names),
        rpt.llm_narrative(sentiment),
        fused["warnings"],
    )

    if args.json:
        print(render.build_json_report(args.symbol, date_str, fused, sentiment, narrative))
    else:
        render.render_report(args.symbol, date_str, fused, sentiment, narrative)

    # L8 埋点: 预测落盘, 供后续准确率追踪
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "symbol": args.symbol, "date": date_str,
            "direction": fused["direction"], "confidence": fused["confidence"],
            **{k: v for k, v in fused["probs"].items()},
            "s_ml": fused["s_ml"], "s_llm": fused["s_llm"], "s_final": fused["s_final"],
            "llm_available": fused["llm_available"],
        }, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
