"""CLI 入口: fetch / features / backtest / news / predict / regime / track
/ drift / retrain / rollback / feedback."""

import argparse


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
    p_fetch.add_argument("--check", action="store_true",
                         help="只做各数据源缓存健康检查, 不发起拉取")

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
    p_pred.add_argument("--mode", choices=("simple", "pro"), default="pro",
                        help="simple=新手模式(方向+置信度+白话提示, 零术语); pro=全量报告(默认)")

    # regime: 市场状态识别 (L4, Phase 3)
    p_regime = sub.add_parser("regime", help="识别当前市场状态(牛/熊/震荡, 规则引擎)")
    p_regime.add_argument("--json", action="store_true", help="输出 JSON 而非终端报表")

    # track: 结算预测埋点, 输出实盘准确率
    p_track = sub.add_parser("track", help="结算预测埋点, 输出实盘准确率")
    p_track.add_argument("--refresh", action="store_true", help="结算时强制重拉K线(忽略缓存)")
    p_track.add_argument("--json", action="store_true", help="输出 JSON 而非终端报表")

    # drift: 特征漂移 PSI 监控 (Phase 7.1); 退出码 2 = 存在 alert 级漂移
    p_drift = sub.add_parser(
        "drift", help="特征漂移检查(近端 vs 训练期 PSI, 告警写 data/alerts.csvl)")
    p_drift.add_argument("symbols", nargs="*", default=None,
                         help="可选: 只查指定票(默认全票池)")
    p_drift.add_argument("--window", type=int, default=None,
                         help="近端对比窗口(交易日, 默认 drift_window=20)")
    p_drift.add_argument("--json", action="store_true", help="输出 JSON 而非终端报表")

    # retrain: 自动重训(回测不劣于现役才替换, 旧代际归档可回滚) (Phase 7.2)
    p_retrain = sub.add_parser("retrain", help="重训票池模型, 不劣于现役才替换(旧模型归档)")
    p_retrain.add_argument("--dry-run", action="store_true",
                           help="演练: 只回测与判定, 不训练最终模型不写工件")
    p_retrain.add_argument("--force", action="store_true",
                           help="跳过对比强制替换(仍归档旧模型)")
    p_retrain.add_argument("--tolerance", type=float, default=None,
                           help="方向准确率容差(默认 retrain_tolerance=0, 即不劣于即替换)")

    # rollback: 恢复最近一次归档的模型代际 (Phase 7.2)
    p_rb = sub.add_parser("rollback", help="回滚到最近一次归档的模型(含漂移基线)")
    p_rb.add_argument("--list", action="store_true", help="只列出归档, 不执行回滚")

    # feedback: 用户反馈收集 (Phase 7.4)
    p_fb = sub.add_parser("feedback", help="记录预测反馈(落 data/feedback.csvl)或查看统计")
    p_fb.add_argument("symbol", nargs="?", default=None, help="被反馈的股票代码")
    p_fb.add_argument("--rating", choices=("helpful", "wrong", "other"), default=None,
                      help="有帮助/有误/其他")
    p_fb.add_argument("--note", default="", help="备注(负反馈建议说明原因, 月度复盘用)")
    p_fb.add_argument("--date", default=None, help="被反馈的预测日期(YYYY-MM-DD, 可选)")
    p_fb.add_argument("--list", action="store_true", help="查看反馈统计(月度复盘)")

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
    if args.command == "regime":
        from stock_forecast.features.regime import cmd_regime

        return cmd_regime(args)
    if args.command == "track":
        from stock_forecast.monitor.track import cmd_track

        return cmd_track(args)
    if args.command == "drift":
        from stock_forecast.monitor.drift import cmd_drift

        return cmd_drift(args)
    if args.command == "retrain":
        from stock_forecast.model.retrain import cmd_retrain

        return cmd_retrain(args)
    if args.command == "rollback":
        from stock_forecast.model.retrain import cmd_rollback

        return cmd_rollback(args)
    if args.command == "feedback":
        from stock_forecast.monitor.feedback import cmd_feedback

        return cmd_feedback(args)
    parser.print_help()
    return 1


def run_predict(args: argparse.Namespace) -> int:
    """predict 渲染层: 编排全部委托 pipeline.predict_once(CLI 与 Web 共用, Phase 6)."""
    import json

    from rich.console import Console

    from stock_forecast import pipeline
    from stock_forecast.output import render

    console = Console()
    result = pipeline.predict_once(args.symbol, use_llm=not args.no_llm)

    for note in result.get("notices", []):
        console.print(f"[yellow]{note}[/yellow]")
    if not result["ok"]:
        console.print(f"[red]{result['error']}[/red]")
        return 1

    # 合规阻断(5.2): 不渲染研判, 不写埋点(pipeline 内已记 compliance_log)
    if result["blocked"]:
        if args.json:
            print(render.build_blocked_json(result["blocked_payload"]))
        else:
            render.render_blocked(result["blocked_payload"])
        return 0

    # 新手模式(6.2): 方向+置信度+一句话理由+白话风险提示, 零术语
    if args.mode == "simple":
        if args.json:
            print(json.dumps(result["simple"], ensure_ascii=False, indent=2))
        else:
            render.render_report_simple(result["simple"])
        return 0

    if args.json:
        print(pipeline.report_json_text(result))
    else:
        render.render_report(
            result["symbol"], result["date"], result["fused"],
            result["sentiment"], result["narrative"],
            regime_info=result["regime_info"],
            scenarios=result["scenarios"],
            track_summary=result["track_summary"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
