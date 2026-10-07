"""M0 骨架冒烟测试: CLI 可解析子命令、配置可加载、退出码透传."""

import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "stock_forecast", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for cmd in ("fetch", "features", "backtest", "news", "predict", "regime", "track",
                "drift", "retrain", "rollback", "feedback"):
        assert cmd in result.stdout


def test_module_exit_code_propagation():
    """退出码必须透传(Phase 7 修复的历史 bug): 错误路径非零退出.

    此前 __main__.py 只调 main() 丢弃返回值, 一切错误都以 0 退出,
    daily.sh/maintain.sh 的 `|| fail=1` 与 drift 退出码 2 约定全部失效。
    predict 不存在的票: 无模型工件 -> "未找到已训练模型" 返回 1;
    有模型 -> 999999 非真实代码, 拉取失败/K线不可用同样返回 1。
    """
    result = subprocess.run(
        [sys.executable, "-m", "stock_forecast", "predict", "999999", "--no-llm"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 1, (result.stdout[-500:], result.stderr[-500:])


def test_settings_defaults():
    from stock_forecast.config import get_settings

    s = get_settings()
    assert len(s.stock_pool) == 20
    assert s.ml_weight + s.llm_weight == 1.0
    assert 0 < s.label_threshold < 0.05
