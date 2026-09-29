"""M0 骨架冒烟测试: CLI 可解析子命令、配置可加载."""

import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "stock_forecast", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for cmd in ("fetch", "features", "backtest", "news", "predict"):
        assert cmd in result.stdout


def test_settings_defaults():
    from stock_forecast.config import get_settings

    s = get_settings()
    assert len(s.stock_pool) == 20
    assert s.ml_weight + s.llm_weight == 1.0
    assert 0 < s.label_threshold < 0.05
