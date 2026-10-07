"""缓存工具单测: saved_within(按文件 mtime 的新鲜度, Phase 2 共享/滞后源用)."""

import pandas as pd

from stock_forecast.data import cache


def test_saved_within(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_CACHE_DIR", tmp_path)
    # 无缓存 -> False
    assert not cache.saved_within("k", days=1)
    # parquet 写入后 -> True
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "x": [1.0]})
    cache.save("k", df)
    assert cache.saved_within("k", days=1)
    assert cache.saved_within("k", days=0) is False  # 已过去若干毫秒
    # pickle(save_json) 同样适用
    cache.save_json("j", {"a": 1})
    assert cache.saved_within("j", days=1)
    # 不存在的 key
    assert not cache.saved_within("nope", days=365)
