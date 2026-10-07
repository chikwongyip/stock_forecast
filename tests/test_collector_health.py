"""fetch --check 数据源健康检查单测: 注入 loader, 不读真实缓存/不触网."""

import pandas as pd

from stock_forecast.data import cache, collector


def _fresh_kline() -> pd.DataFrame:
    today = cache.latest_expected_trade_date()
    dates = pd.bdate_range(end=pd.Timestamp(today), periods=5)
    c = [10.0] * 5
    return pd.DataFrame({
        "date": dates, "open": c, "close": c, "high": c, "low": c, "volume": [1.0] * 5,
    })


def _fresh_fundflow() -> pd.DataFrame:
    today = cache.latest_expected_trade_date()
    dates = pd.bdate_range(end=pd.Timestamp(today), periods=5)
    return pd.DataFrame({"date": dates, "main_net_pct": [0.0] * 5})


def test_all_ok():
    k, ff = _fresh_kline(), _fresh_fundflow()
    store = {"kline_600519": k, "index_000001": k, "fundflow_600519": ff}
    report, expected = collector.source_health("600519", loader=store.get)
    assert [e["status"] for e in report] == ["ok", "ok", "ok"]
    assert all(e["rows"] == 5 for e in report)
    assert all(e["latest_date"] >= expected for e in report)


def test_missing_expired_incomplete():
    stale = _fresh_kline()
    stale["date"] = stale["date"] - pd.Timedelta(days=30)
    no_vol = _fresh_kline().drop(columns=["volume"])
    store = {"kline_600519": stale, "index_000001": no_vol}  # fundflow 无缓存
    report, _ = collector.source_health("600519", loader=store.get)
    by = {e["source"]: e for e in report}
    assert by["kline"]["status"] == "expired"
    assert by["kline"]["fresh"] is False
    assert by["index"]["status"] == "incomplete"
    assert by["index"]["missing_cols"] == ["volume"]
    assert by["fund_flow"]["status"] == "missing"
    assert by["fund_flow"]["rows"] == 0


def test_empty_dataframe_is_missing():
    empty = pd.DataFrame()
    store = {"kline_600519": empty, "index_000001": None, "fundflow_600519": None}
    report, _ = collector.source_health("600519", loader=store.get)
    assert all(e["status"] == "missing" for e in report)


def test_no_date_column_marks_expired_and_incomplete():
    """损坏缓存(无 date 列): 不能判定新鲜度 -> expired+incomplete, 不 crash."""
    broken = pd.DataFrame({"close": [1.0, 2.0]})
    store = {"kline_600519": broken, "index_000001": _fresh_kline(),
             "fundflow_600519": _fresh_fundflow()}
    report, _ = collector.source_health("600519", loader=store.get)
    kline = report[0]
    assert "expired" in kline["status"] and "incomplete" in kline["status"]
    assert "date" in kline["missing_cols"]
