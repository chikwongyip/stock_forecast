"""DeepSeek 情感打分: 新闻批量输入 -> 结构化 JSON 输出.

成本控制: 单次批量调用(而非逐条), 结果按 (symbol, date) 当日缓存.
失败降级: 无 key / 网络失败 / JSON 非法 -> 返回 available=False,
上层(融合)按"LLM 线不可用"处理, 不 crash.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from rich.console import Console

from stock_forecast.config import get_settings, llm_available
from stock_forecast.data import cache
from stock_forecast.llm.news import fetch_news, format_for_prompt

console = Console()

SCORE_FIELDS = ("policy_score", "industry_score", "sentiment_score")

SYSTEM_PROMPT = (
    "你是一名客观中立的金融舆情分析师, 只做信息分析, 严禁给出投资建议或买卖倾向."
)

USER_PROMPT_TEMPLATE = """请分析以下关于股票 {symbol} {stock_name} 的近期新闻, 输出严格的 JSON.

要求:
1. 三个维度各打 -5 ~ +5 的整数分(0=中性, 正=利好, 负=利空):
   - policy_score: 政策面影响(利好政策为正, 利空政策为负)
   - industry_score: 行业景气度(行业向好为正, 恶化/竞争加剧为负)
   - sentiment_score: 整体舆情倾向(正面报道多为正, 负面为负)
2. key_positive / key_negative: 最多各 3 条有利/不利因素, 每条必须能追溯到具体新闻(注明来源序号, 如 "[3] xxx").
3. 与该股明显无关的新闻直接忽略, 不计入任何分数.
4. 若新闻数量少或信息不足, 分数应靠近 0, 并在 confidence 中体现低置信.
5. confidence: 0~1, 表示你对本次打分依据充分程度的自评.
6. reasoning: 一句话说明打分依据.

只输出 JSON, 不要任何其他文字, 字段如下:
{{"policy_score": int, "industry_score": int, "sentiment_score": int,
 "key_positive": [str], "key_negative": [str], "confidence": float, "reasoning": str}}

新闻列表:
{news_block}
"""


class LLMUnavailableError(RuntimeError):
    pass


def _call_deepseek(settings, messages: list[dict], retries: int = 2) -> str:
    """调用 DeepSeek(OpenAI 兼容 SDK); 返回原始文本."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=settings.deepseek_model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise LLMUnavailableError(f"DeepSeek 调用失败: {last_err}")


def _validate(obj: dict) -> dict:
    """字段校验; 非法则抛 ValueError 触发重试/降级."""
    for k in (*SCORE_FIELDS, "key_positive", "key_negative", "confidence", "reasoning"):
        if k not in obj:
            raise ValueError(f"缺少字段 {k}")
    for k in SCORE_FIELDS:
        v = float(obj[k])
        if not -5 <= v <= 5:
            raise ValueError(f"{k}={v} 超出 [-5,5]")
        obj[k] = int(round(v))
    for k in ("key_positive", "key_negative"):
        if not isinstance(obj[k], list):
            raise ValueError(f"{k} 必须是列表")
        obj[k] = [str(x)[:200] for x in obj[k][:3]]
    conf = float(obj["confidence"])
    obj["confidence"] = max(0.0, min(1.0, conf))
    obj["reasoning"] = str(obj["reasoning"])[:300]
    return obj


def analyze_sentiment(symbol: str, stock_name: str = "", force: bool = False) -> dict:
    """返回结构化打分; LLM 线不可用时 available=False, 分数全 0."""
    result = {
        "available": False, "policy_score": 0, "industry_score": 0,
        "sentiment_score": 0, "key_positive": [], "key_negative": [],
        "confidence": 0.0, "reasoning": "", "n_news": 0, "model": "",
    }
    settings = get_settings()
    if not llm_available(settings):
        result["reasoning"] = "LLM 线未启用(缺少 DEEPSEEK_API_KEY 或 LLM_ENABLED=false)"
        return result

    today = datetime.now().date().isoformat()
    cache_key = f"sentiment_{symbol}_{today}"
    if not force:
        cached = cache.load_json(cache_key)
        if isinstance(cached, dict):
            return cached

    news = fetch_news(symbol)
    result["n_news"] = len(news)
    if not news:
        result["reasoning"] = "近 5 天无相关新闻, LLM 线不输出打分"
        return result

    news_block = format_for_prompt(symbol, stock_name or symbol, news)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            symbol=symbol, stock_name=stock_name or symbol, news_block=news_block,
        )},
    ]

    try:
        raw = _call_deepseek(settings, messages)
        obj = _validate(json.loads(raw))
    except (LLMUnavailableError, ValueError, json.JSONDecodeError) as e:
        console.print(f"[yellow]LLM 打分失败, 降级: {e}[/yellow]")
        result["reasoning"] = f"LLM 打分失败: {e}"
        return result

    result.update(obj)
    result["available"] = True
    result["model"] = settings.deepseek_model
    cache.save_json(cache_key, result)
    return result


def cmd_news(args: argparse.Namespace) -> int:
    result = analyze_sentiment(args.symbol, force=getattr(args, "refresh", False))
    if not result["available"]:
        console.print(f"[yellow]LLM 线不可用: {result['reasoning']}[/yellow]")
        return 0
    console.print(f"[bold]{args.symbol} LLM 舆情打分[/bold] ({result['model']}, {result['n_news']} 条新闻)")
    console.print(f"政策面: {result['policy_score']:+d}  行业: {result['industry_score']:+d}  "
                  f"舆情: {result['sentiment_score']:+d}  置信: {result['confidence']:.2f}")
    if result["key_positive"]:
        console.print("[green]有利因素:[/green]")
        for x in result["key_positive"]:
            console.print(f"  + {x}")
    if result["key_negative"]:
        console.print("[red]不利因素:[/red]")
        for x in result["key_negative"]:
            console.print(f"  - {x}")
    console.print(f"[dim]{result['reasoning']}[/dim]")
    return 0
