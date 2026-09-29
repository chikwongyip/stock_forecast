"""全局配置: 通过 pydantic-settings 读取 .env 与默认值."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目数据根目录: 容器内 /app/data, 本地开发时 ./data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_LOG = DATA_DIR / "predictions.csvl"


class Settings(BaseSettings):
    """运行时配置, 环境变量优先于默认值."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    # None = 有 key 即启用; "true"/"false" 显式覆盖
    llm_enabled: str = ""

    # 标签定义: 涨跌幅绝对值超过该阈值才计入涨/跌, 否则记"平"
    label_threshold: float = 0.005

    # 融合权重: 技术面 ML 线 vs LLM 舆情线
    ml_weight: float = 0.7
    llm_weight: float = 0.3

    # 方向判定阈值
    direction_threshold: float = 0.15

    # 极端行情判定: 近 5 日累计涨跌幅超过该值时降级提示
    extreme_move_5d: float = 0.15

    # 回测: walk-forward 初始训练窗比例与每轮预测步长(交易日)
    train_ratio: float = 0.6
    test_window: int = 20

    # 默认票池: 跨行业代表股, 用于训练横截面通用模型
    stock_pool: list[str] = [
        "600519",  # 贵州茅台
        "000858",  # 五粮液
        "601318",  # 中国平安
        "600036",  # 招商银行
        "000001",  # 平安银行
        "600030",  # 中信证券
        "300750",  # 宁德时代
        "002594",  # 比亚迪
        "600900",  # 长江电力
        "601890",  # 万华化学
        "000333",  # 美的集团
        "600276",  # 恒瑞医药
        "601888",  # 中国中免
        "002415",  # 海康威视
        "600887",  # 伊利股份
        "601088",  # 中国神华
        "000002",  # 万科A
        "600028",  # 中国石化
        "601633",  # 长城汽车
        "300059",  # 东方财富
    ]


def get_settings() -> Settings:
    return Settings()


def llm_available(settings: Settings) -> bool:
    """LLM 线是否可用: 显式开关优先, 其次看是否有 key."""
    if settings.llm_enabled.lower() in ("false", "0", "no"):
        return False
    if settings.llm_enabled.lower() in ("true", "1", "yes"):
        return True
    return bool(settings.deepseek_api_key)


DISCLAIMER = (
    "本输出由统计/机器学习模型与 LLM 辅助研判生成, 仅供研究参考, "
    "不构成任何投资建议。日频方向预测准确率上限约 55%~62%, "
    "历史表现不代表未来收益, 据此操作风险自负。"
)
