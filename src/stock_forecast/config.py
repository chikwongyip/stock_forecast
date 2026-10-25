"""全局配置: 通过 pydantic-settings 读取 .env 与默认值."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目数据根目录: 容器内 /app/data, 本地开发时 ./data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_LOG = DATA_DIR / "predictions.csvl"
COMPLIANCE_LOG = DATA_DIR / "compliance_log.jsonl"  # L6 合规违规拦截记录 (Phase 5.2)
# track 摘要缓存, 供 predict/Web 展示 (Phase 6.1)
TRACK_SUMMARY_FILE = DATA_DIR / "track_summary.json"
ALERTS_LOG = DATA_DIR / "alerts.csvl"  # L8 运维告警(漂移/重训/回滚审计, Phase 7.1/7.2)
FEEDBACK_LOG = DATA_DIR / "feedback.csvl"  # 用户反馈 (Phase 7.4)
DRIFT_BASELINE_FILE = MODELS_DIR / \
    "drift_baseline.json"  # PSI 漂移基线(随模型代际归档, Phase 7.1)
MODELS_ARCHIVE_DIR = MODELS_DIR / "archive"  # 重训替换前的旧模型归档(可回滚, Phase 7.2)


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

    # L4 市场状态识别(Phase 3 规则引擎): 上证指数均线排列 + 波动率分位
    regime_ma_fast: int = 20             # 快线窗口(交易日)
    regime_ma_slow: int = 60             # 慢线窗口
    regime_vol_window: int = 250         # 20日波动率分位的回看窗口(交易日)
    regime_vol_min_periods: int = 120    # 分位窗口最少样本(不足则不触发高波动覆盖)
    regime_high_vol_q: float = 0.85      # 波动率分位 ≥ 该值: 多头排列也按震荡处理
    # 融合层按状态调整(Phase 3 / 3.2)
    regime_range_threshold_mult: float = 1.5  # 震荡市: 方向阈值 × 该系数(收紧 -> 更多观望)
    regime_bear_downgrade_up: bool = True     # 熊市: "偏涨"研判置信度下调一档

    # 多模型集成(Phase 4 / 4.5): 成员概率全变差(TV)距离 >= 该值视为分歧大,
    # 置信度下调一档; 单模型运行时不生效
    ensemble_disagree_threshold: float = 0.10

    # L6 风控: 交易状态识别(Phase 5 / 5.1)
    suspend_gap_lookback: int = 20     # 历史停牌断档回看窗口(交易日)
    suspend_gap_threshold: int = 2     # 窗口内断档 >= 该天数 -> 提示"近期有停牌"
    limit_move_tol: float = 0.004      # 涨跌停判定容差(前复权收益 vs 板别限幅)

    # L6 风控: 极端场景扩充(Phase 5 / 5.3)
    extreme_yizi_streak: int = 2       # 连续一字板天数 >= 该值触发提示
    extreme_amp_window: int = 20       # 振幅异常窗口(交易日)
    extreme_amplitude: float = 0.40    # (窗口最高-最低)/中位收盘 >= 该值 -> 振幅异常
    volume_dry_window: int = 60        # 量能枯竭分位回看窗口(交易日)
    volume_dry_quantile: float = 0.05  # 当日量在窗口内分位 <= 该值 -> 量能枯竭
    volume_dry_min_samples: int = 20   # 分位计算最少历史样本

    # L6 合规(Phase 5 / 5.2): 输出前固定校验, 违规阻断输出并记日志
    compliance_block: bool = True      # False = 只告警记录不阻断(应急开关)

    # L7 输出(Phase 6 / 6.1): 预测分布区间的情景条件幅度回看窗口(交易日)
    scenario_lookback: int = 120

    # L8 运维(Phase 7 / 7.1): 特征漂移 PSI 监控
    drift_window: int = 20          # 近端特征分布对比窗口(交易日)
    drift_bins: int = 10            # PSI 分箱数(训练期分位数分箱)
    drift_psi_warn: float = 0.10    # PSI >= 该值 -> warn(记 alerts.csvl)
    drift_psi_alert: float = 0.25   # PSI >= 该值 -> alert(触发重训, drift 退出码 2)
    drift_nan_delta: float = 0.20   # 缺失率较训练期上升 >= 该值 -> alert(覆盖恶化)
    drift_min_samples: int = 20     # 近端窗口全票池合计最少样本数

    # L8 运维(Phase 7 / 7.2): 自动重训
    # 新模型回测方向准确率 >= 旧模型 - 容差 才替换(默认 0 = 不劣于即替换)
    retrain_tolerance: float = 0.0

    # 回测: walk-forward 初始训练窗比例与每轮预测步长(交易日)
    train_ratio: float = 0.6
    test_window: int = 20

    # 默认票池: 跨行业代表股, 用于训练横截面通用模型
    stock_pool: list[str] = [
        "600183"
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
