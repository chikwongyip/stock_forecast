# 股票当日走势预测系统 (MVP)

基于 `sys_design.md` 的务实裁剪实现: A股日频走势预测 CLI 工具。
**技术面 ML 模型线(LightGBM) + LLM 舆情研判线(DeepSeek)双线并行, 融合输出。**

> ⚠️ 预期管理: 日频方向预测准确率 **55%~62% 已属优秀水平**, 回测 >70% 大概率是过拟合或数据泄露。
> 本系统的核心价值在于 **不确定性量化 + 可解释性 + 风险识别**, 不构成任何投资建议。

## 环境

全部在 Docker 中运行, 本机无需安装 Python:

```bash
cp .env.example .env   # 填入 DEEPSEEK_API_KEY (可选, 不填则 LLM 线自动降级)
docker compose build
```

## 使用

```bash
# 1. 拉取并缓存数据 (K线/上证指数/资金流)
docker compose run --rm sf fetch 600519

# 2. 查看特征矩阵与标签分布
docker compose run --rm sf features 600519

# 3. 票池 walk-forward 回测 + 训练最终模型 (首次约需数分钟, 会拉取 20 只票池股票数据)
docker compose run --rm sf backtest --pool

# 4. LLM 舆情线独立调试
docker compose run --rm sf news 600519

# 5. 端到端预测 (需先完成 3)
docker compose run --rm sf predict 600519
docker compose run --rm sf predict 600519 --no-llm   # 纯技术面
docker compose run --rm sf predict 600519 --json     # 机器可读输出

# 测试
docker compose run --rm sf pytest -q
```

## 架构(对应 sys_design.md 8 层的 MVP 裁剪)

```
数据 akshare ──> data/(采集+parquet缓存) ──> features/(指标+对齐+shift防前视+标签)
                                                │
                        model/(LightGBM三分类 + walk-forward回测)   ←─ 技术面线
                        llm/(新闻采集 + DeepSeek结构化情感打分)     ←─ 舆情线
                                                │
                        fusion/(加权融合 0.7/0.3 + 冲突降级 + 极端行情风控 + 免责声明)
                                                │
                        output/(rich 终端报告 / --json)  →  data/predictions.csvl (L8埋点)
```

关键防前视约定: 特征行 X[T] 只含 ≤T-1 收盘后数据(全部特征统一 shift(1));
标签 y[T] = 第 T 日 close-to-close 方向, ±0.5% 内记"平"。
`tests/test_labeling.py` 中有硬断言: 篡改 T 日及之后的原始数据, X[T] 必须一字不变。

## 目录

- `src/stock_forecast/` — 源码(见上方架构)
- `tests/` — 离线单测(不依赖网络/LLM)
- `data/` — 运行时缓存与模型(gitignore)
- `sys_design.md` — 原始设计文档

## 已知边界

- akshare 接口偶发变动: 所有调用收敛在 `data/akshare_client.py`, 修复只改这一处
- 盘中运行时 akshare 日K含当日未收盘行, 最后一行特征含义有偏差; 建议收盘后或开盘前运行
- 资金流数据仅覆盖近 ~120 交易日, 更早样本该组特征为 NaN(LightGBM 原生处理)
- MVP 未做: 市场状态识别(HMM)、多模型集成(LSTM/XGBoost)、Web 界面、监控闭环(见方案"后续路线")
