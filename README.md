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
#    多模型对比: LightGBM / XGBoost / 等权集成 同分割回测 + p_up 相关矩阵, 落盘集成模型+元数据
docker compose run --rm sf backtest --pool

# 4. LLM 舆情线独立调试
docker compose run --rm sf news 600519

# 5. 市场状态识别(牛/熊/震荡, 规则引擎; predict 报告会自动带 T-1 口径状态)
docker compose run --rm sf regime
docker compose run --rm sf regime --json

# 6. 端到端预测 (需先完成 3)
#    自带 L6 风控: 停牌股输出"不适用", 涨跌停/ST 股"仅供参考"(置信度降级),
#    输出前过合规闸门(违规阻断并记 data/compliance_log.jsonl)
docker compose run --rm sf predict 600519
docker compose run --rm sf predict 600519 --no-llm   # 纯技术面
docker compose run --rm sf predict 600519 --json     # 机器可读输出
docker compose run --rm sf predict 600519 --mode simple   # 新手模式(零术语: 方向+把握程度+白话提示)

# 7. 结算预测埋点, 输出实盘准确率(滚动20/60日、置信度/市场状态/分标的分桶、ML线vs融合线)
#    同时落摘要缓存 data/track_summary.json, 供 predict/Web 报告展示历史准确率
docker compose run --rm sf track

# 8. Web 界面 v1(只读, 无账户体系): 当日预测(pro/新手模式, 结果可👍/👎反馈) + 历史准确率曲线
docker compose up -d web    # 浏览器打开 http://localhost:8501

# 9. 运维自动化(Phase 7): 特征漂移 PSI 监控 → 自动重训(不劣于才替换) → 归档回滚
docker compose run --rm sf drift                # 漂移检查(告警落 data/alerts.csvl, 退出码 2 = alert 级)
docker compose run --rm sf retrain --dry-run    # 重训演练(只对比不写); 去掉 --dry-run 才替换+归档旧代际
docker compose run --rm sf rollback             # 回滚到最近归档的模型代际(含漂移基线); --list 只列归档
docker compose run --rm sf feedback 600519 --rating wrong --note "方向判反"   # 反馈; feedback --list 复盘统计
docker compose run --rm --entrypoint bash sf /app/scripts/maintain.sh   # 运维例行(漂移+告警/月度触发重训, 建议周频 crontab)

# 数据源健康检查(纯本地缓存检查, 不发网络请求)
docker compose run --rm sf fetch --check 600519

# 每日例行(收盘后批量预测票池+结算, crontab 用法见脚本头注释)
docker compose run --rm --entrypoint bash sf /app/scripts/daily.sh

# 测试(镜像 entrypoint 是本模块, 跑 pytest 须覆盖 entrypoint)
docker compose run --rm --entrypoint pytest sf -q
docker compose run --rm --entrypoint ruff sf check src tests
```

## 架构(对应 sys_design.md 8 层的 MVP 裁剪)

```
数据 akshare ──> data/(采集+parquet缓存) ──> features/(指标+扩展源sources+对齐+shift防前视+标签)
   (K线/指数/资金流/行业板块/北向/两融/       │
    龙虎榜/增减持/解禁/外盘/宏观)      model/(LightGBM+XGBoost等权集成 + walk-forward回测) ←─ 技术面线
                                       llm/(新闻采集 + DeepSeek结构化情感打分)     ←─ 舆情线
                                       features/regime(MA排列+波动率分位 → 牛/熊/震荡) ←─ 市场状态 L4
                                       risk/(交易状态[停牌/涨跌停/ST] + 极端场景 + 合规校验) ←─ 风控合规 L6
                                                │
                        fusion/(加权融合 0.7/0.3 + 冲突降级 + 状态调整 + 集成分歧度降档 + 极端场景 + 交易状态降级 + 免责声明)
                                                │
                        risk/compliance(输出前合规闸门: 免责声明/无买卖措辞/置信度一致, 违规则阻断)
                                                │
                        output/(rich 报告 / --json / --mode simple 新手模式 / 情景分位区间)  →  data/predictions.csvl (L8埋点)
                                                │
                        monitor/track(埋点结算 → 实盘准确率报表 + 摘要缓存, 度量闭环)
                        monitor/drift(特征 PSI 漂移 → alerts.csvl) + model/retrain(不劣于才替换+归档/回滚)
                        monitor/feedback(用户反馈 CLI/Web → feedback.csvl)          ←─ 运维自动化 L8

入口: CLI(cli.py)与 Web(web/app.py, Streamlit :8501, 只读)共用 pipeline.predict_once
→ 同一个 report dict(Web 与 CLI --json 输出一致性由构造保证)
例行: daily.sh(每日预测+结算) / maintain.sh(周漂移检查, 告警或月度触发重训) / CI(ruff+pytest)
```

关键防前视约定: 特征行 X[T] 只含 ≤T-1 收盘后数据(全部特征统一 shift(1));
标签 y[T] = 第 T 日 close-to-close 方向, ±0.5% 内记"平"。
`tests/test_labeling.py` 中有硬断言: 篡改 T 日及之后的原始数据, X[T] 必须一字不变。
市场状态同理采用 **T-1 口径**(第 T 行状态含 T 日收盘信息, 预测 y[T] 只能用 T-1 行),
`tests/test_regime.py` 有对应硬断言。状态只作用于融合层(震荡市收紧方向阈值、
熊市下调偏涨置信), 不进模型特征, 旧模型无需重训。风控层(L6)同样只作用于融合/输出层:
交易状态(停牌→不适用、涨跌停/ST→仅供参考)与极端场景(连续一字板/振幅异常/量能枯竭)
对置信度做降级, 输出前经合规闸门(免责声明/无买卖措辞/置信度一致)校验, 违规即阻断。

## 目录

- `src/stock_forecast/` — 源码(见上方架构)
- `tests/` — 离线单测(不依赖网络/LLM)
- `data/` — 运行时缓存、模型(含 archive 归档)、埋点、告警与反馈(gitignore)
- `scripts/` — daily.sh(每日例行) / maintain.sh(漂移检查+自动重训例行)
- `.github/workflows/ci.yml` — CI(ruff + pytest, push/PR 触发)
- `docs/experiments.md` — 特征实验与回测对比记录
- `develop_plan.md` — 开发计划与进度日志(7 个 Phase 全部完成)
- `sys_design.md` — 原始设计文档

## 已知边界

- akshare 接口偶发变动: 所有调用收敛在 `data/akshare_client.py`, 修复只改这一处
- 盘中运行时 akshare 日K含当日未收盘行, 最后一行特征含义有偏差; 建议收盘后或开盘前运行
- 资金流数据仅覆盖近 ~120 交易日, 更早样本该组特征为 NaN(LightGBM 原生处理)
- 北向资金逐日净买入已被交易所自 2024-08 起停止披露: 该组特征可能全 NaN 自动降级(fetch 时有告警)
- Phase 2 扩展源(龙虎榜/增减持/解禁/宏观等)的字段映射为多候选防御式编写, 首次真实运行请按 `docs/experiments.md` 的联调清单核对
- 市场状态识别为规则引擎 v1(纯 pandas); HMM v2 明确缓行(准入条件见 `docs/experiments.md`)
- 多模型集成为 LightGBM+XGBoost 等权平均; LSTM 明确后置(准入条件 = 树模型集成在真实数据上显示多样性收益, 见 `docs/experiments.md`)
- 涨跌停判定用"当日收益 vs 板别限幅"近似(前复权价无法精确匹配涨停价); ST 判定依赖股票简称(需联网, 缺失时跳过); 停牌识别以上证交易日历为参照(详见 `docs/experiments.md` Phase 5)
- 合规校验为输出前固定闸门: 免责声明缺失/出现买卖措辞/置信度与警告不一致 → 阻断输出并记 `data/compliance_log.jsonl`
- Web 界面为只读 v1(无账户体系, `docker compose up -d web` → :8501); 与 CLI --json 消费同一 report dict, 一致性由构造保证
- 预测分布区间为三点混合情景示意(悲观/中性/乐观 = 历史条件均值, 截至 T-1), 不是收益分布预测; 报告中历史实盘准确率来自 track 摘要缓存(带"截至"时间, 无缓存自动隐藏)
- 漂移监控 PSI 的有效样本量 ≈ 票数(平滑特征近端窗口内高度自相关): 只查少量票时 alert 多为噪声, 阈值(0.10/0.25)按全票池口径, 真实数据校准是遗留观察项; Phase 7 之前的旧模型无漂移基线, 首次请重跑 `backtest --pool`
- 自动重训的"不劣于才替换"对比为近似口径(新旧回测数据窗口不同); 归档非破坏式、回滚幂等; CI 绿灯待首次 push 验收(命令与本地容器完全同口径)
