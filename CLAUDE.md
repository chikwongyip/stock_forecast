# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

A股日频走势预测 CLI(MVP, 按 `sys_design.md` 8 层方案裁剪): 技术面 LightGBM 三分类线 + DeepSeek 舆情打分线并行, 加权融合(0.7/0.3)输出。代码注释、文档、终端输出均为中文, 新代码保持一致。

## 命令

全部在 Docker 中运行, 本机无需 Python。首次: `cp .env.example .env`(DEEPSEEK_API_KEY 可选, 缺失时 LLM 线自动降级), `docker compose build`。

compose 把 `./src`、`./tests`、`./data` 热挂载进容器, **改代码无需 rebuild**。

```bash
docker compose run --rm sf fetch 600519          # 拉数据写 parquet 缓存
docker compose run --rm sf fetch --check 600519  # 数据源健康检查(纯本地, 不触网)
docker compose run --rm sf features 600519       # 特征矩阵与标签分布
docker compose run --rm sf backtest --pool       # walk-forward 回测 + 训练最终模型落盘(predict 的前置)
docker compose run --rm sf news 600519           # LLM 舆情线单独调试
docker compose run --rm sf regime                # 市场状态识别(牛/熊/震荡, L4 规则引擎)
docker compose run --rm sf predict 600519        # 端到端; --no-llm 纯技术面; --json 机器可读; --mode simple 新手模式(零术语)
docker compose run --rm sf track                 # 结算预测埋点, 输出实盘准确率(度量闭环), 并落摘要缓存供 predict/Web 展示
docker compose run --rm sf drift                 # 特征漂移 PSI 检查(告警落 data/alerts.csvl; 退出码 2 = alert 级)
docker compose run --rm sf retrain --dry-run     # 重训演练(只回测对比不写工件); 去掉 --dry-run 才真正替换+归档
docker compose run --rm sf rollback              # 回滚到最近归档的模型代际(含漂移基线); --list 只列归档
docker compose run --rm sf feedback 600519 --rating wrong --note "方向判反"   # 用户反馈; 无参数/--list = 复盘统计
docker compose run --rm --entrypoint bash sf /app/scripts/daily.sh     # 每日例行(收盘后批量预测+结算)
docker compose run --rm --entrypoint bash sf /app/scripts/maintain.sh  # 运维例行(漂移检查 + 告警/月度触发重训, 建议周频)
docker compose up -d web                         # Web 界面 v1 (Streamlit, http://localhost:8501, 只读)
```

测试(离线, 不依赖网络/LLM)。注意镜像 entrypoint 是 `python -m stock_forecast`, 不接受 `pytest` 作为子命令(README 中 `sf pytest -q` 的写法实际会报 invalid choice), 须覆盖 entrypoint:

```bash
docker compose run --rm --entrypoint pytest sf -q                              # 全量
docker compose run --rm --entrypoint pytest sf tests/test_labeling.py -q       # 单文件
docker compose run --rm --entrypoint pytest sf tests/test_labeling.py::test_no_lookahead_in_features -q  # 单测试
docker compose run --rm --entrypoint ruff sf check src tests                   # lint (line-length 100)
```

## 架构

数据流(括号内为对应模块):

```
akshare(data/akshare_client) → parquet 缓存(data/cache) → 采集编排(data/collector)
  → 特征工程(features/: indicators 技术指标 + sources 扩展源特征 → build 多表对齐+shift → labeling 三分类标签)
  → 技术面线(model/: train LightGBM + registry 注册表[LGBM+XGB 等权集成] + backtest walk-forward 多模型对比) ←─┐
  → 舆情线(llm/: news 采集清洗 + sentiment DeepSeek 结构化打分)  ←─┤ 两线并行
  → 市场状态(features/regime: 上证 MA 排列+波动率分位 → 牛/熊/震荡, L4) ─┤
  → 风控合规(risk/: status 交易状态[停牌/涨跌停/ST] + extreme 极端场景 + compliance 输出前校验, L6) ─┤
  → 融合+风控(fusion/combine, 含集成分歧度降档 + 交易状态降级) → 输出(output/: render 报告 + scenarios 情景区间 + simple 新手模式, llm/report) → data/predictions.csvl 埋点
  → 度量闭环(monitor/track: 埋点结算 → 实盘准确率报表 + track_summary.json 摘要缓存)
  → 运维自动化(monitor/drift: 特征 PSI 漂移 → alerts.csvl; model/retrain: 不劣于现役才替换+代际归档/回滚; monitor/feedback: 用户反馈 → feedback.csvl)

predict 总编排在 `pipeline.py`(predict_once), CLI(cli.run_predict 只做渲染)与
Web(web/app.py, Streamlit, compose `web` 服务)共用同一入口与同一 report dict。
```

扩展数据源(Phase 2, 全部 optional): 行业板块、北向资金、市场级两融、龙虎榜、
股东增减持、解禁日历、外盘(纳指/标普/VIX)、宏观(PMI/CPI/M2, 公布滞后建模)。
采集在 `collector.py`(缓存策略分 is_fresh/saved_within 两类), 特征在 `features/sources.py`。

- `cli.py`: argparse 子命令, handler 延迟 import(避免启动加载重依赖); `run_predict` 只做渲染/退出码, 编排在 `pipeline.predict_once`。
- `pipeline.py`: predict 端到端编排(fetch → 特征 → 推理 → 舆情 → 风控 → 融合 → 合规 → report+埋点), 返回结构化 result(ok/error/notices/blocked/report/simple/...), CLI 与 Web 共用; 埋点在 pipeline 内写入(合规阻断时不写)。
- `web/app.py`: Streamlit Web v1(只读, 无账户体系); 纯函数(report_for_display/accuracy_curve/settle_all)与 UI 分离, `main()` 有 `__main__` 守卫(import 无副作用, 可离线单测); compose `web` 服务用 `entrypoint` 整体覆盖镜像的 CLI ENTRYPOINT, 端口 8501; 预测结果进 session_state(反馈按钮 rerun 不丢报告), 反馈区与 CLI 共用 `monitor.feedback.append_feedback`。
- `monitor/drift.py`: 特征漂移 PSI(Phase 7.1); 基线 `drift_baseline.json` 由 backtest/retrain 落盘(`backtest.persist_final` 统一出口), **与模型代际同归档**; PSI 只算非 NaN 样本 + 缺失率单边上升检查 + 空箱半样本平滑 + 基线版本闸门; 告警落 `alerts.csvl`; 退出码 2 = alert 级(maintain.sh 的触发契约)。
- `model/retrain.py`: 自动重训(Phase 7.2); 替换判定 = 新回测方向准确率 ≥ 旧 `pool_meta.json` 的 backtest 块 − `retrain_tolerance`; 归档清单收敛在 `ARTIFACT_NAMES`(工件+meta+基线整代搬移), `data/models/archive/{ts}/`; rollback 非破坏式(归档保留, 幂等); 采纳/拒绝/回滚全落 alerts.csvl 审计。
- `monitor/feedback.py`: 反馈收集(Phase 7.4); CLI `feedback` 与 Web 按钮共用 `append_feedback`, 落 `feedback.csvl`(schema v1, rating 枚举 helpful/wrong/other); `--list` 输出月度复盘统计。
- `config.py`: pydantic-settings 的 `Settings`(环境变量覆盖), 含 DATA_DIR/CACHE_DIR/MODELS_DIR 路径、20 只默认票池、融合权重与各类阈值。运行时产物(缓存、模型、埋点、track 摘要)都在 gitignore 的 `data/` 下。

## 核心不变量(改动前必读)

1. **防前视契约**: 指标在 `indicators.py`/`features/sources.py`/`features/regime.py` 中按"第 T 行 = 截至 T 日收盘(含)的信息"计算, **一律不在特征层做时间平移**; `build.build_feature_matrix` 对全部特征列统一 `shift(1)` 一次, 使 X[T] 只含 ≤T-1 收盘后数据; 标签 y[T] = 第 T 日 close-to-close 方向(±`label_threshold` 内记"平")。`tests/test_labeling.py::test_no_lookahead_in_features` 有硬断言(篡改 T 日及之后原始数据——含全部扩展源, X[:T+1] 必须一字不变), 任何特征改动必须保持其通过。新增特征若自带 shift, 会造成双重平移。sources.py 中"公布时点晚于数据时点"的源必须显式建模滞后: 事件类只后顾聚合(date ≤ T), 宏观月频自 M+1 月 15 日生效, 日历 ffill 只向后; 新增源必须保持"原始帧 date > T 的行绝不影响第 T 行特征"这一机械可判定性(解禁前瞻日历因此被明确排除)。**市场状态同受此契约约束且更隐蔽**: regime 第 T 行含 T 日收盘信息, 而 predict 的目标是 y[T](X[T] 只含 ≤T-1), 所以融合/报告/埋点必须用 **T-1 行状态**(`regime.regime_for_prediction`/`shifted_regimes` 已内置对齐)——直接用 T 行会让当日收盘经"熊市降档/震荡收紧"泄露进方向判定, track 准确率失真。硬断言: `test_regime.py::test_prediction_uses_previous_close`。
2. **只允许时间有序分割**: 训练内早停验证集取训练窗尾部 20%(`train._time_ordered_val_split`), 回测用 expanding-window walk-forward(`backtest.walk_forward_folds`)。严禁随机分割。
3. **泄露报警不可放松**: 方向准确率(剔除真实"平")> 70% 直接打印泄露警告(`metrics.LEAK_WARNING_THRESHOLD`); 项目预期 55%~62% 已属优秀。不要通过调高阈值等方式"优化"准确率。
4. **akshare 收敛点**: 所有外部数据请求只在 `data/akshare_client.py`(重试 + 中文列名归一化为英文 snake_case)。akshare 接口变动只改这一处, 不要在别处直接调 `ak.*`。
5. **全链路优雅降级, 不 crash**: fund_flow 与全部 Phase 2 扩展源允许为 None(其列作为 optional_cols 保留 NaN, dropna 只看核心列, LightGBM 原生处理 NaN); sources.py 特征函数**恒输出固定列集合**(源缺失 → 全 NaN 列), 保证票池 concat 与模型特征清单跨股票稳定; 事件源语义区分 None(源不可用→NaN)与空帧(无事件→0, 本身是信号); LLM 任何失败(无 key/网络/JSON 非法)→ `available=False`, 融合层权重自动归 1/0; 票池单只股票数据失败不阻塞整体回测; **模型注册表(Phase 4)同口径**: xgboost 缺库时自动移出注册表, 回测/落盘中单成员训练失败即剔除该成员(集成退化为剩余成员), predict 加载 meta 成员失败退化并告警 —— 任何情况下 LightGBM 基线可用即主链路可用。**风控层(Phase 5)同口径**: `risk/*` 三模块均纯函数、任何数据缺失都降级为"正常/适用"不 crash; 股票简称(`fetch_stock_name`)仅用于 ST 识别, 缺失/失败 → 跳过 ST 检查(其余状态照常); index 不可用 → 不判停牌; 极端场景数据不足窗口 → 静默跳过(不误报)。
6. **融合规则**(`fusion/combine.py`): s_ml = p_up − p_down; s_llm = 三个 −5..+5 打分之和 /15; |s_final| 与方向阈值(±0.15)定方向; 两线反向且 |s_ml|<0.2 强制降级"震荡/观望+低置信"; 极端行情追加警告并把高置信降为中; 输出必须携带 DISCLAIMER。**市场状态调整(Phase 3/L4)**: `fuse(regime=...)` 中震荡市方向阈值 ×`regime_range_threshold_mult`(默认 1.5, 收紧→更多观望), 熊市"偏涨"置信度降一档+警告(牛市不调整); regime=None(未识别)完全向后兼容; 参数全在 config `regime_*`。**风控降级(Phase 5/L6)**: `fuse(risk_status=...)`(来自 `risk.status.assess_trading_status`)—— 当前停牌 applicable=False → 强制"震荡/观望+低置信"(不适用); 涨跌停/ST/近期停牌 reference_only=True → 置信度封顶"低"(仅供参考); risk_status=None 完全向后兼容; 极端场景在 `detect_extreme`(近5日涨跌/单日一字板)外追加 `risk.extreme.detect_extreme_scenarios`(连续一字板/近20日振幅/量能枯竭), 任一命中高置信封顶为中。**完整顺序 = 定方向 → 冲突降级(硬规则, 优先) → 熊市降档 → 分歧度降档 → 极端封顶(高→中) → 交易状态降级(最强, 可强制观望/低)**。**合规闸门(5.2)不在 fuse 内**: `risk.compliance.validate` 由 `cli.run_predict` 在 render 前调用(扫方向/警告/研判/LLM摘要: 免责声明存在+无买卖措辞+置信度与警告一致), 违规 → 落 `compliance_log.jsonl` + 阻断输出(`compliance_block` 可关为只告警); 正常 fuse 输出恒通过(全部警告来源都伴随降级)。
7. **标签编码**: 内部标签 {−1 跌, 0 平, 1 涨} 经 `train.LABEL_MAP` 映射为 LightGBM 类别 {0,1,2}; `predict_proba_df` 输出列固定为 p_down/p_flat/p_up 顺序。**注册表契约(Phase 4)**: 所有 `BaseModel` 成员的 `predict_proba` 必须输出同样的固定列序(XGBModel 用原生 Booster + 显式 `num_class=3` 保证训练窗缺类时仍恒为 3 列; xgboost 3.x 的 sklearn 包装器要求标签连续从 0 起, 缺类会 ValueError, 不要用); XGB 参数口径注意 `min_child_weight` 是 hessian 和(≈ LGBM `min_child_samples`×0.2), 不是样本数。
8. **报告单一数据源(Phase 6)**: `render.build_report_dict` 是报告数据的唯一构造点; CLI `--json`(`pipeline.report_json_text`)与 Web(`web.app.report_for_display`)必须消费 `predict_once` 返回的 result["report"] **同一个 dict**, 禁止在任何一端另建第二份报告结构(一致性验收由构造保证, 结构断言 `test_web_consistency.py`)。新手模式(`--mode simple`)文案必须过 `simple.SIMPLE_FORBIDDEN` 黑名单: 已知警告白话映射 + 未知警告统一兜底**不透传原文**(机制保证零术语, 断言 `test_simple.py`)。情景区间(`output/scenarios.py`)的条件幅度必须保持**剔除最后一行收益**(该收益 = 本次预测的标签, 展示口径与模型信息时点一致), 硬断言 `test_scenarios.py::test_no_lookahead_last_row`。
9. **运维自动化契约(Phase 7)**: ① **CLI 退出码必须透传**: `__main__.py` 用 `sys.exit(main())` —— 历史上曾丢弃返回值, 一切非零退出被吞成 0(daily.sh/maintain.sh 的失败分支全部失效), 回归测试 `test_cli.py::test_module_exit_code_propagation`; `drift` 退出码 **2 = alert 级漂移**(非错误, 是 maintain.sh 触发重训的信号, 不要"修复"成 0/1)。② **漂移基线与模型代际同进退**: retrain 替换前整代归档(`retrain.ARTIFACT_NAMES` 清单: 工件+meta+基线), rollback 整代恢复; 禁止跨代际混用基线。③ **重训"不劣于才替换"**: 新回测方向准确率 ≥ 旧 meta.backtest − `retrain_tolerance` 才采纳; 归档非破坏式、回滚幂等; 采纳/拒绝/回滚全落 `alerts.csvl` 审计。④ **PSI 口径三件事不可放松**: NaN 不进分箱(缺失率只做单边上升检查, 覆盖改善不告警)、空箱半样本平滑(0.5/n 为底)、基线版本闸门(旧格式整体拒绝 —— "全部特征评估失败"绝不允许报"分布稳定")。

## 已知边界

- 盘中运行时 akshare 日K含当日未收盘行, 最后一行特征含义有偏差; 系统假定收盘后或开盘前运行。
- 资金流数据仅覆盖近 ~120 交易日, 更早样本该组特征为 NaN。
- 北向资金逐日披露已被交易所自 2024-08 起停止: 该组特征可能全 NaN 自动降级(`fetch_northbound` 检测滞后 >30 天并告警)。
- Phase 2 扩展源的 akshare 字段映射为多候选防御式编写, 未在真实网络下联调(沙箱无网); 首次真实运行按 `docs/experiments.md` 联调清单核对, 映射失败会告警并降级 NaN。
- `predict` 依赖 `backtest --pool` 落盘的注册表工件: `data/models/pool_meta.json`(元数据: 成员/特征列清单/训练时间范围) + `pool_model.txt`(LightGBM) + `pool_xgb.json`(XGBoost); 无 meta 的旧版 `pool_model.txt` 自动回退单 LightGBM 路径。特征一致性检查以注册表特征清单为准, 不一致报"特征缺失(缓存过旧?)"。新增特征列后旧模型仍可 predict(多余列被忽略), 但要让新特征生效必须重跑 backtest。
- 多模型回测(Phase 4)训练时长 ≈ 成员数 × 单模型(`backtest --pool` 每 fold 训练 lgbm+xgb 两个模型); 回测主表仍为 LightGBM 基线口径(跨 Phase 可比), 多模型对比与 p_up 相关矩阵另表输出。
- 埋点 schema v4: v3 +`trading_status`/`applicability`(交易状态与适用性, Phase 5); v3 = +`model`/`disagreement`; `track` 读取端对 v0~v4 全兼容(缺字段回填"未标注"/忽略)。当前停牌股无次日行情 → 结算天然 pending, 不污染准确率。
- 缓存新鲜度两类: 当日即有数据的源按最新数据日期(`is_fresh`); T+1 公布/全市场共享/月频源按文件写入时间(`saved_within`), 避免永远"不新鲜"反复拉网。
- 每批次特征增删须在 `docs/experiments.md` 记录票池回测对比(方向准确率+逐轮std), 无提升即回退。
- 市场状态识别为**规则引擎 v1**(纯 pandas, 不进模型特征矩阵, X schema 不变); HMM v2(计划 3.3 可选项)明确缓行, 准入条件见 `docs/experiments.md`。`fetch_index` 拉 2 年历史供 MA60+250 日波动率分位暖机; 旧 365 天缓存仍可用(分位样本不足自动不触发高波动覆盖), 过期后自愈。
- 埋点 schema v2 起含 `regime`/`regime_as_of`(预测信息时点 T-1 的状态); `track` 对 v0/v1/v2 兼容(旧记录回填"未标注"), 状态分桶只对 v2 起的记录有意义。
- 涨跌停判定(Phase 5/5.1)用"当日收益 vs 板别限幅+容差"近似, **非精确涨停价匹配**: 缓存为前复权价, 复权因子破坏交易所 round(前收×(1±限幅),2) 的分位舍入; 漏判不会有, 误判最多多一条"仅供参考"。ST 判定依赖股票简称(`fetch_stock_name`, 缓存30天, 需联网), 缺失时跳过 ST。真实停牌股 `fetch_kline` 因 is_fresh=False 每次触网重拉(akshare 返回同样陈旧K线, 无害); 沙箱无网时该场景走不通, 已在单测+直接集成校验覆盖。
- 合规违规拦截记录落 `data/compliance_log.jsonl`(gitignore); 正常输出恒通过校验, 该文件通常不存在。`compliance_block=False` 为应急开关(只告警+记日志不阻断)。
- Web 界面(Phase 6)依赖 streamlit(镜像内置, 新增依赖后需 `docker compose build`); compose `web` 服务必须用 `entrypoint` 整体覆盖(镜像 ENTRYPOINT 是 CLI 模块, 用 `command:` 会被当成子命令参数报 invalid choice)。健康检查: `curl localhost:8501/_stcore/health`。
- predict/Web 展示的"历史实盘准确率"来自 `track` 落盘的摘要缓存 `data/track_summary.json`(展示带"截至"时间); 缓存由每次 track 运行刷新(daily.sh 已含), 无缓存/未结算时该段自动隐藏(JSON 为 null), predict 不会因此触网重结算。
- 预测分布区间是**三点混合的情景示意**(悲观/中性/乐观原子 = 回看窗内历史条件均值), 不是收益分布预测; 分位数可能落在同一原子, 输出恒带口径说明。
- PSI 漂移检查的**有效样本量 ≈ 票数而非行数**(平滑特征近端窗口内每票是一段近邻值): 只查 1~2 只票时 alert 多为小样本噪声, 默认阈值(0.10/0.25)按全票池(~400 行)口径, 真实数据校准是遗留观察项(experiments.md Phase 7 观察项①)。Phase 7 之前训练的模型无漂移基线/meta.backtest 块: drift 提示重建(降级返回 0), retrain 对比时默认采纳 —— 首次应重跑 `backtest --pool`。
- retrain 新旧回测的数据窗口不同(重训时历史更长), "不劣于"判定为近似参考而非严格同分割对比。`maintain.sh` 月度节奏靠标记文件 `data/.last_retrain_month`(重训成功——采纳或明确拒绝——才更新)。
- CI(`.github/workflows/ci.yml`)在沙箱无法执行验证, "绿灯"待首次 push 验收; 其命令与容器内口径完全一致(`ruff check src tests` / `pytest -q`), compose 挂载 `./.github` 供 test_ci 静态校验。
- `.gitignore` 曾有 `datas/` 笔误导致 data/ 实际未被忽略(此前 git status 干净只因 data/ 为空), Phase 7 已补 `data/`; alerts.csvl/feedback.csvl/archive 等运行时文件全部在其下。
