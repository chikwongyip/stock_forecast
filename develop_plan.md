# 开发计划 (develop_plan.md)

> 依据: `sys_design.md` 8 层架构 + 当前仓库 MVP 现状盘点
> 说明: `sys_design.md` 现存文本在 §4.3 处截断(目录中的五~十四章正文缺失), 本计划以已有的
> 8 层架构图(§2)、数据源全景(§3.1)、特征工程规格(§4)与 README"已知边界"为准。
> 工作量为单人 + AI 辅助的粗估, 按 1 人日 ≈ 6 小时有效编码计。

---

## 一、现状盘点: MVP 已覆盖 vs 设计缺口

| 设计层 | 已实现 (MVP) | 缺口 |
| --- | --- | --- |
| L1 数据采集 | 日K线 / 上证指数 / 个股资金流 / 个股新闻(akshare 收敛于 `data/akshare_client.py`, parquet 缓存) | 北向资金、融资融券、龙虎榜、板块联动、宏观指标(PMI/CPI/M2/SHIBOR/国债/汇率)、股东增减持、解禁日历、国际市场(美股/港股/VIX/商品) |
| L2 预处理&异常检测 | 优雅降级(单源失败不阻塞)、缓存新鲜度检查 | 系统化的数据源健康检查、异常值检测、缺失值策略文档化 |
| L3 特征工程 | ~20 个纯 pandas 技术指标 + 指数特征 + 资金流特征, 统一 shift(1) 防前视, ±0.5% 三分类标签 | 板块动量、事件衰减、宏观特征、北向/龙虎榜特征 |
| L4 市场状态识别 | 无 | 牛/熊/震荡识别(规则引擎或 HMM), 按状态切换策略 |
| L5 多模型集成 | LightGBM 三分类(票池横截面模型, walk-forward 回测, 早停) | XGBoost、LSTM、模型集成加权、多时间尺度融合 |
| L6 风控&后处理 | 两线冲突降级、极端行情(近5日累计涨跌/一字板)降置信、免责声明 | 停牌/涨跌停/ST 识别、合规检查清单化 |
| L7 输出 | rich 终端报告 + `--json`, 置信度三档, top5 特征归因, 模板化研判文本 | 不确定性区间、模型历史准确率参考、用户自适应(新手/专业) |
| L8 监控闭环 | `data/predictions.csvl` 预测埋点(仅落盘) | 实盘准确率追踪、特征漂移监控、自动重训练、用户反馈 |
| 舆情线 | 新闻采集清洗 + DeepSeek 结构化打分(三维 -5..+5) + 当日缓存 + 全链路降级 | 多源新闻(新浪/雪球)、研报、社交媒体 |
| 基础设施 | Docker 全套运行、离线单测(含防前视硬断言)、源码热挂载 | CI、定时任务、Web 界面 |

**结论**: MVP 已打通"数据→特征→模型→融合→输出"主干, 最大的短板不是模型不够多, 而是
**没有度量闭环**(L8)——无法回答"系统实盘到底准不准", 后续一切增强都无从验证收益。

---

## 二、总则: 全程不可违背的工程约束

以下不变量贯穿所有阶段, 任何新代码合入前必须保持(源自 `CLAUDE.md`):

1. **防前视契约**: 新特征一律在 `indicators.py` 层按"第 T 行 = 截至 T 日收盘信息"计算,
   时间平移只由 `build.py` 的统一 `shift(1)` 完成; 每新增一组特征, 必须扩展
   `tests/test_labeling.py::test_no_lookahead_in_features` 的篡改覆盖面。
2. **只允许时间有序分割**, 严禁随机 split; 回测一律 walk-forward。
3. **泄露报警不放松**: 方向准确率 >70% 视为泄露信号(`metrics.LEAK_WARNING_THRESHOLD`)。
4. **akshare 收敛点**: 新数据源只在 `data/akshare_client.py` 增加函数(重试 + 列名归一化),
   其他模块不得直接调 `ak.*`。
5. **优雅降级**: 任何新数据源/新模型失败都不得让主管道 crash, 缺失按 NaN/权重归零处理。
6. **测试离线可复现**: 单测不依赖网络与 LLM, 用合成数据。

---

## 三、总体路线图

排序原则: **先度量, 再数据, 后模型; 界面最后。** 每个 Phase 结束都是一个可独立交付、
可回退的里程碑。

```
Phase 1  度量闭环 + 数据健康 (L8 基础 / L2)        ████        ~5-7 人日
Phase 2  数据源与特征扩展 (L1 / L3)                ████████    ~8-12 人日
Phase 3  市场状态识别 (L4)                          █████       ~5-8 人日
Phase 4  多模型集成 (L5)                            ██████      ~6-10 人日
Phase 5  风控与合规增强 (L6)                        ███         ~3-4 人日
Phase 6  输出与用户体验 (L7 / Web)                  ██████      ~6-10 人日
Phase 7  运维自动化 (L8 高级 / 部署)                ████        ~4-6 人日
                                                    合计 ~37-57 人日
依赖: 1 → 2 → {3, 4}(可并行) → 5; 6 依赖 1(要展示历史准确率); 7 依赖 1、4
```

---

## 四、分阶段计划

### Phase 1 — 度量闭环 + 数据健康 (最高优先级)

**目标**: 回答"系统实盘准不准"; 没有这个基线, 后续 Phase 3/4 的模型增强无法证明有效。

| # | 任务 | 涉及模块 | 验收标准 |
| --- | --- | --- | --- |
| 1.1 | `track` 子命令: 读取 `predictions.csvl`, 拉取对应股票后续行情, 结算每条预测的命中/未中, 输出滚动 20/60 日方向准确率、置信度分桶准确率、ML 线 vs 融合线对比 | `cli.py`, 新增 `monitor/track.py` | 对合成埋点数据的单测通过; 真实数据下能输出报表 |
| 1.2 | 每日例行: `predict --json` 批量跑票池 + 追加埋点(容器内 cron 或宿主 crontab, 收盘后运行) | 新增 `scripts/daily.sh`, compose 增加 profile | 连续 5 个交易日无人工干预产生埋点 |
| 1.3 | 数据源健康检查: `fetch --check` 输出各源行数/最新日期/列完整性; 缓存过期与列缺失显式告警 | `data/collector.py`, `data/cache.py` | 断网/坏缓存场景下报告明确, 不 crash |
| 1.4 | 埋点 schema 固化: 加 `schema_version` 字段, `track` 兼容多版本 | `cli.py`, `monitor/track.py` | 旧埋点仍可读 |

**风险**: akshare 盘中/收盘数据时序不稳定 → 例行任务固定在收盘后(17:00 之后)运行。

---

### Phase 2 — 数据源与特征扩展 (L1 / L3)

**目标**: 补充与现有特征正交的信息源。按"信息价值 / 接入成本"排序分两批, 每批独立可交付,
每加一组特征都跑一次 walk-forward 回测对比基线(方向准确率与逐轮 std), **无提升即回退**。

**批次 A (~5-7 人日, 结构化日频数据, 接入容易)**

| # | 数据源 | akshare 接口(收敛进 `akshare_client.py`) | 派生特征(进 `build.py`) |
| --- | --- | --- | --- |
| 2.1 | 板块/行业联动 | `stock_board_industry_name_em`, `stock_board_industry_hist_em` | 所属行业 1/5/20 日动量、个股相对行业超额收益、行业内排名 |
| 2.2 | 北向资金 | `stock_hsgt_north_net_flow_in_em` | 北向净流入 1/3/5 日、与指数背离度 |
| 2.3 | 融资融券 | `stock_margin_detail_*` | 融资余额变化率 5/20 日 |

**批次 B (~3-5 人日, 事件型/低频数据, 允许 NaN)**

| # | 数据源 | 派生特征 |
| --- | --- | --- |
| 2.4 | 龙虎榜 | 近 N 日上榜次数、机构净买入(事件衰减加权) |
| 2.5 | 股东增减持 / 解禁日历 | 事件衰减特征(半衰期 ~10 交易日) |
| 2.6 | 国际市场(隔夜美股/纳指/VIX/汇率) | 隔夜外盘收益、VIX 变化 |
| 2.7 | 宏观(PMI/CPI/M2/SHIBOR) | 月频前向填充 + 环比变化(仅作弱特征) |

**统一要求**: 全部作为 optional 列(允许 NaN, 不参与 dropna); 缓存 key 沿用
`{kind}_{symbol}` 约定; 每源失败降级为 None。

**验收**: 防前视硬断言覆盖新特征; 批次 A/B 各出一份"加入前后回测对比"记录(写入
`docs/experiments.md`)。

---

### Phase 3 — 市场状态识别 (L4)

**目标**: 识别牛/熊/震荡三状态, 让融合层按状态调整(而非直接切换多套模型, 控制复杂度)。

| # | 任务 | 说明 |
| --- | --- | --- |
| 3.1 | 规则引擎 v1 | 上证指数 MA20/MA60 排列 + 20 日波动率分位 → 三状态; 纯 pandas, 可单测 |
| 3.2 | 状态接入融合层 | `combine.fuse` 增加 regime 参数: 震荡市收紧方向阈值、熊市下调"偏涨"置信; 规则写入配置可调 |
| 3.3 | HMM v2(可选) | `hmmlearn` 三状态高斯 HMM, 特征 = 指数收益/波动率/成交额; 与规则引擎一致性对比, 差异大时以规则为准并告警 |
| 3.4 | 输出层展示 | 报告与 `--json` 中带当前市场状态及判定依据 |

**验收**: 状态序列单测(合成牛/熊/震荡数据); 回测按状态分桶的方向准确率报表;
接入前后整体指标不回退。 **工作量**: ~5-8 人日(不含 HMM 则 ~3 人日)。

---

### Phase 4 — 多模型集成 (L5)

**目标**: 在 LightGBM 基线之上验证集成增益, 有增益才保留, 避免为集成而集成。

| # | 任务 | 说明 |
| --- | --- | --- |
| 4.1 | 模型注册表抽象 | `model/` 增加统一接口 `fit/predict_proba/save/load`, LightGBM 先迁入; 模型落盘带特征列清单 + 训练数据时间范围元信息 |
| 4.2 | XGBoost 第二模型 | 同特征同标签, 独立 walk-forward; 与 LightGBM 预测相关性分析 |
| 4.3 | 概率平均集成 v1 | 简单等权平均 → 回测对比单模型; 提升 <1pp 则止步于此 |
| 4.4 | LSTM(可选, 明确后置) | 仅当 4.3 显示树模型多样性收益明显时再做; 需引入 torch, 镜像体积与训练时长显著增加 |
| 4.5 | 不确定性量化 | 集成分歧度(模型间概率方差)并入 `combine.ml_uncertainty`, 分歧大 → 置信度降档 |

**验收**: 每个模型独立通过泄露检查; 集成回测报告(准确率 + 逐轮 std + 相关性矩阵)
存档 `docs/experiments.md`。 **工作量**: ~6-10 人日(LSTM 另计 +5)。

---

### Phase 5 — 风控与合规增强 (L6)

**目标**: 把"不该给结论的场景"识别全。

| # | 任务 | 说明 |
| --- | --- | --- |
| 5.1 | 交易状态识别 | 停牌(K线断档)、当日涨跌停(收盘价 = 涨/跌停价)、ST/*ST(名称含 ST) → 输出"不适用/仅供参考"降级 |
| 5.2 | 合规检查清单 | `fusion/combine.py` 输出前固定校验: 免责声明存在、无买卖措辞、置信度与警告一致性; 违规项阻断输出并记日志 |
| 5.3 | 极端场景扩充 | 连续一字板、近 20 日振幅异常、成交量枯竭(量比分位 <5%) |

**验收**: 每项规则一个单测; 用历史真实停牌/ST 股票回放验证降级路径。**工作量**: ~3-4 人日。

---

### Phase 6 — 输出与用户体验 (L7 + Web)

**目标**: 从"自己用的 CLI"到"给别人看的系统"。依赖 Phase 1 的历史准确率数据。

| # | 任务 | 说明 |
| --- | --- | --- |
| 6.1 | 报告增强 | 展示该股/全票池滚动实盘准确率(来自 `track`)、预测分布区间(p25/p50/p75 对应涨跌幅场景描述)、市场状态 |
| 6.2 | 用户自适应 | `--mode simple/pro`: simple 只给方向+置信度+一句话理由+风险提示; pro 全量 |
| 6.3 | Web 界面 v1 | Streamlit(或 FastAPI + 轻前端): 输入代码 → 调用现有 predict 管道 → 渲染报告; 只读, 不做账户体系 |
| 6.4 | 历史浏览 | Web 端展示 `predictions.csvl` 的结算结果与准确率曲线 |

**验收**: Web 端与 CLI `--json` 输出一致性测试; 新手模式不含任何专业术语黑名单词。
**工作量**: ~6-10 人日。

---

### Phase 7 — 运维自动化 (L8 高级 + 部署)

| # | 任务 | 说明 |
| --- | --- | --- |
| 7.1 | 特征漂移监控 | 每周对比近 20 日特征分布 vs 训练期分布(PSI); 超阈值告警写入 `data/alerts.csvl` |
| 7.2 | 自动重训练 | 每月或漂移告警后自动重跑 `backtest --pool`; 新模型仅在回测不劣于旧模型时替换(旧模型归档可回滚) |
| 7.3 | CI | GitHub Actions: ruff + pytest(离线) on push |
| 7.4 | 用户反馈收集 | `feedback` 子命令 / Web 端按钮, 落 `data/feedback.csvl`, 纳入月度复盘 |

**验收**: 重训练流程演练一次(含回滚); CI 绿灯。**工作量**: ~4-6 人日。

---

## 五、里程碑验收一览

| 里程碑 | 完成标志 | 累计工作量 |
| --- | --- | --- |
| M1 度量闭环 | 连续一周自动埋点 + `track` 出实盘准确率报表 | ~5-7 人日 |
| M2 数据扩展 | 批次 A/B 接入, 回测对比存档, 防前视测试全绿 | ~13-19 人日 |
| M3 状态感知 | regime 进入融合与报告, 分状态回测报表 | ~18-27 人日 |
| M4 模型集成 | 集成回测报告, 无增益则明确记录"止步于双模型/单模型" | ~24-37 人日 |
| M5 风控合规 | 停牌/涨跌停/ST 降级路径全覆盖 | ~27-41 人日 |
| M6 对外可用 | Web 界面 + 自适应报告上线 | ~33-51 人日 |
| M7 自动化运维 | 漂移监控 + 自动重训 + CI | ~37-57 人日 |

## 六、主要风险与对策

| 风险 | 对策 |
| --- | --- |
| akshare 接口变动(高频风险) | 收敛点原则 + `fetch --check` 健康检查(1.3) + 接口失败降级不阻塞 |
| 新特征引入前视泄露 | 硬断言测试随特征同步扩展(总则 1); 每批次回测强制过泄露报警 |
| 回测提升但实盘不提升 | Phase 1 的实盘追踪是唯一裁判; `docs/experiments.md` 记录每次"回测 vs 实盘"差异 |
| 范围蔓延(LSTM/Web 做大) | LSTM 明确后置且设准入条件(4.3); Web v1 只读、无账户体系 |
| 准确率预期失控 | 全程坚持 55%~62% 优秀、>70% 报警的项目共识, 输出层永远携带免责声明 |

## 七、明确不做 (本期范围外)

- 日内/分钟级预测(设计定位是日频, 盘中数据语义有偏差)
- 自动交易/下单对接(合规红线, 系统只输出研判)
- 多账户/权限体系、生产级高可用部署
- tushare/Wind 等付费数据源(维持 akshare 免费源, 接口预留可替换)

---

## 八、开发进度日志

### ✅ Phase 1 — 度量闭环 + 数据健康 (已完成, 2026-09-29)

**交付物**

| 任务 | 产出 | 状态 |
| --- | --- | --- |
| 1.1 `track` 子命令 | 新增 `src/stock_forecast/monitor/track.py`: 埋点结算、真实方向标签、ML线 vs 融合线对比、滚动 20/60 方向准确率、置信度分桶命中率; `--json` / `--refresh`; 纯函数(settle/summarize)与 IO 分离 | ✅ |
| 1.2 每日例行 | 新增 `scripts/daily.sh`(批量预测票池 + track, 单票失败不阻塞); compose 挂载 `./scripts`; crontab 用法写在脚本头注释 | ✅ |
| 1.3 数据源健康检查 | `fetch --check`: `collector.source_health()` 纯本地检查行数/最新日期/关键列/新鲜度, loader 可注入; 状态 ok/expired/incomplete/missing, 核心 K 线缺失退出码 1 | ✅ |
| 1.4 埋点 schema 固化 | 埋点写入加 `schema_version=1`; `load_predictions` 对旧记录回填 v0, 坏行静默跳过 | ✅ |

**验证**: `ruff check` 全绿; `pytest` **53 passed**(原 41 + 新增 12); 合成缓存 + 合成埋点在容器内跑通完整 `track` 结算链路(端到端数字手工核对一致)。验证用合成数据已清理。

**顺带修复的历史遗留问题**(开发 Phase 1 时暴露, 与 CLAUDE.md 记录的"镜像不可用"一致):

1. **镜像缺 `libgomp1`** → LightGBM 无法 `import`, 原 `backtest`/`predict`/`pytest` 在容器内全部 crash。Dockerfile 增加 `apt-get install libgomp1`。
2. **依赖版本漂移致 5 个存量测试失败**(HEAD 即失败, 非本次引入): EMA 测试手算用了错误的 α=1/3(标准 span 应为 2/(n+1))、RSI/BOLL 的 `np.allclose` 缺 `equal_nan`、`test_label_thresholds` 用 `==` 比较含 NaN 列表、防前视测试篡改了资金流整列(与"只改 t 行之后"注释矛盾)。逐一修正测试使其符合实现语义与既定不变量, **未改动任何生产逻辑**。
3. **ruff 0.16.9 默认规则集扩大**(新增 DTZ/BLE/SIM/RUF… 30 项告警)→ pyproject 显式固定 `select=["E4","E7","E9","F"]`(= 项目原始口径, 代码中 `# noqa: BLE001` 即为此而留), 保证 lint 跨版本一致。compose 挂载 pyproject 使配置热更新。
4. 清理 E741(`l`→`lo`)、未用 import、未用变量等 F 级问题。

**对后续阶段的提示**: lightgbm `eval_set` 参数有 DeprecationWarning(新版建议 `eval_X/eval_y`), 留到 **Phase 4.1 模型接口抽象**时统一处理, 避免在 Phase 1 扩大改动面。

### ✅ Phase 2 — 数据源与特征扩展 (离线部分完成, 2026-09-30)

**交付物**

| 任务 | 产出 | 状态 |
| --- | --- | --- |
| 2.1 板块/行业联动 | `get_stock_industry`/`get_industry_hist` + `industry_features`(sec_ret1/5/20、sec_vol20、excess_ret1/5) | ✅ 离线 |
| 2.2 北向资金 | `get_northbound` + `north_features`(net1/3/5、div5 背离度); 停更检测告警 | ✅ 离线, ⚠ 数据源疑似已停更 |
| 2.3 融资融券 | `get_margin_sse`(市场级) + `margin_features`(chg5/20) | ✅ 离线 |
| 2.4 龙虎榜 | `get_lhb`(全市场共享缓存) + 通用 `event_decay_features`(半衰期14天/窗口60天) | ✅ 离线 |
| 2.5 增减持/解禁 | `get_holder_events`/`get_unlock_schedule` + 事件衰减(仅后顾, 见范围决策) | ✅ 离线 |
| 2.6 国际市场 | `get_global_index`(.IXIC/.INX/.VIX) + `global_features`(隔夜收益/VIX) | ✅ 离线 |
| 2.7 宏观 | `get_macro_china`(PMI/CPI/M2, 多候选解析) + `macro_features`(公布滞后 M+1月15日生效) | ✅ 离线 |
| 基础设施 | `cache.saved_within`(T+1/共享/月频源新鲜度); `build` extras 管道 + `make_extras`; fetch_all 扩至 11 源 | ✅ |

**验证**: pytest **69 passed**(新增 16), ruff 全绿; 防前视硬断言已扩展覆盖全部新源(统一篡改规则 + "列是活的"防退化断言); 容器内合成缓存端到端: features(73 列) → backtest 训练落盘 → predict 特征一致性通过 → 埋点 v1 → track。

**范围决策与风险**(完整记录见 `docs/experiments.md`): 行业内排名→超额收益(成本); 个股两融→市场级(接口限制); 解禁只用后顾事件(保防前视契约机械可判定性); 汇率裁剪; 北向疑似停更(联调确认后决定去留)。

**⏳ 遗留(需本机网络)**: ① `fetch 600519 --refresh` 真实联调, 按 experiments.md 清单核对字段映射; ② 基线 vs +批次A vs +批次A/B 的票池 walk-forward 回测对比, 填表并按"无提升即回退"裁决。

### ✅ Phase 3 — 市场状态识别 L4 (离线部分完成, 2026-10-02)

**交付物**

| 任务 | 产出 | 状态 |
| --- | --- | --- |
| 3.1 规则引擎 v1 | 新增 `features/regime.py`: 上证 MA20/MA60 排列 + 20日波动率分位(250日窗口百分位秩) → 牛/熊/震荡/未知; 纯 pandas 向量化; 参数全部进 config(`regime_*` 7 项)环境变量可调 | ✅ |
| 3.2 状态接入融合层 | `combine.fuse(regime=...)`: 震荡市方向阈值 ×1.5(收紧→更多观望); 熊市"偏涨"置信度降一档+警告(可配置关闭); 牛市不调整; regime=None 完全向后兼容 | ✅ |
| 3.3 HMM v2(可选) | **缓行**(范围决策): 需新增 hmmlearn 依赖重建镜像, 且规则版真实收益未验证; 触发条件与评估口径见 experiments.md | ⏸ 决策记录 |
| 3.4 输出层展示 | predict 报告/--json 带状态+判定依据(T-1 口径); 新增 `regime` 子命令(当前状态/预测口径/近60日分布/近10日序列); 埋点升 **v2**(+regime/regime_as_of); track 增状态分桶; backtest 增分状态方向准确率报表(验收项) | ✅ |
| 基础设施 | `fetch_index` 拉长到 2 年(MA60+波动率分位暖机; 旧 365 天缓存优雅降级, 过期自愈) | ✅ |

**关键设计 — 信息时点 T-1 对齐(防前视)**: regime 第 T 行 = 截至 T 日收盘状态;
但 predict 预测 y[T] 时 X[T] 只含 ≤T-1 数据, 故融合/报告/埋点一律取 **T-1 行**
(`regime_for_prediction`/`shifted_regimes` 内置对齐)。若直接用 T 行, 当日收盘信息
会经状态调整泄露进方向判定, track 准确率失真。专项硬断言:
`test_prediction_uses_previous_close` + `test_classify_no_lookahead`。

**范围决策**: regime 不进模型特征矩阵(X 保持 73 列, 旧模型兼容无需重训), 只在
融合层做规则调整 —— 与计划"而非直接切换多套模型, 控制复杂度"一致; 高波动覆盖
(多头排列但波动率分位 ≥85% → 震荡)是 v1 的保守化设计, 真实触发频率待本机观察。

**验证**: pytest **90 passed**(新增 21), ruff 全绿; 容器内合成缓存端到端:
`regime`(三段合成指数牛→震荡→熊, 末段判熊市) → `backtest`(分桶报表正常) →
`predict --json`(regime 块 T-1 对齐, 埋点 v2) → `track`(状态分桶表)。验证用合成数据已清理。

**⏳ 遗留(需本机网络)**: ① 真实指数上 `regime` 人工核对判定合理性(尤其高波动覆盖
触发频率); ② `backtest --pool` 填写 experiments.md 分状态方向准确率表(验收项);
③ track 状态分桶需埋点 v2 运行数周后才有实盘对照意义。

### ✅ Phase 4 — 多模型集成 L5 (离线部分完成, 2026-10-02)

**交付物**

| 任务 | 产出 | 状态 |
| --- | --- | --- |
| 4.1 模型注册表 | 新增 `model/registry.py`: `BaseModel` 统一接口, LightGBM 迁入; 落盘带 `pool_meta.json`(特征列清单+训练时间范围+成员清单); 旧版无 meta 的 pool_model.txt 自动回退; **lightgbm `eval_set` 弃用警告已消除**(Phase 1 遗留: 改用 4.7 新式 `eval_X/eval_y`, pyproject 固定 `lightgbm>=4.7,<5`) | ✅ |
| 4.2 XGBoost 第二模型 | `XGBModel`(原生 Booster API + num_class=3 显式声明, 缺类仍恒 3 列); 同特征同标签独立早停; p_up 相关矩阵 + 方向一致率进回测报表; xgboost 进依赖(镜像已重建) | ✅ 离线 |
| 4.3 等权集成 | `EnsembleModel` 概率等权平均; `run_multimodel_backtest` 同分割跑 lgbm/xgb/ensemble, **每组独立泄露报警**(验收项); 单成员失败自动剔除不阻塞; 最终落盘 = 全体可用成员集成 | ✅ 离线 |
| 4.4 LSTM | **不做**(计划原文"明确后置"): 准入条件 = 4.3 真实数据上显示树模型多样性收益明显 | ⏸ 决策记录 |
| 4.5 不确定性量化 | `disagreement(X)` = 成员概率两两 TV 距离均值; `fuse(disagreement=...)` ≥ 阈值(默认 0.10, config 可调)→ 置信度降一档+警告; 单模型 None 不生效; 埋点升 **v3**(+model/disagreement) | ✅ |

**实现要点**(坑与决策全录见 `docs/experiments.md`): xgboost 3.x sklearn 包装器
要求标签连续从 0 起 → 改原生 API; `min_child_weight` 为 hessian 和口径(照抄
LightGBM 的 50 会致小树无法分裂)→ 取 10; 分歧度做成独立降档规则而非揉进
`ml_uncertainty`(两个正交口径, 可解释可单测); top5 归因锚定 LightGBM gain。

**验证**: pytest **105 passed**(新增 16), ruff 全绿, DeprecationWarning 清零;
容器内合成缓存端到端: `backtest`(三组指标+相关矩阵+集成落盘+meta 验收字段) →
`predict --json`(model=lgbm+xgb, disagreement 链路, 埋点 v3) → `track` 兼容。
验证用合成数据已清理。

**⏳ 遗留(需本机网络)**: `backtest --pool` 跑真实票池, 填 experiments.md 的
lgbm/xgb/ensemble 对比表(方向准确率+逐轮std+相关矩阵), 按"ensemble 提升 <1pp
且 std 无改善则止步于单模型"裁决(predict 端可配置回退)。

### ✅ Phase 5 — 风控与合规增强 L6 (离线部分完成, 2026-10-05)

**交付物**(新增 `risk/` 包, 三模块均纯函数无 IO)

| 任务 | 产出 | 状态 |
| --- | --- | --- |
| 5.1 交易状态识别 | `risk/status.py`: 停牌(K线断档, 以上证交易日历为参照)/涨跌停(收益 vs 板别限幅, 前复权口径)/ST(简称含 ST)→ 两级降级: **不适用**(当前停牌→强制观望+低置信)、**仅供参考**(涨跌停/ST/近期停牌→置信度封顶低); 简称经 `collector.fetch_stock_name`(缓存30天)获取, 缺失跳过 ST | ✅ 离线 |
| 5.2 合规检查清单 | `risk/compliance.py`: render 前固定闸门(①免责声明存在 ②无买卖/诱导措辞[词组级黑名单扫描方向/警告/研判/LLM摘要] ③置信度与警告一致+方向合法); 违规阻断输出并落 `data/compliance_log.jsonl`; `compliance_block` 应急开关 | ✅ |
| 5.3 极端场景扩充 | `risk/extreme.py`: 连续一字板(≥2日)/近20日振幅异常(高低差/中位收盘≥40%)/量能枯竭(量分位≤5%且低于中位, 中位数守卫防等量误报); 经 `fuse` 接入, 任一命中高置信封顶为中 | ✅ |

**实现要点**(全录见 `docs/experiments.md`): 涨跌停判定用"收益 vs 板别限幅"(主板10/
创业板科创板20/北交所30/主板ST5)+容差, 而非"价=涨停价"(前复权破坏分位舍入); 停牌两级
降级(当前停牌=不适用 vs 近期复牌=仅供参考); 合规黑名单词组级避免误伤"涨跌停/偏涨跌";
置信度-警告一致性做成兜底断言(正常输出恒通过, 捕获未来回归); 埋点升 **v4**
(+trading_status/applicability), track 对 v0~v4 全兼容。

**验证**: pytest **151 passed**(新增 46: status 20 / extreme 8 / compliance 13 /
fuse 集成 6), ruff 全绿; 容器内合成缓存端到端: `backtest`(集成落盘)→ `predict` 三场景
(正常=适用 / 停牌=不适用[直接集成校验] / ST=仅供参考[全CLI])→ 埋点 v4 → `track` 分桶 →
合规阻断渲染冒烟。验证用合成数据已清理。

**⏳ 遗留(需本机网络, 验收项)**: 用历史真实**停牌/ST/涨跌停**股票回放验证降级路径
(experiments.md 待填表); ST/涨跌停识别的简称与 pct_change 字段随首次真实运行一并核对。

### ✅ Phase 6 — 输出与用户体验 L7+Web (离线部分完成, 2026-10-07)

**交付物**

| 任务 | 产出 | 状态 |
| --- | --- | --- |
| 6.1 报告增强 | track 摘要缓存 `data/track_summary.json`(新增 by_symbol 分标的块)→ predict/Web 展示全票池滚动20/60日 + 该股命中率; `output/scenarios.py` 预测分布区间(三分类概率×历史条件幅度 → p25/p50/p75 情景描述, **条件幅度剔除最后一行收益防前视, 有硬断言**); 市场状态已有(Phase 3) | ✅ |
| 6.2 用户自适应 | `predict --mode simple/pro`(默认 pro): simple = 方向+把握程度+一句话理由+白话风险提示; **验收"零术语黑名单词"通过**(已知警告白话映射+未知警告兜底不透传, 机制保证) | ✅ |
| 6.3 Web 界面 v1 | Streamlit `web` 服务(只读无账户): predict 编排抽到新模块 **`pipeline.py`**, CLI 与 Web 消费**同一个 report dict** —— 一致性验收由构造保证 + `test_web_consistency.py` 结构断言 | ✅ |
| 6.4 历史浏览 | Web 历史页: 埋点 → 结算 → 指标卡 + 累计/滚动命中率曲线(`accuracy_curve` 纯函数) + 分标的汇总表 + 明细 | ✅ |

**实现要点**(全录见 `docs/experiments.md`): 情景区间 = 三点混合近似(非收益分布承诺,
输出口径说明); track 摘要走缓存不在预测时重结算(predict 不额外触网); web 服务复用
同一镜像, compose 须用 `entrypoint` 整体覆盖(镜像 ENTRYPOINT 是 CLI, `command:`
会被当子命令 —— 已踩坑修复)。

**验证**: pytest **172 passed**(新增 21), ruff 全绿; 容器端到端: predict --json
(scenarios+track_summary 链路)→ track(摘要缓存+分标的表)→ predict 回填 →
`--mode simple`(容器内 find_jargon 0 命中)→ `compose up web`(health ok/主页 200/
bare 模式全脚本+真实数据渲染/结算/曲线全链路 0 异常)。验证数据已清理。

**⏳ 遗留(需本机网络)**: 真实数据下核对情景幅度合理性与 Web/CLI 一致性抽查;
`daily.sh` 例行刷新 track 摘要后报告数字与 track 报表一致性; simple 模式给真实
新手试读补充黑名单。

### ✅ Phase 7 — 运维自动化 L8+CI (离线部分完成, 2026-10-07)

**交付物**

| 任务 | 产出 | 状态 |
| --- | --- | --- |
| 7.1 特征漂移监控 | `monitor/drift.py`: PSI(近 20 日 vs 训练期基线, NaN 不进分箱 + 缺失率单边检查 + 空箱半样本平滑 + 基线版本闸门); 基线随 backtest/retrain 落盘并**与模型代际同归档**; warn/alert 落 `data/alerts.csvl`; `drift` 子命令退出码 2 = alert 级(自动化契约) | ✅ |
| 7.2 自动重训+回滚 | `model/retrain.py`: `retrain [--dry-run/--force/--tolerance]` 同口径回测 → **方向准确率不劣于现役才替换**(meta 新增 backtest 块); 替换前整代归档(工件+meta+基线) `data/models/archive/{ts}/`; `rollback [--list]` 非破坏式恢复; 采纳/拒绝/回滚全审计; `scripts/maintain.sh`(漂移触发 + 月度标记文件) | ✅ |
| 7.3 CI | `.github/workflows/ci.yml`: push/PR → ruff + pytest(离线), 与容器同口径; 静态校验进单测 | ✅ 待首次 push 验收绿灯 |
| 7.4 反馈收集 | `monitor/feedback.py`: `feedback` 子命令(记录/统计/负反馈复盘表) + Web 👍/👎 按钮, 共用 `append_feedback` 落 `data/feedback.csvl`(schema v1); Web 结果进 session_state | ✅ |

**实现要点**(全录见 `docs/experiments.md`): PSI 三处口径修正(NaN 单边检查防
覆盖改善误报 / 半样本平滑压噪声底 / 版本闸门防假绿); 有效样本量≈票数(少量票
判读保守, 阈值待真实校准); **顺带修复历史 bug: `__main__.py` 丢弃退出码**
(此前一切非零退出被吞成 0, daily.sh 失败分支从未生效)。另: 沙箱部分网络可达,
拿到 lhb/unlock/north 真实返回结构 —— 已录入 experiments.md 作为 Phase 2 联调线索。

**验证**: pytest **215 passed, 1 skipped**(新增 44), ruff 全绿; 容器端到端:
backtest(基线+meta.backtest) → drift(评估/分级/退出码 2/告警落盘/量能篡改检出)
→ retrain --dry-run(对比表+零写入) → retrain(采纳+整代归档+审计) →
rollback(恢复旧代+幂等) → maintain.sh 两分支(告警触发重训+月度标记 / 无告警跳过)
→ feedback(CLI 记录+复盘表) → web(health/主页/bare 全链路+web 反馈落盘)。
验证数据已清理。

**⏳ 遗留(需本机网络)**: PSI 阈值真实校准(观察数周水平类特征噪声底);
真实票池重训时长与月度节奏; **CI 绿灯待首次 push 验收**; Phase 2 联调清单
(沙箱观察已给出具体映射修正线索)。

---

## 九、收尾状态(2026-10-07)

**7 个 Phase 全部完成离线部分**, 路线图收官。剩余工作全部为"需本机网络/真实
数据"的验证与校准项, 汇总入口: `docs/experiments.md` 各 Phase 的
"真实数据验证(⏳)"小节 —— 建议执行顺序: Phase 2 联调清单(数据源地基,
沙箱已给出 lhb/unlock/north 修正线索) → Phase 4 回测对比表(集成去留裁决) →
Phase 3/5/6 验收项 → Phase 7 阈值校准 + CI 首推。
