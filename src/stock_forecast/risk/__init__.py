"""L6 风控与合规层 (Phase 5).

三个纯函数模块(无 IO, 离线可单测):
- `status`: 交易状态识别(停牌/涨跌停/ST)→ 适用性降级(不适用/仅供参考);
- `extreme`: 极端场景扩充(连续一字板/近20日振幅异常/量能枯竭)→ 风险提示;
- `compliance`: 输出前合规校验(免责声明存在/无买卖措辞/置信度与警告一致),
  违规阻断输出并记 `data/compliance_log.jsonl`。

status 与 extreme 的判定结果经 `fusion.combine.fuse` 融合进最终研判;
compliance 是 render 之前的最后一道闸门(见 cli.run_predict)。
"""
