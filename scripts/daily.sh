#!/usr/bin/env bash
# 每日例行(Phase 1.2): 批量预测默认票池 + 结算埋点。收盘后运行(建议 17:30 之后)。
#
# 宿主 crontab 示例(注意改成仓库绝对路径):
#   30 17 * * 1-5 cd /path/to/stock_forecast && docker compose run --rm --entrypoint bash sf /app/scripts/daily.sh >> data/daily.log 2>&1
#
# 手动执行:
#   docker compose run --rm --entrypoint bash sf /app/scripts/daily.sh
set -uo pipefail
cd /app

POOL=$(python -c "from stock_forecast.config import get_settings; print(' '.join(get_settings().stock_pool))")

fail=0
for sym in $POOL; do
  echo "=== $(date '+%F %T') predict $sym ==="
  # 单票失败(LLM 超时/数据缺失)不阻塞其余票, 记失败继续
  python -m stock_forecast predict "$sym" --json || { echo "predict $sym 失败"; fail=1; }
done

echo "=== $(date '+%F %T') track ==="
python -m stock_forecast track || fail=1

exit $fail
