#!/usr/bin/env bash
# 运维例行(Phase 7): 特征漂移检查 + 自动重训(月度例行 / 漂移告警触发)。
# 建议接在 daily.sh 之后运行(缓存已新鲜, drift 不额外触网)。
#
# 宿主 crontab 示例(每周一 18:00; 注意改成仓库绝对路径):
#   0 18 * * 1 cd /path/to/stock_forecast && docker compose run --rm --entrypoint bash sf /app/scripts/maintain.sh >> data/maintain.log 2>&1
#
# 重训触发条件(满足其一):
#   1. drift 退出码 2 = 存在 alert 级漂移(PSI >= drift_psi_alert);
#   2. 月度例行: 标记文件 data/.last_retrain_month 不是本月(无论 cron 频率如何,
#      每月首次运行必检一次)。重训成功(采纳或明确拒绝)才更新标记; 失败下月重试。
#
# 手动执行:
#   docker compose run --rm --entrypoint bash sf /app/scripts/maintain.sh
set -uo pipefail
cd /app

echo "=== $(date '+%F %T') drift ==="
python -m stock_forecast drift
rc=$?

need_retrain=0
if [ $rc -eq 2 ]; then
  echo "--- 检测到 alert 级漂移, 触发重训"
  need_retrain=1
fi

month=$(date +%Y-%m)
marker=/app/data/.last_retrain_month
if [ "$(cat "$marker" 2>/dev/null || true)" != "$month" ]; then
  echo "--- 本月($month)尚未重训, 触发月度例行重训"
  need_retrain=1
fi

if [ $need_retrain -eq 1 ]; then
  echo "=== $(date '+%F %T') retrain ==="
  if python -m stock_forecast retrain; then
    mkdir -p /app/data
    echo "$month" > "$marker"
  else
    echo "retrain 失败, 标记不更新(下次运行重试); 现役模型未受影响"
  fi
else
  echo "=== 无需重训(无漂移告警且本月已检) ==="
fi

exit 0
