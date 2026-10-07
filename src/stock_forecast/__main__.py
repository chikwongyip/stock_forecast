import sys

from stock_forecast.cli import main

if __name__ == "__main__":
    # 退出码必须透传(Phase 7 发现的历史 bug: 此前丢弃 main() 返回值,
    # 一切非零退出码被吞成 0, daily.sh/maintain.sh 的失败分支全部失效)
    sys.exit(main())
