FROM python:3.12-slim

# 时区设为上海, 与 A 股交易日历一致; pip 走清华镜像源
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 依赖层单独缓存: 先用空壳包只装依赖, 源码变更时不重装
COPY pyproject.toml ./
RUN mkdir -p src/stock_forecast && touch src/stock_forecast/__init__.py \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir pytest ruff

# 真实源码; 运行时 compose 会把 ./src 挂载覆盖, 实现免重建热更新
COPY src ./src

WORKDIR /app
ENV PYTHONPATH=/app/src

RUN mkdir -p /app/data

ENTRYPOINT ["python", "-m", "stock_forecast"]
