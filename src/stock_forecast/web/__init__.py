"""Web 界面 v1 (Phase 6 / 6.3+6.4): Streamlit 只读展示, 无账户体系.

启动: `docker compose up web` → http://localhost:8501
预测编排复用 `pipeline.predict_once`(与 CLI 同一入口), Web 展示数据源 =
result["report"](CLI --json 输出的同一 dict), 一致性由构造保证。
"""
