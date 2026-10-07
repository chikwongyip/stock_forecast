"""CI 工作流文件校验 (Phase 7.3).

GitHub Actions 无法离线执行, 此处做静态校验: 文件存在、触发条件、
lint/test 步骤与 Python 版本口径(与 pyproject/Dockerfile 一致)。
"""

from pathlib import Path

import pytest

CI_FILE = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"


def test_ci_workflow_exists_with_required_steps():
    assert CI_FILE.exists(), "缺少 .github/workflows/ci.yml"
    text = CI_FILE.read_text(encoding="utf-8")
    for needle in ("actions/checkout", "actions/setup-python",
                   "ruff check src tests", "pytest -q", "3.12"):
        assert needle in text, f"CI 缺少关键内容: {needle}"


def test_ci_triggers_on_push_and_pr():
    text = CI_FILE.read_text(encoding="utf-8")
    assert "push" in text and "pull_request" in text


def test_ci_yaml_parses_if_available():
    yaml = pytest.importorskip("yaml", reason="pyyaml 未安装, 跳过结构解析")
    doc = yaml.safe_load(CI_FILE.read_text(encoding="utf-8"))
    assert "jobs" in doc
    job = next(iter(doc["jobs"].values()))
    runs = [s.get("run", "") for s in job["steps"]]
    assert any("ruff check" in r for r in runs)
    assert any("pytest" in r for r in runs)
