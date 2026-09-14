from pathlib import Path


def test_gate7_receipt_is_immutable_ci_input():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in workflow
    assert "git push" not in workflow
    assert "git commit" not in workflow
    assert "if [ -f results/gate7.json ]" not in workflow
    assert "python experiments/gate7_load_compensation.py --seed 17 --worlds 24 --out /tmp/gate7.json" in workflow
    assert "cmp /tmp/gate7.json results/gate7.json" in workflow
