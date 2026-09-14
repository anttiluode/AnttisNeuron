from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_site_files_exist():
    for name in ("index.html", "styles.css", "app.js"):
        assert (SITE / name).is_file(), name


def test_site_uses_relative_assets_and_has_required_sections():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert 'href="styles.css"' in html
    assert 'src="app.js"' in html
    assert 'href="/styles.css"' not in html
    assert 'src="/app.js"' not in html
    for phrase in (
        "Wild hand waving. Executable receipts underneath.",
        "What is actually being claimed?",
        "Executable receipts",
        "Gate 6: the rule met other worlds",
        "Next falsification",
        "prior work",
        "exact",
        "synthetic",
        "open hypothesis",
    ):
        assert phrase in html


def test_receipts_are_runtime_loaded_not_authoritative_js_constants():
    js = (SITE / "app.js").read_text(encoding="utf-8")
    for name in ("gate2.json", "gate4.json", "gate5.json", "gate5b.json", "gate6.json"):
        assert f"results/{name}" in js
    # Headline scientific values should come from committed receipts, not stale JS literals.
    for forbidden in ("0.9001914311455128", "0.892400960888445", "0.0004983253901210636"):
        assert forbidden not in js
    assert "receipt unavailable" in js
    assert "receipt incomplete" in js


def test_receipts_have_fields_consumed_by_site():
    import json

    specs = {
        "gate2.json": ("raw_linear_accuracy", "square_accuracy", "mean_filter_alignment"),
        "gate4.json": ("fixed_alignment", "coupled_alignment", "delta_alignment"),
        "gate5.json": ("frozen_heldout_mean_alignment", "adaptive_heldout_mean_alignment", "adaptive_minus_frozen"),
        "gate5b.json": ("diagnostic_conditions", "slow_subspace_principal_angles_degrees"),
        "gate6.json": ("aggregate", "n_worlds", "alignment_metric"),
    }
    for filename, fields in specs.items():
        data = json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))
        for field in fields:
            assert field in data, (filename, field)


def test_pages_workflow_publishes_only_public_site_and_receipts():
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "cp -R site/. _site/" in workflow
    for name in ("gate2.json", "gate4.json", "gate5.json", "gate5b.json", "gate6.json"):
        assert f"results/{name}" in workflow
    assert "docs/superpowers" not in workflow
    assert "actions/deploy-pages" in workflow
