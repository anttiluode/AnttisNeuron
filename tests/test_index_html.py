from pathlib import Path


def test_index_html_is_self_contained_and_scientifically_scaffolded():
    html = Path("index.html").read_text(encoding="utf-8")
    assert "Hand waving ends here. Executable claims begin here." in html
    for gate in (
        "Gate 0",
        "Gate 1",
        "Gate 2",
        "Gate 3",
        "Gate 4",
        "Gate 5",
        "Gate 5B",
        "Gate 6",
    ):
        assert gate in html
    lowered = html.lower()
    assert "<script src=" not in lowered
    assert 'rel="stylesheet" href=' not in lowered
    assert "video_url" not in lowered
    assert "https://www.youtube.com/embed/" not in lowered


def test_index_html_keeps_ais_claim_hypothetical():
    html = Path("index.html").read_text(encoding="utf-8")
    assert "AIS / output boundary" in html
    assert "future hypothesis" in html.lower()
