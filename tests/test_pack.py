from affordability.cli import main


def test_demo_pack(tmp_path):
    out = tmp_path / "pack.html"
    main(["--demo", "--out", str(out)])
    html = out.read_text()
    assert "Affordability Evidence Pack" in html
    assert html.count("data:image/png;base64,") == 2
    assert (tmp_path / "borderline_declines.csv").exists()
