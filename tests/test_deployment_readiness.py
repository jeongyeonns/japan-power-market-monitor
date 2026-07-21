from pathlib import Path

from utils.eprx_loader import find_eprx_files, load_all_eprx_data
from utils.jepx_loader import find_jepx_files, load_all_jepx_data


def test_missing_market_directories_return_empty_results(tmp_path):
    missing_eprx = tmp_path / "missing-eprx"
    assert find_eprx_files(missing_eprx) == []
    eprx_data, eprx_errors, eprx_summary = load_all_eprx_data(missing_eprx)
    assert eprx_data.empty and eprx_errors.empty and eprx_summary.empty

    missing_jepx = tmp_path / "missing-jepx"
    assert find_jepx_files(missing_jepx).empty
    jepx_long, jepx_wide, errors, warnings, summary = load_all_jepx_data(missing_jepx)
    assert all(frame.empty for frame in (jepx_long, jepx_wide, errors, warnings, summary))


def test_empty_and_invalid_files_do_not_raise_uncaught_errors(tmp_path):
    (tmp_path / "empty.csv").write_bytes(b"")
    (tmp_path / "invalid.csv").write_text("not,a,market,file\n1,2,3,4", encoding="utf-8")
    eprx_data, _, eprx_summary = load_all_eprx_data(tmp_path)
    assert eprx_data.empty
    assert not eprx_summary.empty
    jepx_long, _, errors, _, jepx_summary = load_all_jepx_data(tmp_path)
    assert jepx_long.empty
    assert not jepx_summary.empty
    assert not errors.empty or jepx_summary["status"].ne("정상").any()


def test_deployment_files_and_secret_ignores_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "runtime.txt").read_text(encoding="utf-8").strip() == "python-3.14"
    assert (root / ".streamlit" / "config.toml").is_file()
    ignores = (root / ".gitignore").read_text(encoding="utf-8")
    for entry in (".streamlit/secrets.toml", ".env", ".pytest_cache/", "data/jepx/raw/*"):
        assert entry in ignores
