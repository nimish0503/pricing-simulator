import polars as pl
import pytest

from shelfsim.deploy import DEPLOY_DIR, check


def test_check_rejects_raw_rows(tmp_path):
    pl.DataFrame(
        {"store_id": ["1"], "product_id": ["a"], "week_start": ["2020-01-06"], "units": [3]}
    ).write_csv(tmp_path / "leak.csv")
    with pytest.raises(ValueError, match="store x product x week"):
        check(tmp_path)


def test_committed_deploy_folder_has_no_raw_rows():
    for client in [p for p in DEPLOY_DIR.iterdir() if p.is_dir()]:
        check(client)
