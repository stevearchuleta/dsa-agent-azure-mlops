import json

import pandas as pd

from dsa.cli import main


def test_cli_list_routes(capsys) -> None:
    exit_code = main(["--list-routes"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Supported routes:" in captured.out
    assert "- eda:" in captured.out
    assert "- sql:" in captured.out


def test_cli_missing_inputs_message(capsys) -> None:
    exit_code = main(["plot", "a", "histogram", "by", "target"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Category: plot" in captured.out
    assert "Dispatched: no" in captured.out
    assert "Missing inputs: df, value_col, target_col" in captured.out


def test_cli_json_output_for_sql_route(capsys) -> None:
    exit_code = main(["--json", "run", "a", "select", "query"])

    captured = capsys.readouterr()
    json_payload = json.loads(captured.out)

    assert exit_code == 0
    assert json_payload["route"]["category"] == "sql"
    assert json_payload["dispatched"] is False
    assert "database_path" in json_payload["missing_inputs"]


def test_cli_dispatches_eda_from_csv(tmp_path, capsys) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"age": [21, 22], "target": [0, 1]}).to_csv(
        csv_path,
        index=False,
    )

    exit_code = main(
        [
            "--csv",
            str(csv_path),
            "show",
            "dataframe",
            "overview",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Category: eda" in captured.out
    assert "Dispatched: yes" in captured.out
    assert '"row_count": 2' in captured.out