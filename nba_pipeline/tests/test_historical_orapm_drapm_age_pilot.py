import pandas as pd

from nba_pipeline.scripts.run_historical_orapm_drapm_age_pilot import main


def test_pilot_cli_reuses_annual_results_and_writes_curve_package(tmp_path):
    annual_dir = tmp_path / "annual"
    output_dir = tmp_path / "pilot"
    annual_dir.mkdir()
    ages = []
    for season in [2010, 2011]:
        rows = []
        for index in range(40):
            age = 19.0 + (index % 12) + (season - 2010)
            rows.append(
                {
                    "player_id": 1000 + index,
                    "player_name": f"Player {index}",
                    "off": 0.1 * index + 0.2 * (age - 27.0),
                    "def": -(0.05 * index - 0.1 * (age - 27.0)),
                    "off_poss": 1500,
                    "def_poss": 1500,
                }
            )
            ages.append(
                {
                    "nba_id": 1000 + index,
                    "season": season,
                    "date": f"{season}-05-01",
                    "age": age,
                }
            )
        pd.DataFrame(rows).to_csv(annual_dir / f"rapm_{season}_rs_results.csv", index=False)
    age_path = tmp_path / "ages.csv"
    pd.DataFrame(ages).to_csv(age_path, index=False)

    assert main(
        [
            "2010",
            "2011",
            "--annual-results-dir",
            str(annual_dir),
            "--age-source",
            str(age_path),
            "--output-dir",
            str(output_dir),
            "--skip-solves",
            "--smoothing-alpha",
            "0.01",
            "--bootstrap-reps",
            "0",
        ]
    ) == 0

    assert (output_dir / "annual_orapm_drapm_panel.csv").exists()
    assert (output_dir / "panel_receipt.json").exists()
    assert (output_dir / "age_curve.csv").exists()
    assert (output_dir / "fit_receipt.json").exists()

