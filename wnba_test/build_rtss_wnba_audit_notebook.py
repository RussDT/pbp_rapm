#!/usr/bin/env python3
"""Build and execute the reproducible RTSS WNBA feature-audit notebook."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


DEFAULT_INPUT = Path("wnba_test/Results/rtss_feature_audit")
DEFAULT_NOTEBOOK = Path("wnba_test/analysis/rtss_wnba_feature_audit.ipynb")


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_path = args.output.resolve()
    input_dir = args.input_dir.resolve()
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
    nb["cells"] = [
        markdown(
            """
            # WNBA RTSS Feature Audit

            Reproducible review of the local 2023–2025 rich WNBA markings
            export. The notebook reads only compact audit artifacts generated
            by `wnba_test/audit_rtss_wnba_features.py`; it does not load the
            multi-gigabyte source again.

            Modeling rule: provider shot-quality outputs remain benchmarks,
            and post-shot fields remain targets/diagnostics. They are not
            eligible pre-shot predictors.
            """
        ),
        code(
            f"""
            from pathlib import Path
            import json
            import pandas as pd
            from IPython.display import display

            INPUT = Path({str(input_dir)!r})
            manifest = json.loads((INPUT / "audit_manifest.json").read_text())
            inventory = pd.read_csv(INPUT / "game_inventory.csv")
            tables = pd.read_csv(INPUT / "table_coverage.csv")
            parity = pd.read_csv(INPUT / "two_d_three_d_parity.csv")
            columns = pd.read_csv(INPUT / "column_coverage.csv")
            column_catalog = pd.read_csv(INPUT / "column_modeling_catalog.csv")
            lineups = pd.read_csv(INPUT / "lineup_quality.csv")
            chance_players = pd.read_csv(INPUT / "chance_player_quality.csv")
            shots = pd.read_csv(INPUT / "shot_feature_coverage.csv")
            identities = pd.read_csv(INPUT / "identity_coverage.csv")
            catalog = pd.read_csv(INPUT / "feature_catalog.csv")
            manifest
            """
        ),
        markdown("## Coverage and game inventory"),
        code(
            """
            game_types = (
                inventory.groupby(["season", "game_type_code"], as_index=False)
                .agg(games=("game_id", "nunique"))
                .sort_values(["season", "game_type_code"])
            )
            display(game_types)
            display(
                inventory.groupby("season", as_index=False).agg(
                    games=("game_id", "nunique"),
                    possessions=("possessions", "sum"),
                    chances=("chances", "sum"),
                    shots=("shots", "sum"),
                    passes=("passes", "sum"),
                    touches=("touches", "sum"),
                )
            )
            """
        ),
        markdown("## Lineup integrity and identity crosswalk"),
        code(
            """
            display(lineups.round(6))
            display(chance_players.round(6))
            display(identities.round(6))
            """
        ),
        markdown("## Major event-table density"),
        code(
            """
            key_tables = [
                "possessions", "chances", "shots", "passes", "touches",
                "picks", "off_ball_screens", "rotations", "poss_matchups",
                "shot_contests", "boxout_opportunities", "defender_stance",
            ]
            density = (
                tables[tables["table"].isin(key_tables)]
                .pivot(index="table", columns="season", values="rows_per_game")
                .reindex(key_tables)
            )
            display(density.round(1))
            """
        ),
        markdown("## 2D base versus 3D enrichment parity"),
        code(
            """
            parity_key = parity[parity["table"].isin([
                "possessions", "chances", "shots", "passes", "touches",
                "picks", "rotations", "poss_matchups", "shot_contests",
            ])]
            display(parity_key[[
                "season", "table", "games_2d_nonempty", "games_3d_nonempty",
                "row_count_equal_pct", "common_id_pct_of_2d", "common_id_pct_of_3d",
            ]].round(6))
            """
        ),
        markdown("## Shot-context coverage"),
        code(
            """
            pct_columns = [c for c in shots if c.endswith("_coverage")]
            display(shots[["season", "shots", *pct_columns]].round(6))
            """
        ),
        markdown("## Schema drift candidates"),
        code(
            """
            drift = (
                columns.pivot_table(
                    index=["table", "column"],
                    columns="season",
                    values="coverage",
                    aggfunc="max",
                )
                .reset_index()
            )
            season_cols = [c for c in drift.columns if isinstance(c, int)]
            if not season_cols:
                season_cols = [c for c in drift.columns if str(c).isdigit()]
            drift["coverage_range"] = drift[season_cols].max(axis=1) - drift[season_cols].min(axis=1)
            display(drift.sort_values("coverage_range", ascending=False).head(40).round(6))
            """
        ),
        markdown("## Recommended build order"),
        code(
            """
            display(catalog.sort_values(["priority", "family"])[
                ["priority", "family", "grain", "model_role", "source_tables", "notes"]
            ])
            """
        ),
        markdown("## Column-level modeling roles"),
        code(
            """
            display(
                column_catalog.groupby(
                    ["modeling_role", "recommended_use"], as_index=False
                ).agg(columns=("column", "size"))
                .sort_values("columns", ascending=False)
            )
            display(
                column_catalog[
                    column_catalog["modeling_role"].isin(
                        ["provider benchmark", "target / realized outcome", "post-release / pose diagnostic"]
                    )
                ][["table", "column", "modeling_role", "recommended_use", "reason"]]
                .sort_values(["modeling_role", "table", "column"])
                .head(80)
            )
            """
        ),
        markdown(
            """
            ## Decision

            The data is suitable for a new WNBA enrichment layer, but it should
            not be poured directly into the current impact metric.

            1. Build exact game/event/player crosswalks and a possession/chance
               fact model.
            2. Replace 2023–2025 lineup exposure with the rich five-on-five
               source only after possession and stint validation.
            3. Train MID/3P context-only expected-points models with
               season-forward, game-grouped evaluation.
            4. Add separate defensive contest/rotation, creation/screening, and
               boxout/rebounding summaries as prior-game-only player features.
            5. Admit a feature family to VPM/box priors only if it improves an
               untouched future season beyond the existing no-prior and
               accepted-prior lanes.
            """
        ),
    ]

    nbf.write(nb, notebook_path)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    executed = client.execute()
    nbf.write(executed, notebook_path)
    print(notebook_path)


if __name__ == "__main__":
    main()
