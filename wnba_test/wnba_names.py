"""Shared player-name lookup for the local WNBA pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_PATHS = [
    ROOT / "wnba_test" / "player_index_map.csv",
    ROOT
    / "nba_pipeline"
    / "external"
    / "gabriel_wnba"
    / "player_index_map.csv",
]
DEFAULT_NORMALIZED_DIRS = [
    ROOT / "wnba_test" / "GabrielNormalizedClean",
    ROOT / "wnba_test" / "GabrielNormalized",
]


def valid_name(value: object) -> str | None:
    name = str(value or "").strip()
    if not name or name.lower() == "nan":
        return None
    return name


def load_wnba_name_map(
    map_paths: list[Path] | None = None,
    normalized_dirs: list[Path] | None = None,
) -> dict[str, str]:
    names: dict[str, str] = {}
    candidate_maps = DEFAULT_MAP_PATHS if map_paths is None else map_paths
    candidate_normalized_dirs = (
        DEFAULT_NORMALIZED_DIRS if normalized_dirs is None else normalized_dirs
    )
    for path in candidate_maps:
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "EntityId" not in frame.columns:
            continue
        name_column = "pbp_name" if "pbp_name" in frame.columns else "ref_name"
        if name_column not in frame.columns:
            continue
        for player_id, player_name in zip(frame["EntityId"], frame[name_column]):
            try:
                key = str(int(float(player_id)))
            except (TypeError, ValueError):
                continue
            name = valid_name(player_name)
            if name is not None:
                names[key] = name

    for directory in candidate_normalized_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("WNBA*.csv")):
            frame = pd.read_csv(
                path,
                usecols=lambda column: column in {"personid", "playername"},
                low_memory=False,
            )
            if not {"personid", "playername"}.issubset(frame.columns):
                continue
            for player_id, player_name in zip(frame["personid"], frame["playername"]):
                try:
                    key = str(int(float(player_id)))
                except (TypeError, ValueError):
                    continue
                if key == "0":
                    continue
                name = valid_name(player_name)
                if name is not None:
                    names[key] = name
        if names:
            break
    return names
