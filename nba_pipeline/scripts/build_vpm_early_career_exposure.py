#!/usr/bin/env python3
"""Build dated 1997–2013 NBA career exposure from the original ten-player rows.

This is an entrant-prior source ledger, not a new impact fit. It uses exactly the
same valid-lineup filter as the saved `nba_history.parquet` and verifies every
player/season/phase total against that archive before writing a new result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
RAPM = HERE.parents[2]
VPM = RAPM.parent / "vpm"
sys.path.insert(0, str(VPM))

from vpm.harness.entrant_priors import SLOTS, raw_season  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def one_season(pipeline: Path, year: int, playoffs: bool) -> tuple[pd.DataFrame, dict]:
    raw = pipeline / "processed" / f"RAPM{year % 100:02d}{'_PS' if playoffs else ''}.parquet"
    frame = raw_season(pipeline, year, playoffs)
    if frame is None:
        return pd.DataFrame(), {}
    if frame.empty:
        raise ValueError(f"Empty original valid-lineup source: {raw}")
    if frame.game_date.isna().any() or frame.game_id.isna().any():
        raise ValueError(f"Missing game identity/date: {raw}")
    if frame.groupby("game_id").game_date.nunique().gt(1).any():
        raise ValueError(f"Game has multiple dates: {raw}")
    parts: list[pd.DataFrame] = []
    for start in range(0, len(frame), 50_000):
        block = frame.iloc[start : start + 50_000]
        n = len(block)
        long = pd.DataFrame(
            {
                "player_id": block[SLOTS].to_numpy(np.int64).reshape(-1),
                "game_id": np.repeat(block.game_id.to_numpy(), len(SLOTS)),
                "game_date": np.repeat(block.game_date.to_numpy(), len(SLOTS)),
            }
        )
        counts = long.groupby(["player_id", "game_id", "game_date"], sort=False).size()
        parts.append(counts.rename("slots").reset_index())
        if len(long) != 10 * n:
            raise ValueError("Ten-player slot expansion changed row count")
    out = pd.concat(parts, ignore_index=True)
    out = out.groupby(["player_id", "game_id", "game_date"], sort=False).slots.sum().reset_index()
    out["possessions"] = out.pop("slots").astype(np.float64) / 2.0
    out.insert(0, "season", year)
    out.insert(1, "playoffs", playoffs)
    if out.possessions.sum() != 5 * len(frame):
        raise ValueError(f"Original ten-slot exposure mass failed: {raw}")
    return out, {str(raw.resolve()): sha(raw)}


def build(out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Preserve existing exposure ledger: {out}")
    pipeline = RAPM / "nba_pipeline"
    history_path = VPM / "vpm/data/entrants/nba_history.parquet"
    history = pd.read_parquet(history_path)
    parts: list[pd.DataFrame] = []
    pins = {str(history_path.resolve()): sha(history_path), str(HERE): sha(HERE)}
    seasons = []
    for year in range(1997, 2014):
        for playoffs in (False, True):
            part, source_pin = one_season(pipeline, year, playoffs)
            if part.empty:
                continue
            parts.append(part)
            pins.update(source_pin)
            seasons.append({"season": year, "playoffs": playoffs, "player_games": len(part)})
    ledger = pd.concat(parts, ignore_index=True)
    keys = ["season", "playoffs", "player_id"]
    got = ledger.groupby(keys, sort=True).possessions.sum().sort_index()
    expected = history.loc[history.season.between(1997, 2013)].groupby(keys, sort=True).possessions.sum().sort_index()
    if not got.index.equals(expected.index) or not np.array_equal(got.to_numpy(), expected.to_numpy()):
        raise ValueError("Dated original rows do not reproduce every saved annual career exposure total")
    if ledger.duplicated(keys + ["game_id"]).any():
        raise ValueError("Duplicate player-game exposure identity")
    ledger = ledger.sort_values(["game_date", "season", "playoffs", "game_id", "player_id"]).reset_index(drop=True)
    if ledger.game_date.isna().any() or not np.isfinite(ledger.possessions).all():
        raise ValueError("Nonfinite dated career exposure")
    out.mkdir(parents=True)
    data = out / "player_game_exposure.parquet"
    ledger.to_parquet(data, index=False, compression="zstd")
    pd.testing.assert_frame_equal(pd.read_parquet(data), ledger, check_exact=True)
    if any(sha(Path(p)) != digest for p, digest in pins.items()):
        raise ValueError("Original career source changed while building ledger")
    receipt = {
        "status": "passed_exact_annual_reconciliation",
        "rows": len(ledger),
        "players": int(ledger.player_id.nunique()),
        "games": int(ledger[["season", "playoffs", "game_id"]].drop_duplicates().shape[0]),
        "seasons": seasons,
        "first_game_date": str(ledger.game_date.min()),
        "last_game_date": str(ledger.game_date.max()),
        "input_sha256": pins,
        "ledger_sha256": sha(data),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = build(parser.parse_args().out.resolve())
    print(json.dumps({k: result[k] for k in ["status", "rows", "players", "games", "first_game_date", "last_game_date"]}))
