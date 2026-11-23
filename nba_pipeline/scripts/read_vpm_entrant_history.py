#!/usr/bin/env python3
"""Read a dated entrant-aware VPM research board with explicit provenance.

This is a local inspection reader. The published historical reader is gated on
the separate full-feature source-graph and entrant-history audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results/estimated_skills/vpm_entrant_daily_full_v1"
AUDIT = Path(__file__).resolve().parents[1] / "results/estimated_skills/vpm_entrant_daily_full_audit_v1/receipt.json"
FIELDS = ["player_id", "player_name", "estimate_date", "season", "ovpm", "dvpm", "vpm",
    "own_prior_off", "own_prior_def", "background_prior_off", "background_prior_def",
    "combined_prior_off", "combined_prior_def", "effective_background_weight",
    "background_weight", "background_weight_identified", "career_nba_possessions",
    "entrant_career_exposure_complete", "active_season", "last_archive_positive_observation_date"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(root: Path, day: str, player_id: int | None, top: int, recent: bool) -> dict:
    cut = pd.Timestamp(day)
    if cut != pd.Timestamp(str(cut.date())) or not pd.Timestamp("1996-11-01") <= cut <= pd.Timestamp("2026-10-01"):
        raise ValueError("Use an exact historical date through October 1, 2026")
    folder = root / "cuts" / day
    if (folder / "cold.json").exists():
        return dict(status="unestimated_no_earlier_admitted_lineups", estimate_date=day, players=[])
    manifest = json.loads((folder / "manifest.json").read_text())
    meta = manifest["source_metadata"]
    if meta["estimate_date"] != day or meta["prior_regime"] != "full_feature_own_with_frozen_entrant_background":
        raise ValueError("Dated entrant board identity differs")
    for name, digest in manifest["output_hashes"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Dated entrant board output changed")
    frame = pd.read_parquet(folder / "ratings.parquet", columns=FIELDS)
    if len(frame) != manifest["rows"] or frame.player_id.duplicated().any():
        raise ValueError("Dated entrant player census differs")
    if np.max(abs(frame.ovpm + frame.dvpm - frame.vpm)) > 1e-9:
        raise ValueError("Dated entrant O/D/Total closure failed")
    if recent:
        last = pd.to_datetime(frame.last_archive_positive_observation_date)
        frame = frame.loc[last.ge(cut - pd.Timedelta(days=365))]
    if player_id is not None:
        frame = frame.loc[frame.player_id.eq(player_id)]
    else:
        frame = frame.sort_values(["vpm", "player_id"], ascending=[False, True]).head(top)
    rows = frame.astype(object).where(pd.notnull(frame), None).to_dict("records")
    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, pd.Timestamp):
                row[key] = value.isoformat()
            elif isinstance(value, np.generic):
                row[key] = value.item()
    status = "research_preview_pending_full_history_audit"
    if root == ROOT.resolve() and AUDIT.exists():
        audit = json.loads(AUDIT.read_text())
        if (audit.get("status") == "passed_independent_6108_date_entrant_numerical_audit_with_qualified_base_provenance"
                and audit.get("producer_receipt_sha256") == sha(root / "receipt.json")
                and audit.get("registration_sha256") == sha(root / "registration.json")):
            status = "research_history_numerically_audited_base_provenance_pending"
    return dict(status=status, estimate_date=day,
        prior_regime=meta["prior_regime"], entrant_model_fit_cut=meta["entrant_model_fit_cut"],
        entrant_model_backcast=meta["historical_entrant_model_backcast"],
        source_players=manifest["rows"], displayed_players=len(rows), players=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", required=True)
    parser.add_argument("--player-id", type=int)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--recent", action="store_true")
    args = parser.parse_args()
    if args.top <= 0:
        raise ValueError("Positive top count required")
    print(json.dumps(read(args.root.resolve(), args.date, args.player_id, args.top, args.recent),
        indent=2, allow_nan=False))
