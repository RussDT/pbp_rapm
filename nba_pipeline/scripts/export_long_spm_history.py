"""Export actual FG+FT calibration rows on existing canonical RAPM boundaries.

This isolated source adapter recovers 2014/2015 warm-up seasons only. It does
not fit priors, choose models, read SQ estimates or overwrite canonical data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SLOTS = [f"{side}{i}" for side in ("O", "D") for i in range(1, 6)]
ALIGNMENT = ["game_id", "event_num", "game_date", *SLOTS]
OUTPUT_COLUMNS = [*ALIGNMENT, "actual_points", "source_season_end_year",
                  "season_phase", "source_row", "valid_lineup"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_game_ids(values):
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def actual_rows(target, canonical, season, phase):
    """Validate row identity before attaching an actual-only target vector."""
    if season not in (2014, 2015) or phase not in ("RS", "PS"):
        raise ValueError("Only registered 2014/2015 RS/PS calibration warm-ups")
    if len(target) != len(canonical):
        raise ValueError("Actual and canonical possession counts differ")
    for column in ALIGNMENT:
        if column == "game_id":
            a, b = (normalized_game_ids(f[column]).to_numpy() for f in (target, canonical))
        elif column == "game_date":
            a, b = (pd.to_datetime(f[column], utc=True).to_numpy() for f in (target, canonical))
        else:
            a, b = (pd.to_numeric(f[column], errors="raise").to_numpy() for f in (target, canonical))
        if not np.array_equal(a, b, equal_nan=column not in ("game_id", "game_date")):
            raise ValueError(f"Actual/canonical possession order mismatch: {column}")
    points = pd.to_numeric(target.Off_Diff, errors="raise").to_numpy(float)
    if not np.isfinite(points).all() or not np.allclose(points, np.rint(points), atol=1e-12, rtol=0):
        raise ValueError("Actual FG+FT target must be finite integer points")
    out = canonical[ALIGNMENT].copy()
    out["game_id"] = normalized_game_ids(out.game_id)
    out["game_date"] = pd.to_datetime(out.game_date, utc=True).dt.tz_localize(None)
    ids = out[SLOTS].apply(pd.to_numeric, errors="raise").to_numpy(float)
    valid = np.isfinite(ids).all(axis=1) & (ids > 0).all(axis=1) & (ids < 1610612737).all(axis=1)
    valid &= (np.diff(np.sort(ids, axis=1), axis=1) != 0).all(axis=1)
    out["actual_points"] = points
    out["source_season_end_year"] = season
    out["season_phase"] = phase
    out["source_row"] = np.arange(len(out), dtype=np.int64)
    out["valid_lineup"] = valid
    return out[OUTPUT_COLUMNS]


def export_season(pipeline, output, season, phase, processor=None):
    if season not in (2014, 2015) or phase not in ("RS", "PS"):
        raise ValueError("Only registered 2014/2015 RS/PS calibration warm-ups")
    pipeline, output = Path(pipeline), Path(output)
    suffix = "_PS" if phase == "PS" else ""
    raw = pipeline / "raw_data" / f"NBA{season % 100:02d}{suffix}.parquet"
    canonical_path = pipeline / "processed" / f"RAPM{season % 100:02d}{suffix}.parquet"
    package = pipeline / "scripts/process_rapm_blocks"
    inputs = {str(p.resolve()): sha(p) for p in
              [raw, canonical_path, Path(__file__), *sorted(package.glob("*.py"))]}
    dest = output / f"actual_{season}_{phase}.parquet"
    stamp = dest.with_suffix(".manifest.json")
    if dest.exists() or stamp.exists():
        if not (dest.exists() and stamp.exists()):
            raise ValueError("Incomplete warm-up export; use a fresh output directory")
        prior = json.loads(stamp.read_text())
        if prior["inputs"] != inputs or prior["artifact_sha256"] != sha(dest):
            raise ValueError("Frozen warm-up source/artifact changed; use a new version")
        return prior
    if processor is None:
        sys.path.insert(0, str(pipeline / "scripts"))
        processor = importlib.import_module("process_rapm_blocks.process_rapm")
    # Actual mode explicitly uses observed FT results. Disable the optional
    # provider merge as well so no whole-season shooter percentages are fetched.
    previous_fetch = processor.fetch_player_stats_supabase
    try:
        processor.fetch_player_stats_supabase = lambda *args, **kwargs: None
        result = processor.process_rapm_py(str(raw), season % 100,
            f"{season - 1}-{season % 100:02d}", o_luck=0, d_luck=0, ft_value_mode="actual")
    finally:
        processor.fetch_player_stats_supabase = previous_fetch
    if result is None:
        raise ValueError("Canonical actual-point processor returned no data")
    frame = actual_rows(result[0], pd.read_parquet(canonical_path), season, phase)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    receipt = dict(schema_version=1, season=season, phase=phase, rows=len(frame),
        games=int(frame.game_id.nunique()), valid_lineup_rows=int(frame.valid_lineup.sum()),
        first_game=str(frame.game_date.min().date()), last_game=str(frame.game_date.max().date()),
        actual_points=float(frame.actual_points.sum()), point_target="observed FG+FT, no luck adjustment",
        canonical_alignment_columns=ALIGNMENT, all_canonical_rows_retained=True,
        invalid_lineups="marked explicitly; consumer must filter valid_lineup before fitting",
        inputs=inputs, artifact_sha256=sha(dest), status="warmup_source_not_model",
        protected_season=2027)
    stamp.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "results/long_spm_actual_warmup_v1")
    ap.add_argument("--years", default="2014,2015")
    ap.add_argument("--phases", default="RS,PS")
    args = ap.parse_args()
    for year in map(int, args.years.split(",")):
        for phase in args.phases.split(","):
            print(json.dumps(export_season(args.pipeline_root, args.out, year, phase)), flush=True)
