"""Independent raw-event scoring audit for immutable long-SPM warm-up rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids(values):
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def compare_scores(raw, exported):
    """Price deduped raw made FGs/FTs independently of possession construction."""
    raw = raw.drop_duplicates().copy()
    raw["game_id"] = ids(raw.game_id)
    text = raw.home_description.fillna("") + " " + raw.visitor_description.fillna("")
    event = pd.to_numeric(raw.event_type, errors="raise")
    points = np.where(event.eq(1), np.where(text.str.contains("3PT", case=False), 3, 2),
        np.where(event.eq(3) & ~text.str.contains("MISS", case=False), 1, 0))
    expected = pd.Series(points, index=raw.index).groupby(raw.game_id).sum()
    exported = exported.copy()
    exported["game_id"] = ids(exported.game_id)
    actual = exported.groupby("game_id").actual_points.sum()
    all_ids = sorted(set(expected.index) | set(actual.index))
    rows = []
    for game in all_ids:
        raw_points = None if game not in expected else float(expected[game])
        canonical_points = None if game not in actual else float(actual[game])
        difference = None if raw_points is None or canonical_points is None else canonical_points - raw_points
        rows.append(dict(game_id=game, raw_event_points=raw_points,
            canonical_actual_points=canonical_points, difference=difference,
            accepted=difference == 0))
    return pd.DataFrame(rows)


def audit(pipeline, root):
    pipeline, root = Path(pipeline), Path(root)
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    records, checked = [], []
    for year in (2014, 2015):
        for phase in ("RS", "PS"):
            suffix = "_PS" if phase == "PS" else ""
            raw = pipeline / "raw_data" / f"NBA{year % 100:02d}{suffix}.parquet"
            exported = root / f"actual_{year}_{phase}.parquet"
            manifest_path = exported.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            if sha(exported) != manifest["artifact_sha256"]:
                raise ValueError("Export artifact changed")
            for path, expected in manifest["inputs"].items():
                if sha(path) != expected:
                    raise ValueError(f"Export source changed: {path}")
            canonical = pipeline / "processed" / f"RAPM{year % 100:02d}{suffix}.parquet"
            for path in (raw, canonical, exported, manifest_path):
                sources[str(path.resolve())] = sha(path)
            table = compare_scores(pd.read_parquet(raw), pd.read_parquet(exported))
            table["season"] = year
            table["phase"] = phase
            checked.extend(table.to_dict(orient="records"))
            records.append(dict(season=year, phase=phase, checked_games=len(table),
                coherent_games=int(table.accepted.sum()),
                quarantined_games=int((~table.accepted).sum())))
    bad = [r for r in checked if not r["accepted"]]
    result = dict(schema_version=1, status="independent_actual_game_scoring_gate",
        checked_games=len(checked), coherent_games=len(checked) - len(bad),
        quarantined_game_ids=sorted({r["game_id"] for r in bad}),
        mismatches=bad, seasons=records, sources=sources,
        raw_scoring="exact-row dedupe; madeFG=2or3 from event1; madeFT=1 from event3 excluding MISS",
        quarantine_contract="exclude listed games from both calibration likelihood and matching-period feature aggregation",
        export_manifests_unchanged=True, protected_season=2027)
    destination = root / "score_accounting.json"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise ValueError("Frozen accounting audit changed; use a new audit version")
    else:
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-root", type=Path, default=ROOT)
    ap.add_argument("--root", type=Path, default=ROOT / "results/long_spm_actual_warmup_v1")
    args = ap.parse_args()
    receipt = audit(args.pipeline_root, args.root)
    print(json.dumps({k: receipt[k] for k in
        ("checked_games", "coherent_games", "quarantined_game_ids", "seasons")}))
