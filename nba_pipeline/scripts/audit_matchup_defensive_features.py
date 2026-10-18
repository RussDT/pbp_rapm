"""Independent aggregation, source and date audit of matchup skill summaries."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIT_SCALE_ONE = {"matchup_assignment_bpm", "matchup_assignment_obpm", "matchup_assignment_pos_difficulty",
    "matchup_primary_share", "matchup_concentration", "matchup_coverage",
    "matchup_guard_share", "matchup_wing_share", "matchup_big_share"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_snapshot_values(atoms, board, feature, cutoff):
    """Independent weighted sums, population prior and zero-coverage mask."""
    before = atoms.loc[pd.to_datetime(atoms.game_date).lt(pd.Timestamp(cutoff))]
    numerator, denominator = feature + "__numerator", feature + "__denominator"
    if numerator not in before or denominator not in before:
        if board[feature].notna().any():
            raise ValueError("Feature has values without count atoms")
        return 0.
    weight = np.power(2., (pd.to_datetime(before.game_date) - pd.Timestamp(cutoff)).dt.total_seconds().to_numpy() / (550 * 86400))
    counts = pd.DataFrame(dict(player_id=before.player_id,
        num=before[numerator].to_numpy() * weight, den=before[denominator].to_numpy() * weight))
    totals = counts.groupby("player_id")[["num", "den"]].sum()
    population = counts.num.sum() / counts.den.sum() if counts.den.sum() > 0 else np.nan
    scale = 1. if feature in UNIT_SCALE_ONE else 100.
    values = totals.reindex(board.player_id)
    expected = scale * (values.num.to_numpy() + 1000 * population) / (values.den.to_numpy() + 1000)
    if feature == "matchup_coverage":
        # Coverage measures acquisition/exposure, not player talent. It retains
        # the full official denominator and has no population pseudo-evidence.
        expected = values.num.to_numpy() / values.den.replace(0, np.nan).to_numpy()
    expected[values.den.le(0).to_numpy()] = np.nan
    actual = board[feature].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0, equal_nan=True)
    if feature + "__exposure" in board:
        np.testing.assert_allclose(board[feature + "__exposure"], values.den, atol=1e-8, rtol=0)
    finite = np.isfinite(expected)
    return float(np.max(np.abs(expected[finite] - actual[finite]), initial=0))


def audit(output):
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for filename, expected in manifest["source_hashes"].items():
        if sha(filename) != expected:
            raise ValueError(f"Source changed: {filename}")
    for filename, expected in manifest["artifact_hashes"].items():
        if sha(output / filename) != expected:
            raise ValueError(f"Feature artifact changed: {filename}")
    metadata_lineage = []
    for filename in manifest["source_hashes"]:
        path = Path(filename)
        if path.name != "player_seasons.parquet" or not (path.parent / "manifest.json").exists():
            continue
        source_manifest = json.loads((path.parent / "manifest.json").read_text())
        for source, expected in source_manifest.get("sources", {}).items():
            if sha(source) != expected:
                raise ValueError("Opponent BPM upstream source changed: " + source)
        if source_manifest.get("artifacts", {}).get(path.name) != sha(path):
            raise ValueError("Opponent BPM source manifest does not match feature bytes")
        metadata_lineage.append(dict(manifest_path=str(path.parent / "manifest.json"),
            manifest_sha256=sha(path.parent / "manifest.json"), rows=source_manifest["rows"],
            unmatched_exact_rows=source_manifest["unmatched_rows"], publication_vintage_certified=False))
    builder = ROOT / "scripts/build_matchup_defensive_features.py"
    if sha(builder) != manifest["code_sha256"]:
        raise ValueError("Feature construction code changed after export")
    features = pd.read_parquet(output / "features.parquet")
    atoms = pd.read_parquet(output / "defender_game_atoms.parquet")
    provenance = pd.read_parquet(output / "feature_provenance.parquet")
    coverage = pd.read_parquet(output / "game_coverage.parquet")
    names = sum(manifest["feature_groups"].values(), [])
    if features.duplicated(["player_id", "estimate_date"]).any():
        raise ValueError("Duplicate player-cut summary")
    if atoms.duplicated(["game_id", "player_id"]).any() or not atoms.season_type.isin(["rs", "ps"]).all():
        raise ValueError("Invalid defender-game atom grain/phase")
    if atoms.season.max() >= 2027 or pd.to_datetime(features.estimate_date).ge("2026-10-01").any():
        raise ValueError("Protected2027 evidence/emission")
    for date in ["last_observation_date", "parameter_fit_end"]:
        present = provenance[date].notna()
        if not pd.to_datetime(provenance.loc[present, date]).lt(pd.to_datetime(provenance.loc[present, "estimate_date"])).all():
            raise ValueError("Feature observation/parameter date reaches cutoff")
    if provenance.duplicated(["player_id", "estimate_date", "feature"]).any():
        raise ValueError("Duplicate per-feature provenance")
    maximum = 0.
    checked = 0
    for cut, board in features.groupby("estimate_date", sort=True):
        expected_players = set(atoms.loc[pd.to_datetime(atoms.game_date).lt(cut), "player_id"])
        if set(board.player_id) != expected_players:
            raise ValueError("Snapshot player universe differs from prior atoms")
        for name in names:
            maximum = max(maximum, independent_snapshot_values(atoms, board, name, cut))
            checked += len(board)
    # Complete row identities apply only where all three coverage masks match.
    columns = {name: name + "__denominator" for name in ["matchup_points_saved100", "matchup_fg_points_saved100", "matchup_ft_points_saved100"]}
    same = (atoms[columns["matchup_points_saved100"]].eq(atoms[columns["matchup_fg_points_saved100"]])
            & atoms[columns["matchup_points_saved100"]].eq(atoms[columns["matchup_ft_points_saved100"]])
            & atoms[columns["matchup_points_saved100"]].gt(0))
    gaps = (atoms.matchup_points_saved100__numerator
            - atoms.matchup_fg_points_saved100__numerator - atoms.matchup_ft_points_saved100__numerator)
    np.testing.assert_allclose(gaps.loc[same], 0, atol=1e-9, rtol=0)
    # FG and FT defects must disable their own lanes, not create zero talent.
    if "ft_structurally_missing" in coverage and coverage.loc[coverage.ft_structurally_missing, "ft_valid"].any():
        raise ValueError("Structurally absent FT rows admitted")
    unit_checks = {}
    for feature in ["matchup_primary_share", "matchup_concentration", "matchup_guard_share", "matchup_wing_share", "matchup_big_share"]:
        values = features[feature].dropna()
        if not values.between(0, 1).all():
            raise ValueError("Invalid assignment fraction: " + feature)
        unit_checks[feature] = dict(observed=len(values), minimum=float(values.min()) if len(values) else None,
                                   maximum=float(values.max()) if len(values) else None)
    shares = features[["matchup_guard_share", "matchup_wing_share", "matchup_big_share"]].dropna()
    np.testing.assert_allclose(shares.sum(axis=1), 1., atol=1e-10, rtol=0)
    # Independent future-append test on the arithmetic seam, retaining the
    # entire player/day state. Appended future atoms cannot change earlier cuts.
    first = pd.Timestamp(features.estimate_date.min())
    board = features.loc[pd.to_datetime(features.estimate_date).eq(first)]
    future = atoms.head(min(50, len(atoms))).copy()
    future["game_date"] = first + pd.Timedelta(days=30)
    for column in future:
        if column.endswith("__numerator"):
            future[column] = 1e12
    appended = pd.concat([atoms, future], ignore_index=True)
    for name in names:
        independent_snapshot_values(appended, board, name, first)
    result = dict(status="passed", feature_rows=len(features), atom_rows=len(atoms),
        independent_feature_cells=checked, snapshot_max_gap=maximum,
        feature_count=len(names), cuts=features.estimate_date.astype(str).unique().tolist(),
        common_mask_point_closure_rows=int(same.sum()),
        future_append_snapshot_invariant=True, fractions=unit_checks,
        opponent_metadata_lineage=metadata_lineage,
        source_manifest_sha256=sha(manifest_path), validation_code_sha256=sha(__file__),
        source_hashes=manifest["source_hashes"], feature_status=manifest["status"],
        limitations=["Partial archive outputs remain smoke tests, not predictive evidence.",
            "Snapshot invariance does not independently refit the raw opponent-pricing model."], protected_outcomes=0)
    (output / "independent_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "feature_rows", "atom_rows", "independent_feature_cells", "snapshot_max_gap", "feature_status"]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
