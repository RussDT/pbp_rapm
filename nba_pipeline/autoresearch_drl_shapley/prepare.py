#!/usr/bin/env python3
"""
Fixed prepare/evaluation harness for autoresearch-style DRL/Shapley experiments.

Only `train.py` should be modified during research loops.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
SCRIPTS_DIR = PIPELINE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_drl_shapley as base  # noqa: E402


TRAIN_START = 21
TRAIN_END = 26
OUTPUT_YEAR = 26
SEASON_TYPE = "RS"
SEED = 7
VALIDATION_FRAC = 0.20
RESEARCH_VALID_FRAC = 0.50
MAX_GAMES_PER_SEASON = 20

SCORE_WEIGHTS = {
    "rmse": 0.55,
    "brier": 0.25,
    "logloss": 0.20,
}
TARGETS = {
    "research_score": 0.95,
    "audit_score": 0.98,
    "rmse_vs_tree": 0.85,
    "brier_vs_logistic": 1.05,
    "logloss_vs_logistic": 1.10,
}

CACHE_DIR = HERE / "cache"
BUNDLE_PATH = CACHE_DIR / "prepared_bundle.pkl"
SPLITS_PATH = CACHE_DIR / "prepared_splits.npz"
CONTEXT_PATH = CACHE_DIR / "prepared_context.json"
RESULTS_TSV = HERE / "results.tsv"
LATEST_SUMMARY = HERE / "latest_summary.json"


@dataclass
class PreparedExperiment:
    bundle: base.ArrayBundle
    train_idx: np.ndarray
    research_valid_idx: np.ndarray
    audit_valid_idx: np.ndarray
    research_baselines: Dict[str, Dict[str, float]]
    audit_baselines: Dict[str, Dict[str, float]]
    context: Dict[str, object]


def build_args(rebuild_dataset: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        output_year=OUTPUT_YEAR,
        season_type=SEASON_TYPE,
        seed=SEED,
        batch_size=512,
        epochs=2,
        attributor_epochs=1,
        allocator_epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        validation_frac=VALIDATION_FRAC,
        patience=2,
        hidden_size=128,
        embedding_dim=64,
        lineup_dim=20,
        num_attn_heads=8,
        max_games_per_season=MAX_GAMES_PER_SEASON,
        max_shapley_states=64,
        shapley_permutations=8,
        permutation_tests=100,
        min_pair_possessions=500,
        leaderboard_min_possessions=500,
        report_shrinkage_k=1500.0,
        device="auto",
        rebuild_dataset=rebuild_dataset,
        skip_baselines=False,
    )


def ensure_results_tsv() -> None:
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "timestamp\tcommit\tresearch_score\taudit_score\trmse\tbrier\tlogloss\tstatus\tdescription\n"
    )


def split_validation_indices(bundle: base.ArrayBundle, valid_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_dates = np.unique(bundle.game_date_ord[valid_idx])
    valid_dates.sort()
    if valid_dates.size >= 2:
        research_count = max(1, int(np.floor(valid_dates.size * RESEARCH_VALID_FRAC)))
        research_count = min(research_count, valid_dates.size - 1)
        research_dates = set(valid_dates[:research_count].tolist())
        audit_dates = set(valid_dates[research_count:].tolist())
        research_idx = valid_idx[np.isin(bundle.game_date_ord[valid_idx], list(research_dates))]
        audit_idx = valid_idx[np.isin(bundle.game_date_ord[valid_idx], list(audit_dates))]
    else:
        valid_games = np.unique(bundle.game_id[valid_idx])
        valid_games.sort()
        research_count = max(1, int(np.floor(valid_games.size * RESEARCH_VALID_FRAC)))
        research_count = min(research_count, max(1, valid_games.size - 1))
        research_games = set(valid_games[:research_count].tolist())
        audit_games = set(valid_games[research_count:].tolist())
        research_idx = valid_idx[np.isin(bundle.game_id[valid_idx], list(research_games))]
        audit_idx = valid_idx[np.isin(bundle.game_id[valid_idx], list(audit_games))]

    if research_idx.size == 0 or audit_idx.size == 0:
        midpoint = max(1, valid_idx.size // 2)
        research_idx = valid_idx[:midpoint]
        audit_idx = valid_idx[midpoint:]
    return research_idx.astype(np.int64), audit_idx.astype(np.int64)


def normalize_bundle_maps(bundle: base.ArrayBundle, player_name_map, player_team_map, player_pos_group) -> None:
    lookup, vocab = base.build_player_lookup(bundle)
    base.remap_player_arrays(bundle, lookup)
    bundle.player_vocab = vocab
    bundle.player_name_map = {lookup.get(pid, 0): name for pid, name in player_name_map.items() if pid in lookup}
    bundle.player_team_map = {lookup.get(pid, 0): team for pid, team in player_team_map.items() if pid in lookup}
    bundle.player_pos_group = {lookup.get(pid, 0): pos for pid, pos in player_pos_group.items() if pid in lookup}


def prepare_experiment(rebuild_dataset: bool = False) -> PreparedExperiment:
    ensure_results_tsv()
    args = build_args(rebuild_dataset=rebuild_dataset)
    base.seed_everything(args.seed)

    player_meta = base.fetch_player_metadata(args.train_start, args.train_end)
    player_name_map, player_team_map, player_pos_group, _pos_weights_map, aliases = base.build_player_maps(player_meta)
    transition_df = base.load_or_build_dataset(args, aliases)
    bundle = base.materialize_arrays(
        transition_df,
        output_year=args.output_year,
        player_name_map=player_name_map,
        player_team_map=player_team_map,
        player_pos_group=player_pos_group,
    )
    normalize_bundle_maps(bundle, player_name_map, player_team_map, player_pos_group)

    train_idx, valid_idx = base.forward_chaining_split(bundle, output_year=args.output_year, validation_frac=args.validation_frac)
    base.scale_numeric_features(bundle, train_idx)
    research_valid_idx, audit_valid_idx = split_validation_indices(bundle, valid_idx)

    research_logistic, research_tree, _ = base.evaluate_baselines(bundle, train_idx, research_valid_idx)
    audit_logistic, audit_tree, _ = base.evaluate_baselines(bundle, train_idx, audit_valid_idx)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with BUNDLE_PATH.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez(
        SPLITS_PATH,
        train_idx=train_idx,
        research_valid_idx=research_valid_idx,
        audit_valid_idx=audit_valid_idx,
    )

    context = {
        "config": {
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "output_year": OUTPUT_YEAR,
            "season_type": SEASON_TYPE,
            "seed": SEED,
            "validation_frac": VALIDATION_FRAC,
            "research_valid_frac": RESEARCH_VALID_FRAC,
            "max_games_per_season": MAX_GAMES_PER_SEASON,
        },
        "sizes": {
            "n_transitions": int(len(bundle.numeric)),
            "n_games": int(np.unique(bundle.game_id).size),
            "n_players": int(bundle.player_ids.max()),
            "train_rows": int(train_idx.size),
            "research_valid_rows": int(research_valid_idx.size),
            "audit_valid_rows": int(audit_valid_idx.size),
        },
        "targets": TARGETS,
        "score_weights": SCORE_WEIGHTS,
        "research_baselines": {
            "logistic": research_logistic,
            "tree": research_tree,
        },
        "audit_baselines": {
            "logistic": audit_logistic,
            "tree": audit_tree,
        },
    }
    CONTEXT_PATH.write_text(json.dumps(context, indent=2))

    return PreparedExperiment(
        bundle=bundle,
        train_idx=train_idx,
        research_valid_idx=research_valid_idx,
        audit_valid_idx=audit_valid_idx,
        research_baselines=context["research_baselines"],
        audit_baselines=context["audit_baselines"],
        context=context,
    )


def load_prepared() -> PreparedExperiment:
    if not BUNDLE_PATH.exists() or not SPLITS_PATH.exists() or not CONTEXT_PATH.exists():
        raise FileNotFoundError("Prepared cache missing. Run `python prepare.py` first.")
    with BUNDLE_PATH.open("rb") as handle:
        bundle = pickle.load(handle)
    split_arrays = np.load(SPLITS_PATH)
    context = json.loads(CONTEXT_PATH.read_text())
    return PreparedExperiment(
        bundle=bundle,
        train_idx=split_arrays["train_idx"].astype(np.int64),
        research_valid_idx=split_arrays["research_valid_idx"].astype(np.int64),
        audit_valid_idx=split_arrays["audit_valid_idx"].astype(np.int64),
        research_baselines=context["research_baselines"],
        audit_baselines=context["audit_baselines"],
        context=context,
    )


def ensure_prepared(rebuild_dataset: bool = False) -> PreparedExperiment:
    if rebuild_dataset or not (BUNDLE_PATH.exists() and SPLITS_PATH.exists() and CONTEXT_PATH.exists()):
        return prepare_experiment(rebuild_dataset=rebuild_dataset)
    return load_prepared()


def compute_score(metrics: Dict[str, float], baselines: Dict[str, Dict[str, float]]) -> float:
    logistic = baselines["logistic"]
    tree = baselines["tree"]
    score = 0.0
    score += SCORE_WEIGHTS["rmse"] * (metrics["rmse_final_margin"] / tree["rmse_final_margin"])
    score += SCORE_WEIGHTS["brier"] * (metrics["brier"] / logistic["brier"])
    score += SCORE_WEIGHTS["logloss"] * (metrics["logloss"] / logistic["logloss"])
    if metrics.get("monotonic_positive_clutch_home_win_prob", 1.0) <= metrics.get("monotonic_negative_clutch_home_win_prob", 0.0):
        score += 0.05
    score += 0.02 * max(0.0, -metrics.get("entropy_variance_corr", 0.0))
    return float(score)


def evaluate_model(model: torch.nn.Module, prepared: PreparedExperiment, split: str, device: torch.device) -> Dict[str, float]:
    if split == "research":
        indices = prepared.research_valid_idx
        baselines = prepared.research_baselines
    elif split == "audit":
        indices = prepared.audit_valid_idx
        baselines = prepared.audit_baselines
    else:
        raise ValueError(f"Unknown split: {split}")
    metrics = base.evaluate_value_model(model, prepared.bundle, indices, device)
    metrics["score"] = compute_score(metrics, baselines)
    return metrics


def save_summary(payload: Dict[str, object]) -> None:
    LATEST_SUMMARY.write_text(json.dumps(payload, indent=2))


def main() -> None:
    prepared = prepare_experiment(rebuild_dataset=False)
    print("---")
    print(f"n_transitions:       {prepared.context['sizes']['n_transitions']}")
    print(f"n_games:             {prepared.context['sizes']['n_games']}")
    print(f"train_rows:          {prepared.context['sizes']['train_rows']}")
    print(f"research_valid_rows: {prepared.context['sizes']['research_valid_rows']}")
    print(f"audit_valid_rows:    {prepared.context['sizes']['audit_valid_rows']}")
    print(f"research_tree_rmse:  {prepared.research_baselines['tree']['rmse_final_margin']:.6f}")
    print(f"research_log_brier:  {prepared.research_baselines['logistic']['brier']:.6f}")
    print(f"research_logloss:    {prepared.research_baselines['logistic']['logloss']:.6f}")
    print(f"target_score:        {TARGETS['research_score']:.6f}")


if __name__ == "__main__":
    main()
