#!/usr/bin/env python3
"""Develop a time-decayed, date-correct EPM-skill prior for STOP RAPM.

The structured defensive prior is built from expected steals, expected blocks
times a stabilized block-recovery probability, and a time-decayed forecast of
total offensive fouls drawn (OFFD). Charges are an auxiliary persistence
feature only because OFFD also contains moving screens and other drawn
offensive fouls that are not separately classified in play-by-play.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

import rapm as rapm_solver
import tune_stop_feature_prior as feature_prior
import tune_stop_personal_prior as personal


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PIPELINE_ROOT / "results"
EPM_DIR = PIPELINE_ROOT / "cache" / "epm"
DEFAULT_BOXSCORE = feature_prior.DEFAULT_BOXSCORE
HALF_LIFE_GRID = [365.0, 550.0, 700.0, 1000.0]
MAPPER_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
RECOVERY_K_GRID = [5.0, 10.0, 20.0, 50.0, 100.0]
OFFD_EB_K = [500.0, 1500.0, 3000.0, 6000.0]
ARM_COLUMNS = {
    "epm_steals": ["p_stl_100"],
    "epm_steals_recovered_blocks": ["p_stl_100", "expected_recovered_blocks_100"],
    "epm_steals_recovered_blocks_offd": [
        "p_stl_100",
        "expected_recovered_blocks_100",
        "expected_offd_100",
    ],
}
OFFD_NUMERIC = [
    "td_offd_rate",
    "offd_eb_500",
    "offd_eb_1500",
    "offd_eb_3000",
    "offd_eb_6000",
    "td_charge_rate",
    "above_league_year_share",
    "positive_excess_rate",
    "log_td_exposure",
    "observed_seasons",
]
OFFD_CATEGORICAL = ["Pos2_filled", "Defensive_Role_filled"]


def load_box_history(path: Path) -> pd.DataFrame:
    columns = [
        "nba_id",
        "Name",
        "year",
        "playoffs",
        "DefPoss",
        "Pos2",
        "Defensive Role",
        "OFFD_100",
        "d_Blocks_Per100",
        "block_recov_percent",
        "v2_charges_drawn_total",
    ]
    box = pd.read_csv(path, usecols=columns, low_memory=False)
    if "playoffs" in box:
        box = box[pd.to_numeric(box["playoffs"], errors="coerce").fillna(0).eq(0)].copy()
    numeric = [
        "nba_id",
        "year",
        "DefPoss",
        "OFFD_100",
        "d_Blocks_Per100",
        "block_recov_percent",
        "v2_charges_drawn_total",
    ]
    for column in numeric:
        box[column] = pd.to_numeric(box[column], errors="coerce")
    box = box.dropna(subset=["nba_id", "year", "DefPoss", "OFFD_100"])
    box = box[box["DefPoss"] > 0].copy()
    box["nba_id"] = box["nba_id"].astype(int)
    box["year"] = box["year"].astype(int)
    if box.duplicated(["nba_id", "year"]).any():
        raise ValueError("boxscore source has duplicate regular-season player-years")
    box["offd_count"] = box["OFFD_100"] * box["DefPoss"] / 100.0
    box["block_count"] = box["d_Blocks_Per100"] * box["DefPoss"] / 100.0
    box["recovered_block_count"] = box["block_count"] * box["block_recov_percent"]
    box["charge_count"] = box["v2_charges_drawn_total"]
    box["league_offd_rate"] = box["year"].map(
        box.groupby("year").apply(
            lambda group: 100.0 * group["offd_count"].sum() / group["DefPoss"].sum(),
            include_groups=False,
        )
    )
    return box.sort_values(["nba_id", "year"]).reset_index(drop=True)


def season_decay_weight(source_end: int, year: pd.Series, half_life_days: float) -> pd.Series:
    age_days = (int(source_end) - pd.to_numeric(year, errors="raise")) * 365.25
    return np.power(0.5, age_days / float(half_life_days))


def build_td_mechanism_panel(box: pd.DataFrame, source_end: int, half_life_days: float) -> pd.DataFrame:
    selected = box[box["year"] <= int(source_end)].copy()
    selected["decay"] = season_decay_weight(source_end, selected["year"], half_life_days)
    selected["weighted_exposure"] = selected["DefPoss"] * selected["decay"]
    selected["weighted_offd"] = selected["offd_count"] * selected["decay"]
    selected["weighted_blocks"] = selected["block_count"] * selected["decay"]
    selected["weighted_recovered_blocks"] = selected["recovered_block_count"] * selected["decay"]
    selected["weighted_charges"] = selected["charge_count"] * selected["decay"]
    selected["above_league"] = selected["OFFD_100"] > selected["league_offd_rate"]
    selected["positive_excess_count"] = (
        np.maximum(selected["OFFD_100"] - selected["league_offd_rate"], 0.0)
        * selected["DefPoss"]
        / 100.0
        * selected["decay"]
    )

    league_offd = 100.0 * selected["weighted_offd"].sum() / selected["weighted_exposure"].sum()
    league_recovery = (
        selected["weighted_recovered_blocks"].sum() / selected["weighted_blocks"].sum()
    )
    rows = []
    for player_id, group in selected.groupby("nba_id", sort=False):
        exposure = float(group["weighted_exposure"].sum())
        offd = float(group["weighted_offd"].sum())
        blocks = float(group["weighted_blocks"].sum())
        recovered = float(group["weighted_recovered_blocks"].sum())
        charge_mask = group["charge_count"].notna()
        charge_exposure = float(group.loc[charge_mask, "weighted_exposure"].sum())
        charges = float(group.loc[charge_mask, "weighted_charges"].sum())
        latest = group.sort_values("year")
        pos = latest["Pos2"].dropna().astype(str)
        role = latest["Defensive Role"].dropna().astype(str)
        row = {
            "nba_id": int(player_id),
            "player_name": str(latest.iloc[-1]["Name"]),
            "source_end": int(source_end),
            "half_life_days": float(half_life_days),
            "td_exposure": exposure,
            "td_offd_count": offd,
            "td_offd_rate": 100.0 * offd / exposure if exposure > 0 else np.nan,
            "td_block_count": blocks,
            "td_recovered_block_count": recovered,
            "td_recovery_rate": recovered / blocks if blocks > 0 else np.nan,
            "td_charge_rate": 100.0 * charges / charge_exposure if charge_exposure > 0 else np.nan,
            "above_league_year_share": float(
                np.average(group["above_league"].astype(float), weights=group["weighted_exposure"])
            ),
            "positive_excess_rate": (
                100.0 * group["positive_excess_count"].sum() / exposure if exposure > 0 else np.nan
            ),
            "log_td_exposure": np.log1p(exposure),
            "observed_seasons": int(group["year"].nunique()),
            "league_offd_rate": league_offd,
            "league_recovery_rate": league_recovery,
            "Pos2_filled": pos.iloc[-1] if not pos.empty else "Unknown",
            "Defensive_Role_filled": role.iloc[-1] if not role.empty else "Unknown",
        }
        for k in OFFD_EB_K:
            row[f"offd_eb_{int(k)}"] = (
                100.0 * offd + k * league_offd
            ) / (exposure + k)
        rows.append(row)
    return pd.DataFrame(rows)


def annual_targets(box: pd.DataFrame) -> pd.DataFrame:
    target = box.copy()
    target["target_offd_100"] = target["OFFD_100"]
    target["target_recovery_rate"] = target["block_recov_percent"]
    target["target_block_count"] = target["block_count"]
    return target[
        [
            "nba_id",
            "year",
            "DefPoss",
            "target_offd_100",
            "target_recovery_rate",
            "target_block_count",
        ]
    ].rename(columns={"year": "target_year", "DefPoss": "target_def_poss"})


def build_examples(box, half_life_days, first_source_end=2016, last_target_year=2026):
    target = annual_targets(box)
    frames = []
    for source_end in range(int(first_source_end), int(last_target_year)):
        panel = build_td_mechanism_panel(box, source_end, half_life_days)
        panel["target_year"] = source_end + 1
        frames.append(panel.merge(target, on=["nba_id", "target_year"], how="inner", validate="one_to_one"))
    return pd.concat(frames, ignore_index=True)


def select_offd_mapper(examples: pd.DataFrame, max_target_year: int):
    train = examples[examples["target_year"] <= int(max_target_year)].copy()
    groups = train["target_year"].astype(int).to_numpy()
    splits = min(5, len(np.unique(groups)))
    receipt = []
    for alpha in MAPPER_ALPHA_GRID:
        for fold, (fit_idx, val_idx) in enumerate(
            GroupKFold(n_splits=splits).split(train, groups=groups), 1
        ):
            fit = train.iloc[fit_idx]
            val = train.iloc[val_idx]
            model = feature_prior.make_mapper(OFFD_NUMERIC, OFFD_CATEGORICAL, alpha)
            fit_weights = np.minimum(fit["target_def_poss"].to_numpy(float), 3000.0)
            model.fit(
                fit[OFFD_NUMERIC + OFFD_CATEGORICAL],
                fit["target_offd_100"],
                ridge__sample_weight=fit_weights,
            )
            prediction = np.clip(model.predict(val[OFFD_NUMERIC + OFFD_CATEGORICAL]), 0.0, None)
            mse = mean_squared_error(
                val["target_offd_100"],
                prediction,
                sample_weight=np.minimum(val["target_def_poss"].to_numpy(float), 3000.0),
            )
            receipt.append({"alpha": alpha, "fold": fold, "mse": mse})
    receipt = pd.DataFrame(receipt)
    best_alpha = float(receipt.groupby("alpha")["mse"].mean().idxmin())
    return best_alpha, receipt


def fit_offd_mapper(examples, prediction_panel, max_target_year, alpha):
    train = examples[examples["target_year"] <= int(max_target_year)].copy()
    model = feature_prior.make_mapper(OFFD_NUMERIC, OFFD_CATEGORICAL, alpha)
    weights = np.minimum(train["target_def_poss"].to_numpy(float), 3000.0)
    model.fit(
        train[OFFD_NUMERIC + OFFD_CATEGORICAL],
        train["target_offd_100"],
        ridge__sample_weight=weights,
    )
    return np.clip(model.predict(prediction_panel[OFFD_NUMERIC + OFFD_CATEGORICAL]), 0.0, None)


def recovery_prediction(panel: pd.DataFrame, pseudo_blocks: float) -> np.ndarray:
    return (
        panel["td_recovered_block_count"].fillna(0).to_numpy(float)
        + float(pseudo_blocks) * panel["league_recovery_rate"].to_numpy(float)
    ) / (panel["td_block_count"].fillna(0).to_numpy(float) + float(pseudo_blocks))


def select_history_hyperparameters(box: pd.DataFrame, max_target_year=2023):
    offd_rows = []
    recovery_rows = []
    example_cache = {}
    for half_life in HALF_LIFE_GRID:
        examples = build_examples(box, half_life)
        example_cache[half_life] = examples
        alpha, cv = select_offd_mapper(examples, max_target_year)
        offd_rows.append(
            {
                "half_life_days": half_life,
                "mapper_alpha": alpha,
                "mse": float(cv[cv["alpha"] == alpha]["mse"].mean()),
            }
        )
        history = examples[examples["target_year"] <= int(max_target_year)].dropna(
            subset=["target_recovery_rate", "target_block_count"]
        )
        for k in RECOVERY_K_GRID:
            prediction = recovery_prediction(history, k)
            mse = mean_squared_error(
                history["target_recovery_rate"],
                prediction,
                sample_weight=np.minimum(history["target_block_count"].to_numpy(float), 100.0),
            )
            recovery_rows.append({"half_life_days": half_life, "pseudo_blocks": k, "mse": mse})
    offd_receipt = pd.DataFrame(offd_rows)
    recovery_receipt = pd.DataFrame(recovery_rows)
    offd_best = offd_receipt.loc[offd_receipt["mse"].idxmin()]
    recovery_best = recovery_receipt.loc[recovery_receipt["mse"].idxmin()]
    return (
        float(offd_best["half_life_days"]),
        float(offd_best["mapper_alpha"]),
        float(recovery_best["half_life_days"]),
        float(recovery_best["pseudo_blocks"]),
        offd_receipt,
        recovery_receipt,
        example_cache,
    )


def load_epm_snapshot(source_end: int) -> pd.DataFrame:
    path = EPM_DIR / f"epm_all_{int(source_end)}-06-30_go0.csv"
    columns = ["requested_date", "date", "season", "player_id", "player_name", "p_stl_100", "p_blk_100"]
    epm = pd.read_csv(path, usecols=columns)
    epm["player_id"] = pd.to_numeric(epm["player_id"], errors="raise").astype(int)
    epm["p_stl_100"] = pd.to_numeric(epm["p_stl_100"], errors="coerce")
    epm["p_blk_100"] = pd.to_numeric(epm["p_blk_100"], errors="coerce")
    if set(epm["requested_date"].astype(str)) != {f"{source_end}-06-30"}:
        raise ValueError(f"EPM snapshot {path} does not match requested as-of date")
    if epm["player_id"].duplicated().any():
        raise ValueError(f"EPM snapshot {path} contains duplicate players")
    return epm.rename(columns={"player_id": "nba_id", "player_name": "epm_player_name"})


def build_structured_prior(
    box,
    source_end,
    offd_half_life,
    offd_alpha,
    recovery_half_life,
    recovery_k,
    max_mapper_target_year,
    arm,
    scale=1.0,
):
    offd_examples = build_examples(box, offd_half_life)
    offd_panel = build_td_mechanism_panel(box, source_end, offd_half_life)
    offd_panel["expected_offd_100"] = fit_offd_mapper(
        offd_examples, offd_panel, max_mapper_target_year, offd_alpha
    )
    recovery_panel = build_td_mechanism_panel(box, source_end, recovery_half_life)
    recovery_panel["expected_recovery_rate"] = recovery_prediction(recovery_panel, recovery_k)
    merged = offd_panel.merge(
        recovery_panel[["nba_id", "expected_recovery_rate"]],
        on="nba_id",
        how="outer",
        validate="one_to_one",
    ).merge(load_epm_snapshot(source_end), on="nba_id", how="outer", validate="one_to_one")
    merged["player_name"] = merged["player_name"].fillna(merged["epm_player_name"])
    merged["expected_recovered_blocks_100"] = (
        merged["p_blk_100"] * merged["expected_recovery_rate"]
    )
    for column in ARM_COLUMNS[arm]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        mean = np.average(
            merged.loc[merged[column].notna(), column],
            weights=merged.loc[merged[column].notna(), "td_exposure"].fillna(0).clip(lower=1),
        )
        merged[column] = merged[column].fillna(mean)
    merged["expected_stop_100"] = merged[ARM_COLUMNS[arm]].sum(axis=1)
    valid = merged["td_exposure"].fillna(0) > 0
    league = np.average(
        merged.loc[valid, "expected_stop_100"], weights=merged.loc[valid, "td_exposure"]
    )
    merged["relative_expected_stop_100"] = merged["expected_stop_100"] - league
    merged["prior_value"] = -float(scale) * merged["relative_expected_stop_100"]
    merged["arm"] = arm
    merged["prior_scale"] = float(scale)
    return merged


def prepare_td_fold(start_year, end_year, validation_year, half_life_days):
    train = personal.load_stop_window(start_year, end_year)
    validation = personal.load_stop_window(validation_year, validation_year)
    players = personal._all_players(train, validation)
    player_to_col = {}
    for player in players:
        player_to_col[f"{player}_off"] = len(player_to_col)
        player_to_col[f"{player}_def"] = len(player_to_col)
    train_ha = rapm_solver.transform_to_home_away_format(train)
    validation_ha = rapm_solver.transform_to_home_away_format(validation)
    X_train, y_train, _ = rapm_solver.build_simple_design_matrix(train_ha, player_to_col)
    X_validation, y_validation, _ = rapm_solver.build_simple_design_matrix(validation_ha, player_to_col)
    weights = rapm_solver.compute_time_decay_weights(
        train_ha, decay_base=rapm_solver.half_life_to_decay_base(half_life_days)
    )
    return {
        "train_start": start_year,
        "train_end": end_year,
        "validation_year": validation_year,
        "X_train": X_train,
        "y_train": y_train - np.average(y_train, weights=weights),
        "train_weights": weights,
        "X_validation": X_validation,
        "y_validation": y_validation - np.mean(y_validation),
        "player_to_col": player_to_col,
        "players": players,
        "train_frame": train,
        "train_rows": len(train),
        "validation_rows": len(validation),
    }


def attach_prior(fold, prior_frame):
    centered, offset, coverage = personal.center_prior_on_training_defense(
        prior_frame[["nba_id", "prior_value"]], fold["train_frame"], fold["players"]
    )
    output = dict(fold)
    output["def_prior"] = centered
    output["prior_offset"] = offset
    output["coverage"] = coverage
    return output


def evaluate(folds, off_alpha, def_alpha, use_prior=True):
    rows = []
    for fold in folds:
        beta = rapm_solver.simplified_alternating_rapm(
            fold["X_train"],
            fold["y_train"],
            fold["player_to_col"],
            fold["players"],
            alpha_offense=float(off_alpha),
            alpha_defense=float(def_alpha),
            max_iter=100,
            tol=1e-4,
            off_prior=None,
            def_prior=fold["def_prior"] if use_prior else None,
            weights=fold["train_weights"],
        )
        prediction = fold["X_validation"] @ beta
        rows.append(
            {
                "validation_year": fold["validation_year"],
                "off_alpha": float(off_alpha),
                "def_alpha": float(def_alpha),
                "use_prior": bool(use_prior),
                "mse": float(np.mean((fold["y_validation"] - prediction) ** 2)),
                "prior_coverage": fold.get("coverage", 0.0),
            }
        )
    return rows


def tune_alphas(folds, grid, seed_off=3000.0, seed_def=5000.0):
    off_scores = {}
    receipt = []
    for alpha in grid:
        rows = evaluate(folds, alpha, seed_def, True)
        for row in rows:
            row.update({"axis": "off", "candidate_alpha": alpha})
        receipt.extend(rows)
        off_scores[alpha] = np.mean([row["mse"] for row in rows])
    best_off = min(off_scores, key=off_scores.get)
    def_scores = {}
    for alpha in grid:
        rows = evaluate(folds, best_off, alpha, True)
        for row in rows:
            row.update({"axis": "def", "candidate_alpha": alpha})
        receipt.extend(rows)
        def_scores[alpha] = np.mean([row["mse"] for row in rows])
    best_def = min(def_scores, key=def_scores.get)
    return best_off, best_def, pd.DataFrame(receipt)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxscore", type=Path, default=DEFAULT_BOXSCORE)
    parser.add_argument("--rapm-half-life", type=float, default=700.0)
    parser.add_argument("--alpha-grid", default=personal.DEFAULT_GRID)
    parser.add_argument("--scale-grid", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--receipt", type=Path, default=RESULTS_DIR / "stop_td_epm_prior_forward_cv.csv")
    parser.add_argument(
        "--component-receipt",
        type=Path,
        default=RESULTS_DIR / "stop_td_epm_component_grid.csv",
    )
    parser.add_argument("--prior-output", type=Path, default=RESULTS_DIR / "stop_td_epm_prior_2027.csv")
    parser.add_argument("--history-receipt", type=Path, default=RESULTS_DIR / "stop_td_mechanism_hyperparameters.csv")
    return parser.parse_args()


def apply_component_weights(prior, steal_coef, recovered_block_coef, offd_coef):
    output = prior.copy()
    output["expected_stop_100"] = (
        float(steal_coef) * output["p_stl_100"]
        + float(recovered_block_coef) * output["expected_recovered_blocks_100"]
        + float(offd_coef) * output["expected_offd_100"]
    )
    valid = output["td_exposure"].fillna(0) > 0
    league = np.average(
        output.loc[valid, "expected_stop_100"],
        weights=output.loc[valid, "td_exposure"],
    )
    output["relative_expected_stop_100"] = output["expected_stop_100"] - league
    output["prior_value"] = -output["relative_expected_stop_100"]
    output["steal_coef"] = float(steal_coef)
    output["recovered_block_coef"] = float(recovered_block_coef)
    output["offd_coef"] = float(offd_coef)
    return output


def tune_component_weights(base_folds, raw_priors):
    receipt = []

    def score(stage, steal_coef, recovered_block_coef, offd_coef):
        folds = [
            attach_prior(
                base,
                apply_component_weights(
                    prior, steal_coef, recovered_block_coef, offd_coef
                ),
            )
            for base, prior in zip(base_folds, raw_priors)
        ]
        rows = evaluate(folds, 2000.0, 5000.0, True)
        for row in rows:
            row.update(
                {
                    "stage": stage,
                    "steal_coef": steal_coef,
                    "recovered_block_coef": recovered_block_coef,
                    "offd_coef": offd_coef,
                }
            )
        receipt.extend(rows)
        return float(np.mean([row["mse"] for row in rows]))

    steal_grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    block_grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    offd_grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    steal_scores = {value: score("steal", value, 0.0, 0.0) for value in steal_grid}
    steal = min(steal_scores, key=steal_scores.get)
    block_scores = {
        value: score("recovered_block", steal, value, 0.0) for value in block_grid
    }
    block = min(block_scores, key=block_scores.get)
    offd_scores = {value: score("offd", steal, block, value) for value in offd_grid}
    offd = min(offd_scores, key=offd_scores.get)
    steal_scores = {
        value: score("steal_pass2", value, block, offd) for value in steal_grid
    }
    steal = min(steal_scores, key=steal_scores.get)
    block_scores = {
        value: score("recovered_block_pass2", steal, value, offd)
        for value in block_grid
    }
    block = min(block_scores, key=block_scores.get)
    offd_scores = {
        value: score("offd_pass2", steal, block, value) for value in offd_grid
    }
    offd = min(offd_scores, key=offd_scores.get)
    return steal, block, offd, pd.DataFrame(receipt)


def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    box = load_box_history(args.boxscore)
    (
        offd_half_life,
        offd_alpha,
        recovery_half_life,
        recovery_k,
        offd_receipt,
        recovery_receipt,
        _,
    ) = select_history_hyperparameters(box, max_target_year=2023)
    history_receipt = pd.concat(
        [
            offd_receipt.assign(stage="offd_mapper"),
            recovery_receipt.assign(stage="block_recovery"),
        ],
        ignore_index=True,
    )
    history_receipt.to_csv(args.history_receipt, index=False)
    print(
        f"history selection: OFFD half-life={offd_half_life:g}, alpha={offd_alpha:g}; "
        f"recovery half-life={recovery_half_life:g}, pseudo-blocks={recovery_k:g}"
    )

    specs = [(2021, 2023, 2024), (2022, 2024, 2025), (2023, 2025, 2026)]
    base_folds = [prepare_td_fold(a, b, c, args.rapm_half_life) for a, b, c in specs]
    receipt = []
    arm_priors = {}
    for arm in ARM_COLUMNS:
        priors = []
        folds = []
        for (_, source_end, _), base in zip(specs, base_folds):
            prior = build_structured_prior(
                box,
                source_end,
                offd_half_life,
                offd_alpha,
                recovery_half_life,
                recovery_k,
                source_end,
                arm,
                scale=1.0,
            )
            priors.append(prior)
            folds.append(attach_prior(base, prior))
        arm_priors[arm] = priors
        rows = evaluate(folds, 3000.0, 5000.0, True)
        for row in rows:
            row.update({"stage": "arm", "arm": arm, "prior_scale": 1.0})
        receipt.extend(rows)
        print(f"arm {arm}: mean future MSE={np.mean([row['mse'] for row in rows]):.10f}")
    raw_full_priors = arm_priors["epm_steals_recovered_blocks_offd"]
    steal_coef, block_coef, offd_coef, component_receipt = tune_component_weights(
        base_folds, raw_full_priors
    )
    component_receipt.to_csv(args.component_receipt, index=False)
    selected_folds = [
        attach_prior(
            base,
            apply_component_weights(prior, steal_coef, block_coef, offd_coef),
        )
        for base, prior in zip(base_folds, raw_full_priors)
    ]
    best_off, best_def, alpha_receipt = tune_alphas(
        selected_folds,
        personal.parse_grid(args.alpha_grid),
        seed_off=2000.0,
        seed_def=5000.0,
    )
    alpha_receipt["stage"] = "alpha"
    alpha_receipt["arm"] = "component_weighted"
    alpha_receipt["steal_coef"] = steal_coef
    alpha_receipt["recovered_block_coef"] = block_coef
    alpha_receipt["offd_coef"] = offd_coef
    receipt.extend(alpha_receipt.to_dict("records"))

    zero_folds = []
    for base in base_folds:
        zero = dict(base)
        zero["def_prior"] = {player: 0.0 for player in base["players"]}
        zero_folds.append(zero)
    zero_off, zero_def, zero_receipt = tune_alphas(
        zero_folds, personal.parse_grid(args.alpha_grid)
    )
    zero_receipt["stage"] = "zero_alpha"
    zero_receipt["arm"] = "zero_prior"
    zero_receipt["prior_scale"] = 0.0
    receipt.extend(zero_receipt.to_dict("records"))
    selected_rows = evaluate(selected_folds, best_off, best_def, True)
    zero_rows = evaluate(zero_folds, zero_off, zero_def, True)

    out = pd.DataFrame(receipt)
    out["selected_arm"] = "component_weighted"
    out["selected_steal_coef"] = steal_coef
    out["selected_recovered_block_coef"] = block_coef
    out["selected_offd_coef"] = offd_coef
    out["selected_off_alpha"] = best_off
    out["selected_def_alpha"] = best_def
    out["zero_off_alpha"] = zero_off
    out["zero_def_alpha"] = zero_def
    out.to_csv(args.receipt, index=False)

    final_prior = build_structured_prior(
        box,
        2026,
        offd_half_life,
        offd_alpha,
        recovery_half_life,
        recovery_k,
        2026,
        "epm_steals_recovered_blocks_offd",
        scale=1.0,
    )
    final_prior = apply_component_weights(
        final_prior, steal_coef, block_coef, offd_coef
    )
    final_prior = final_prior[final_prior["requested_date"].notna()].copy()
    active = final_prior["td_exposure"].fillna(0) > 0
    active_league = np.average(
        final_prior.loc[active, "expected_stop_100"],
        weights=final_prior.loc[active, "td_exposure"],
    )
    final_prior["relative_expected_stop_100"] = (
        final_prior["expected_stop_100"] - active_league
    )
    final_prior["prior_value"] = -final_prior["relative_expected_stop_100"]
    final_prior["active_prior_league_expected_stop_100"] = active_league
    final_prior["selected_off_alpha"] = best_off
    final_prior["selected_def_alpha"] = best_def
    final_prior["rapm_half_life_days"] = args.rapm_half_life
    final_prior["offd_half_life_days"] = offd_half_life
    final_prior["recovery_half_life_days"] = recovery_half_life
    final_prior["recovery_pseudo_blocks"] = recovery_k
    final_prior.to_csv(args.prior_output, index=False)

    print(
        f"selected weights: steals={steal_coef:g}, recovered_blocks={block_coef:g}, "
        f"OFFD={offd_coef:g}; alphas={best_off:g}/{best_def:g}"
    )
    print(f"selected mean future MSE={np.mean([row['mse'] for row in selected_rows]):.10f}")
    print(f"optimized TD zero-prior MSE={np.mean([row['mse'] for row in zero_rows]):.10f}")
    for selected, zero in zip(selected_rows, zero_rows):
        print(
            f"{selected['validation_year']}: selected={selected['mse']:.10f}, "
            f"zero={zero['mse']:.10f}, gain={zero['mse'] - selected['mse']:+.10f}"
        )
    print(f"wrote {args.receipt}")
    print(f"wrote {args.component_receipt}")
    print(f"wrote {args.prior_output}")


if __name__ == "__main__":
    main()
