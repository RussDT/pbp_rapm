"""LightGBM fusion layer over the WNBA Kalman skill + box states (research).

DARKO-style second stage: for each estimated skill, a gradient-boosted model
predicts the next game's observed rate from strictly-prior information only:

- the latest Kalman posterior mean of ALL 14 box-stat rates
  (``kalman_box/predictions.parquet``) and ALL 8 skill rates plus their
  posterior variances (``estimated_skills/predictions.parquet``), each taken
  from the player's most recent observation STRICTLY before the game date
  (merge_asof, no exact matches), so cross-stat structure (FT% -> 3P%,
  block/rim interplay, volume trends) can inform each skill;
- schedule/history context: days since the skill was last observed, career
  observation count, career exposure to date, within-season prior game count;
- the as-of league mean of the target skill (strictly-before-date pooling,
  as in the panel's centering table);
- the game-log listed position (categorical), known pregame.

The model is a RESIDUAL corrector: it fits ``obs - kalman_pred`` with heavy
regularization, and ``fusion_pred = kalman_pred + correction``. A first
direct-rate variant (600 trees, lr 0.03, num_leaves 31) lost to plain Kalman
on every skill, with losses shrinking monotonically as training seasons
accumulated and gain spread nearly uniformly across ~40 features — diffuse
noise-fitting, not signal. That dev-diagnosed failure motivated this residual
formulation; both are recorded in WNBA_ESTIMATED_SKILLS.md, and because the
first variant's 2026 numbers were viewed, the fusion layer's 2026 column is
labeled report-only evidence, weaker than the Kalman panel's untouched
holdout. Current-game exposure (attempts/minutes) is never a feature. Training is
season-expanding walk-forward: each dev season s in 2023-2025 is predicted by
a model trained on seasons < s; 2026 is predicted by one frozen model trained
on <= 2025. Rows and exposure weights are identical to the Kalman panel's
eligible sets, so the fusion-vs-Kalman comparison is paired. Per-skill gate:
fusion replaces the plain Kalman reading only where it wins on dev; 2026 is
confirmation evidence, never selection.

Outputs under ``Results/gameday/estimated_skills/``: fusion_report.json,
fusion_predictions.parquet. Generated artifacts stay out of git.

Run: ``.venv/bin/python wnba_test/build_wnba_skill_fusion.py [--skills rim_pct,...]``
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

import build_wnba_estimated_skills as skills_mod
import wnba_gameday_common as common

OUT_DIR = common.GAMEDAY_DIR / "estimated_skills"
BOX_PRED_PATH = common.GAMEDAY_DIR / "kalman_box" / "predictions.parquet"
SKILL_PRED_PATH = OUT_DIR / "predictions.parquet"

DEV_SEASONS = (2023, 2024, 2025)
HOLDOUT_SEASON = 2026
SEED = 20260826

LGB_PARAMS = dict(
    objective="regression",
    n_estimators=200,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=150,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    importance_type="gain",
    random_state=SEED,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


def state_features(pred: pd.DataFrame, key: str, cols: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Per stat/skill: (person_id, game_date, renamed value cols) sorted for asof."""
    out = {}
    for name, grp in pred.groupby(key):
        frame = grp[["person_id", "game_date", *cols]].rename(
            columns={c: f"{new}_{name}" for c, new in cols.items()}
        )
        out[name] = frame.sort_values("game_date", kind="mergesort")
    return out


def asof_join(base: pd.DataFrame, feature: pd.DataFrame) -> pd.DataFrame:
    """Latest feature row strictly before each base game_date, per person."""
    return pd.merge_asof(
        base.sort_values("game_date", kind="mergesort"),
        feature,
        on="game_date",
        by="person_id",
        allow_exact_matches=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skills", type=str, default=None)
    args = parser.parse_args()
    names = skills_mod.SKILL_NAMES if args.skills is None else [
        s.strip() for s in args.skills.split(",") if s.strip()
    ]

    panel = skills_mod.load_panel()
    box_pred = pd.read_parquet(BOX_PRED_PATH)
    box_pred["game_date"] = pd.to_datetime(box_pred["game_date"])
    skill_pred = pd.read_parquet(SKILL_PRED_PATH)
    skill_pred["game_date"] = pd.to_datetime(skill_pred["game_date"])

    box_feats = state_features(box_pred, "stat", {"post_rate": "box"})
    skill_feats = state_features(
        skill_pred, "skill", {"post_rate": "sk", "post_var": "skv"}
    )
    feat_cols = [f"box_{s}" for s in box_feats] + [
        c for s in skill_feats for c in (f"sk_{s}", f"skv_{s}")
    ]

    report: dict[str, dict] = {}
    pred_frames: list[pd.DataFrame] = []

    for spec in [s for s in skills_mod.SKILLS if s.name in names]:
        sub = skills_mod.skill_subpanel(panel, spec)
        centers, _ = skills_mod.league_center(sub, spec)
        sub["league_asof"] = centers
        sub["career_exposure"] = (
            sub.groupby("player_idx")["exposure"].cumsum() - sub["exposure"]
        )
        sub["season_step"] = sub.groupby(["player_idx", "season"]).cumcount()
        sub["position_cat"] = sub["position"].fillna("").astype("category")

        base = sub[
            [
                "person_id", "game_id", "game_date", "season", "exposure",
                "obs_rate", "step_k", "days_gap", "career_exposure",
                "season_step", "league_asof", "position_cat",
            ]
        ].copy()
        for frame in box_feats.values():
            base = asof_join(base, frame)
        for frame in skill_feats.values():
            base = asof_join(base, frame)

        # Paired comparison surface: the panel's own one-step-ahead prediction.
        kalman = skill_pred[skill_pred["skill"] == spec.name][
            ["person_id", "game_id", "pred_rate"]
        ].rename(columns={"pred_rate": "kalman_pred"})
        base = base.merge(kalman, on=["person_id", "game_id"], how="left")
        assert base["kalman_pred"].notna().all(), f"{spec.name}: unmatched kalman rows"

        eligible = base["step_k"] > 0
        features = [
            "days_gap", "step_k", "career_exposure", "season_step",
            "league_asof", "position_cat", *feat_cols,
        ]

        fold_rows: list[pd.DataFrame] = []
        importances: dict[str, float] = {}
        for target_season in (*DEV_SEASONS, HOLDOUT_SEASON):
            train = base[(base["season"] < target_season) & eligible]
            test = base[(base["season"] == target_season) & eligible]
            if train.empty or test.empty:
                continue
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(
                train[features], train["obs_rate"] - train["kalman_pred"],
                sample_weight=train["exposure"],
                categorical_feature=["position_cat"],
            )
            out = test[
                ["person_id", "game_id", "game_date", "season", "exposure",
                 "obs_rate", "kalman_pred"]
            ].copy()
            out["fusion_pred"] = test["kalman_pred"].to_numpy() + model.predict(
                test[features]
            )
            out["skill"] = spec.name
            fold_rows.append(out)
            if target_season == HOLDOUT_SEASON:
                gains = dict(zip(features, model.feature_importances_))
                total = sum(gains.values()) or 1.0
                importances = {
                    k: round(v / total, 4)
                    for k, v in sorted(gains.items(), key=lambda kv: -kv[1])[:12]
                }

        folds = pd.concat(fold_rows, ignore_index=True)
        pred_frames.append(folds)

        def wmse(frame: pd.DataFrame, col: str) -> float:
            diff = frame["obs_rate"] - frame[col]
            return float(
                np.sum(frame["exposure"] * diff * diff) / np.sum(frame["exposure"])
            )

        skill_report: dict[str, object] = {}
        for window, mask in (
            ("dev", folds["season"].isin(DEV_SEASONS)),
            ("holdout_2026", folds["season"] == HOLDOUT_SEASON),
        ):
            frame = folds[mask]
            fusion = wmse(frame, "fusion_pred")
            kal = wmse(frame, "kalman_pred")
            skill_report[window] = {
                "n_obs": int(len(frame)),
                "fusion_wmse": fusion,
                "kalman_wmse": kal,
                "pct_impr_vs_kalman": 100.0 * (kal - fusion) / kal,
            }
        per_season = {}
        for season, frame in folds.groupby("season"):
            per_season[int(season)] = {
                "fusion_wmse": wmse(frame, "fusion_pred"),
                "kalman_wmse": wmse(frame, "kalman_pred"),
            }
        skill_report["per_season"] = per_season
        skill_report["dev_gate_pass"] = bool(
            skill_report["dev"]["pct_impr_vs_kalman"] > 0
        )
        skill_report["top_feature_gain_share_holdout_model"] = importances
        report[spec.name] = skill_report
        d, h = skill_report["dev"], skill_report["holdout_2026"]
        print(
            f"{spec.name:>10}: dev fusion vs kalman {d['pct_impr_vs_kalman']:+.2f}% "
            f"({d['n_obs']} obs) | 2026 {h['pct_impr_vs_kalman']:+.2f}% ({h['n_obs']} obs)"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_parquet(OUT_DIR / "fusion_predictions.parquet", index=False)
    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "lgb_params": {k: v for k, v in LGB_PARAMS.items() if k != "verbose"},
        "n_features": len(
            ["days_gap", "step_k", "career_exposure", "season_step",
             "league_asof", "position_cat", *feat_cols]
        ),
        "protocol": (
            "season-expanding walk-forward; dev seasons predicted by models "
            "trained strictly earlier; 2026 predicted by one frozen <=2025 "
            "model; features are strictly-prior Kalman states via "
            "non-exact-match asof joins; current-game exposure excluded; "
            "per-skill gate decided on dev only"
        ),
    }
    (OUT_DIR / "fusion_report.json").write_text(
        json.dumps({"summary": meta, "per_skill": report}, indent=2)
    )
    n_pass = sum(1 for r in report.values() if r["dev_gate_pass"])
    print(
        f"\nwrote {OUT_DIR}/(fusion_report.json, fusion_predictions.parquet); "
        f"dev gate passed for {n_pass}/{len(report)} skills"
    )


if __name__ == "__main__":
    main()
