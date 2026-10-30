#!/usr/bin/env python3
"""Build the pre-2016 bridge for the frozen Russell ShotQuality v2 series.

Native ``RUSSELL_SHOTQUALITY v2`` remains authoritative for 2016 onward.  This
script learns, separately for midrange attempts and threes, how much of v2 can
be reproduced from information that is available throughout the 1997-2026 raw
PBP archive.  It deliberately does not rebuild or overwrite native v2.

The bridge has two stages:

1. a historical-feature student distills native ``p_v2``;
2. a logistic stack combines that student with the already-frozen portable
   ``rich_transfer_v3`` probability and calibrates against recorded outcomes.

The evaluation contract is chronological and fixed:

* 2016-2020: fit the v2 distiller;
* 2021-2022: fit the two-input outcome stack;
* 2023-2026: held-out comparison against native v2 and portable v3.

A bucket is emitted from the v2 stack only if it improves fidelity to native
v2 and does not worsen held-out Brier loss against portable v3.  Otherwise the
portable v3 probability is the explicit fallback.  Once that decision is
recorded, the emission distiller uses 2016-2022 and its stack is fit on
2023-2026 before scoring 1997-2015.  No pre-2016 outcome fits the emitted
bridge.  Historical outcomes and safe-crossfit estimates are audit-only.

The consolidated output carries ``zone``, ``p_make``, and ``ev_pts`` so it can
be consumed by ``build_vpm_target_swap.py`` after the bridge is deliberately
admitted to a historical VPM experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


PIPELINE_ROOT = REPO_ROOT / "nba_pipeline"
VPM_DATA_ROOT = REPO_ROOT.parent / "vpm" / "vpm" / "data"
DEFAULT_V2_PARQUETS = [
    VPM_DATA_ROOT / "russell_sq_v2_ev_16_22.parquet",
    VPM_DATA_ROOT / "russell_sq_v2_ev_21_26.parquet",
]
DEFAULT_FEATURE_DIR = (
    PIPELINE_ROOT
    / "results"
    / "historical_shot_quality"
    / "rich_transfer_v3_scored"
)
DEFAULT_TRANSITION_DIR = PIPELINE_ROOT / "results" / "sq_transition_backfill"
DEFAULT_SAFE_CROSSFIT = (
    PIPELINE_ROOT
    / "results"
    / "historical_shot_quality"
    / "safe_crossfit_full"
    / "safe_crossfit_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT
    / "results"
    / "historical_shot_quality"
    / "russell_v2_historical_bridge"
)

KEYS = ["season_end_year", "game_id", "event_num"]
BUCKETS = ("midrange", "three")
EVALUATION_TRAIN_YEARS = (2016, 2020)
EVALUATION_STACK_YEARS = (2021, 2022)
EVALUATION_TEST_YEARS = (2023, 2026)
EMISSION_DISTILL_YEARS = (2016, 2022)
EMISSION_STACK_YEARS = (2023, 2026)

STUDENT_CATEGORICAL = [
    "shot_zone",
    "shot_family",
    "modifier_bundle",
    "event_action_type_str",
    "season_type",
    "distance_source",
    "shot_clock_reset_reason",
    "shot_clock_confidence",
]
STUDENT_NUMERIC = [
    "shot_distance_ft",
    "shot_clock_est",
    "shot_clock_fraction",
    "elapsed_since_reset",
    "effective_reset_len",
    "period",
    "seconds_remaining_quarter",
    "shooter_pregame_make_rate",
    "shooter_pregame_log_attempts",
    "transition_probability",
    "historical_sq_make_prob",
    "historical_sq_direct_make_prob",
    "historical_sq_distilled_make_prob",
    "historical_sq_component_gap",
]
FEATURE_COLUMNS = list(dict.fromkeys([
    "source_file",
    "season_end_year",
    "season_type",
    "game_id",
    "event_num",
    "shooter_id",
    "model_bucket",
    "shot_value",
    "is_make",
    *STUDENT_CATEGORICAL,
    *[column for column in STUDENT_NUMERIC if column != "transition_probability"],
]))
OUTPUT_COLUMNS = [
    "season_end_year",
    "season_type",
    "source_file",
    "game_id",
    "event_num",
    "shooter_id",
    "zone",
    "made",
    "shot_value",
    "p_portable_v3",
    "p_v2_historical_student",
    "p_make",
    "ev_pts",
    "bridge_source",
    "transition_probability",
    "shooter_pregame_make_rate",
    "shooter_pregame_log_attempts",
    "student_portable_gap",
]


def _safe_probability(values: Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-5, 1.0 - 1e-5)


def _logit(values: Iterable[float]) -> np.ndarray:
    probability = _safe_probability(values)
    return np.log(probability / (1.0 - probability))


def _sigmoid(values: Iterable[float]) -> np.ndarray:
    linear = np.clip(np.asarray(values, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-linear))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_feature_filename(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"rich_transfer_v3_NBA(\d{2})(_PS)?\.parquet", path.name)
    if not match:
        raise ValueError(f"Unrecognized historical SQ feature file: {path.name}")
    short_year = int(match.group(1))
    year = 2000 + short_year if short_year <= 26 else 1900 + short_year
    return year, "ps" if match.group(2) else "rs"


def discover_feature_paths(
    feature_dir: Path,
    *,
    start_year: int,
    end_year: int,
) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(feature_dir.glob("rich_transfer_v3_NBA??*.parquet")):
        try:
            year, _ = parse_feature_filename(path)
        except ValueError:
            continue
        if start_year <= year <= end_year:
            selected.append(path)
    expected = 2 * (end_year - start_year + 1)
    if len(selected) != expected:
        raise FileNotFoundError(
            f"Expected {expected} RS/PS feature files for {start_year}-{end_year}; "
            f"found {len(selected)} under {feature_dir}"
        )
    return selected


def load_v2_labels(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {*KEYS, "zone", "made", "shot_value", "p_v2"}
    parts: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing v2 columns: {missing}")
        parts.append(frame[[*KEYS, "zone", "made", "shot_value", "p_v2"]].copy())
    labels = pd.concat(parts, ignore_index=True)
    labels["game_id"] = labels["game_id"].astype(str).str.zfill(10)
    labels["event_num"] = pd.to_numeric(labels["event_num"], errors="raise").astype(np.int64)
    labels["zone"] = labels["zone"].replace({"mid": "midrange"}).astype(str)
    duplicate = labels.duplicated(KEYS, keep=False)
    duplicate_rows = int(duplicate.sum())
    if duplicate_rows:
        conflict = (
            labels.loc[duplicate]
            .groupby(KEYS, observed=True)
            .agg(
                zones=("zone", "nunique"),
                made_values=("made", "nunique"),
                shot_values=("shot_value", "nunique"),
                probabilities=("p_v2", "nunique"),
            )
        )
        conflict = conflict[(conflict > 1).any(axis=1)]
        if not conflict.empty:
            examples = ["/".join(map(str, key)) for key in conflict.index[:5]]
            raise ValueError(
                "Overlapping native v2 parquets disagree on duplicate keys: "
                + ", ".join(examples)
            )
    labels = labels.drop_duplicates(KEYS, keep="first").reset_index(drop=True)
    if not labels["p_v2"].between(0.0, 1.0).all():
        raise ValueError("Native v2 probabilities must lie in [0, 1]")
    if not labels["zone"].isin(BUCKETS).all():
        raise ValueError("Native v2 zones must be mid/midrange or three")
    return labels, {
        "input_rows": int(sum(len(part) for part in parts)),
        "unique_rows": int(len(labels)),
        "overlap_duplicate_rows": duplicate_rows,
    }


def _load_transition_for_file(
    transition_dir: Path,
    *,
    year: int,
    season_type: str,
) -> pd.DataFrame:
    path = transition_dir / f"transition_backfill_{year}_{season_type}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    transition = pd.read_parquet(
        path,
        columns=["game_id", "event_num", "transition_probability"],
    )
    transition["game_id"] = transition["game_id"].astype(str).str.zfill(10)
    transition["event_num"] = pd.to_numeric(
        transition["event_num"], errors="raise"
    ).astype(np.int64)
    if transition.duplicated(["game_id", "event_num"]).any():
        raise ValueError(f"{path.name} contains duplicate shot keys")
    return transition


def load_feature_file(path: Path, transition_dir: Path) -> pd.DataFrame:
    year, season_type = parse_feature_filename(path)
    frame = pd.read_parquet(path, columns=FEATURE_COLUMNS)
    frame["game_id"] = frame["game_id"].astype(str).str.zfill(10)
    frame["event_num"] = pd.to_numeric(frame["event_num"], errors="raise").astype(np.int64)
    transition = _load_transition_for_file(
        transition_dir,
        year=year,
        season_type=season_type,
    )
    frame = frame.merge(
        transition,
        on=["game_id", "event_num"],
        how="left",
        validate="one_to_one",
    )
    if frame["transition_probability"].isna().any():
        raise ValueError(f"{path.name} has missing transition probability joins")
    if not frame["model_bucket"].isin(BUCKETS).all():
        raise ValueError(f"{path.name} contains unsupported model buckets")
    return frame


def load_feature_range(
    feature_dir: Path,
    transition_dir: Path,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    paths = discover_feature_paths(
        feature_dir,
        start_year=start_year,
        end_year=end_year,
    )
    return pd.concat(
        [load_feature_file(path, transition_dir) for path in paths],
        ignore_index=True,
    )


def _build_distiller(max_iter: int) -> Pipeline:
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    max_categories=254,
                ),
            ),
        ]
    )
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical, STUDENT_CATEGORICAL),
            ("numeric", numeric, STUDENT_NUMERIC),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=max_iter,
                    max_leaf_nodes=31,
                    min_samples_leaf=100,
                    l2_regularization=3.0,
                    categorical_features=list(range(len(STUDENT_CATEGORICAL))),
                    random_state=53,
                ),
            ),
        ]
    )


def fit_distiller(rows: pd.DataFrame, *, max_iter: int) -> Pipeline:
    model = _build_distiller(max_iter)
    model.fit(rows, _logit(rows["p_v2"]))
    return model


def predict_distiller(model: Pipeline, rows: pd.DataFrame) -> np.ndarray:
    return _sigmoid(model.predict(rows))


def fit_outcome_stack(
    portable_probability: Iterable[float],
    student_probability: Iterable[float],
    made: Iterable[int],
) -> LogisticRegression:
    features = np.column_stack(
        [_logit(portable_probability), _logit(student_probability)]
    )
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(features, np.asarray(made, dtype=int))
    return model


def predict_outcome_stack(
    model: LogisticRegression,
    portable_probability: Iterable[float],
    student_probability: Iterable[float],
) -> np.ndarray:
    features = np.column_stack(
        [_logit(portable_probability), _logit(student_probability)]
    )
    return model.predict_proba(features)[:, 1]


def probability_metrics(
    rows: pd.DataFrame,
    probability: Iterable[float],
    *,
    native_v2: Iterable[float] | None = None,
) -> dict[str, float]:
    p = _safe_probability(probability)
    y = rows["is_make"].astype(int).to_numpy()
    result = {
        "shots": int(len(rows)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_bias": float(p.mean() - y.mean()),
    }
    if native_v2 is not None:
        teacher = _safe_probability(native_v2)
        result.update(
            {
                "rmse_vs_v2": float(np.mean((p - teacher) ** 2) ** 0.5),
                "mae_vs_v2": float(np.mean(np.abs(p - teacher))),
                "correlation_vs_v2": float(np.corrcoef(p, teacher)[0, 1]),
            }
        )
    return result


def paired_game_brier_interval(
    rows: pd.DataFrame,
    candidate: Iterable[float],
    reference: Iterable[float],
    *,
    repetitions: int,
) -> dict[str, float]:
    y = rows["is_make"].astype(int).to_numpy()
    candidate_p = _safe_probability(candidate)
    reference_p = _safe_probability(reference)
    gain = (y - reference_p) ** 2 - (y - candidate_p) ** 2
    grouped = (
        pd.DataFrame(
            {
                "game_id": rows["game_id"].astype(str).to_numpy(),
                "gain": gain,
            }
        )
        .groupby("game_id", observed=True)
        .agg(gain_sum=("gain", "sum"), shots=("gain", "size"))
    )
    rng = np.random.default_rng(20260829)
    indices = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
    denominator = grouped["shots"].to_numpy()[indices].sum(axis=1)
    samples = grouped["gain_sum"].to_numpy()[indices].sum(axis=1) / denominator
    return {
        "brier_gain_vs_portable": float(gain.mean()),
        "brier_gain_ci_low": float(np.quantile(samples, 0.025)),
        "brier_gain_ci_high": float(np.quantile(samples, 0.975)),
    }


def choose_bridge_source(
    *,
    portable_brier: float,
    bridge_brier: float,
    portable_rmse_vs_v2: float,
    bridge_rmse_vs_v2: float,
) -> tuple[str, dict[str, bool]]:
    gates = {
        "actual_brier_no_regression": bridge_brier <= portable_brier,
        "native_v2_fidelity_improved": bridge_rmse_vs_v2 < portable_rmse_vs_v2,
    }
    return (
        "v2_distilled_stack" if all(gates.values()) else "portable_v3_fallback",
        gates,
    )


def _year_mask(rows: pd.DataFrame, years: tuple[int, int]) -> pd.Series:
    return rows["season_end_year"].between(years[0], years[1])


def fit_evaluation(
    rows: pd.DataFrame,
    *,
    max_iter: int,
    bootstrap_repetitions: int,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, dict[str, object]]]:
    packages: dict[str, object] = {}
    metric_rows: list[dict[str, object]] = []
    decisions: dict[str, dict[str, object]] = {}
    for bucket in BUCKETS:
        bucket_rows = rows[rows["model_bucket"].eq(bucket)].copy()
        train = bucket_rows[_year_mask(bucket_rows, EVALUATION_TRAIN_YEARS)]
        stack_rows = bucket_rows[_year_mask(bucket_rows, EVALUATION_STACK_YEARS)]
        test = bucket_rows[_year_mask(bucket_rows, EVALUATION_TEST_YEARS)].copy()
        if min(len(train), len(stack_rows), len(test)) == 0:
            raise ValueError(f"Empty chronological split for {bucket}")

        distiller = fit_distiller(train, max_iter=max_iter)
        stack_student = predict_distiller(distiller, stack_rows)
        stack = fit_outcome_stack(
            stack_rows["historical_sq_make_prob"],
            stack_student,
            stack_rows["is_make"],
        )
        test_student = predict_distiller(distiller, test)
        test_bridge = predict_outcome_stack(
            stack,
            test["historical_sq_make_prob"],
            test_student,
        )
        test["p_v2_historical_student"] = test_student
        test["p_v2_historical_stack"] = test_bridge

        model_probabilities = {
            "native_v2": test["p_v2"].to_numpy(),
            "portable_v3": test["historical_sq_make_prob"].to_numpy(),
            "v2_distiller_raw": test_student,
            "v2_distilled_stack": test_bridge,
        }
        model_metrics: dict[str, dict[str, float]] = {}
        for model_name, probability in model_probabilities.items():
            values = probability_metrics(
                test,
                probability,
                native_v2=test["p_v2"],
            )
            model_metrics[model_name] = values
            metric_rows.append(
                {
                    "model_bucket": bucket,
                    "model": model_name,
                    "split": "heldout_2023_2026",
                    **values,
                }
            )
        interval = paired_game_brier_interval(
            test,
            test_bridge,
            test["historical_sq_make_prob"],
            repetitions=bootstrap_repetitions,
        )
        source, gates = choose_bridge_source(
            portable_brier=model_metrics["portable_v3"]["brier"],
            bridge_brier=model_metrics["v2_distilled_stack"]["brier"],
            portable_rmse_vs_v2=model_metrics["portable_v3"]["rmse_vs_v2"],
            bridge_rmse_vs_v2=model_metrics["v2_distilled_stack"]["rmse_vs_v2"],
        )
        decisions[bucket] = {
            "selected_source": source,
            "gates": gates,
            **interval,
            "stack_coefficients": {
                "portable_v3_logit": float(stack.coef_[0, 0]),
                "v2_student_logit": float(stack.coef_[0, 1]),
                "intercept": float(stack.intercept_[0]),
            },
        }
        packages[bucket] = {"distiller": distiller, "stack": stack}
    return packages, pd.DataFrame(metric_rows), decisions


def fit_emission_models(
    rows: pd.DataFrame,
    decisions: dict[str, dict[str, object]],
    *,
    max_iter: int,
) -> dict[str, object]:
    packages: dict[str, object] = {}
    for bucket in BUCKETS:
        bucket_rows = rows[rows["model_bucket"].eq(bucket)].copy()
        distill_rows = bucket_rows[_year_mask(bucket_rows, EMISSION_DISTILL_YEARS)]
        stack_rows = bucket_rows[_year_mask(bucket_rows, EMISSION_STACK_YEARS)]
        distiller = fit_distiller(distill_rows, max_iter=max_iter)
        student = predict_distiller(distiller, stack_rows)
        stack = fit_outcome_stack(
            stack_rows["historical_sq_make_prob"],
            student,
            stack_rows["is_make"],
        )
        packages[bucket] = {
            "distiller": distiller,
            "stack": stack,
            "selected_source": decisions[bucket]["selected_source"],
        }
    return packages


def score_bridge_rows(
    rows: pd.DataFrame,
    emission_models: dict[str, object],
) -> pd.DataFrame:
    output = rows.copy()
    output["p_v2_historical_student"] = np.nan
    output["p_make"] = np.nan
    output["bridge_source"] = ""
    for bucket in BUCKETS:
        mask = output["model_bucket"].eq(bucket)
        if not mask.any():
            continue
        package = emission_models[bucket]
        student = predict_distiller(package["distiller"], output.loc[mask])
        stack = predict_outcome_stack(
            package["stack"],
            output.loc[mask, "historical_sq_make_prob"],
            student,
        )
        source = str(package["selected_source"])
        selected = (
            stack
            if source == "v2_distilled_stack"
            else output.loc[mask, "historical_sq_make_prob"].to_numpy()
        )
        output.loc[mask, "p_v2_historical_student"] = student
        output.loc[mask, "p_make"] = _safe_probability(selected)
        output.loc[mask, "bridge_source"] = source
    if output[["p_v2_historical_student", "p_make"]].isna().any().any():
        raise ValueError("Historical bridge scoring produced missing probabilities")
    if not output["p_make"].between(0.0, 1.0).all():
        raise ValueError("Historical bridge probabilities must lie in [0, 1]")
    output["zone"] = output["model_bucket"].replace({"midrange": "mid"})
    output["made"] = output["is_make"].astype(int)
    output["p_portable_v3"] = output["historical_sq_make_prob"]
    output["ev_pts"] = output["p_make"] * output["shot_value"]
    output["student_portable_gap"] = (
        output["p_v2_historical_student"] - output["p_portable_v3"]
    ).abs()
    return output[OUTPUT_COLUMNS].copy()


def score_historical_range(
    *,
    feature_dir: Path,
    transition_dir: Path,
    emission_models: dict[str, object],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for path in discover_feature_paths(
        feature_dir,
        start_year=start_year,
        end_year=end_year,
    ):
        rows = load_feature_file(path, transition_dir)
        scored = score_bridge_rows(rows, emission_models)
        outputs.append(scored)
        print(
            f"Scored {len(scored):,} shots from {path.name}",
            flush=True,
        )
    output = pd.concat(outputs, ignore_index=True)
    if output.duplicated(KEYS).any():
        raise ValueError("Historical bridge output contains duplicate shot keys")
    return output


def historical_audit(
    bridge: pd.DataFrame,
    safe_crossfit_path: Path | None,
) -> pd.DataFrame:
    work = bridge.copy()
    work["era"] = pd.cut(
        work["season_end_year"],
        bins=[1996, 2003, 2013, 2015],
        labels=["1997-2003", "2004-2013", "2014-2015"],
    ).astype(str)
    if safe_crossfit_path is not None and safe_crossfit_path.exists():
        safe = pd.read_parquet(
            safe_crossfit_path,
            columns=[
                "season_end_year",
                "game_id",
                "event_num",
                "safe_crossfit_make_prob",
            ],
            filters=[("season_end_year", "<=", int(work["season_end_year"].max()))],
        )
        safe["game_id"] = safe["game_id"].astype(str).str.zfill(10)
        work = work.merge(safe, on=KEYS, how="left", validate="one_to_one")
    else:
        work["safe_crossfit_make_prob"] = np.nan

    records: list[dict[str, object]] = []
    for (era, bucket), group in work.groupby(["era", "zone"], observed=True):
        for model, column in [
            ("historical_bridge", "p_make"),
            ("portable_v3", "p_portable_v3"),
            ("v2_student_raw", "p_v2_historical_student"),
            ("safe_crossfit_audit", "safe_crossfit_make_prob"),
        ]:
            selected = group[["made", column]].dropna()
            if selected.empty:
                continue
            probability = _safe_probability(selected[column])
            made = selected["made"].astype(int).to_numpy()
            records.append(
                {
                    "era": era,
                    "zone": bucket,
                    "model": model,
                    "shots": int(len(selected)),
                    "brier": float(brier_score_loss(made, probability)),
                    "log_loss": float(log_loss(made, probability, labels=[0, 1])),
                    "calibration_bias": float(probability.mean() - made.mean()),
                }
            )
    return pd.DataFrame(records)


def season_seam_audit(
    bridge: pd.DataFrame,
    native_v2: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    historical = bridge[
        ["season_end_year", "zone", "made", "p_make", "bridge_source"]
    ].rename(columns={"p_make": "series_probability", "bridge_source": "series_source"})
    native = native_v2[["season_end_year", "zone", "made", "p_v2"]].copy()
    native["zone"] = native["zone"].replace({"midrange": "mid"})
    native = native.rename(columns={"p_v2": "series_probability"})
    native["series_source"] = "native_v2"
    series = pd.concat([historical, native], ignore_index=True)
    records: list[dict[str, object]] = []
    for (season, zone, source), group in series.groupby(
        ["season_end_year", "zone", "series_source"], observed=True
    ):
        probability = _safe_probability(group["series_probability"])
        made = group["made"].astype(int).to_numpy()
        records.append(
            {
                "season_end_year": int(season),
                "zone": str(zone),
                "series_source": str(source),
                "shots": int(len(group)),
                "mean_probability": float(probability.mean()),
                "actual_make_rate": float(made.mean()),
                "calibration_bias": float(probability.mean() - made.mean()),
                "brier": float(brier_score_loss(made, probability)),
                "log_loss": float(log_loss(made, probability, labels=[0, 1])),
            }
        )
    audit = pd.DataFrame(records).sort_values(
        ["season_end_year", "zone"], kind="stable"
    )
    seam: dict[str, dict[str, float]] = {}
    for zone in ("mid", "three"):
        prior = audit[(audit["season_end_year"].eq(2015)) & audit["zone"].eq(zone)]
        native_first = audit[(audit["season_end_year"].eq(2016)) & audit["zone"].eq(zone)]
        if len(prior) != 1 or len(native_first) != 1:
            raise ValueError(f"Expected one 2015/2016 seam row for {zone}")
        left = prior.iloc[0]
        right = native_first.iloc[0]
        seam[zone] = {
            "probability_change_2016_minus_2015": float(
                right["mean_probability"] - left["mean_probability"]
            ),
            "actual_rate_change_2016_minus_2015": float(
                right["actual_make_rate"] - left["actual_make_rate"]
            ),
            "calibration_bias_change_2016_minus_2015": float(
                right["calibration_bias"] - left["calibration_bias"]
            ),
        }
    return audit, seam


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-parquets", nargs="+", type=Path, default=DEFAULT_V2_PARQUETS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--transition-dir", type=Path, default=DEFAULT_TRANSITION_DIR)
    parser.add_argument("--safe-crossfit", type=Path, default=DEFAULT_SAFE_CROSSFIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2015)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.start_year > args.end_year or args.end_year >= EVALUATION_TRAIN_YEARS[0]:
        raise ValueError("Historical bridge scoring years must end before 2016")

    labels, duplicate_audit = load_v2_labels(args.v2_parquets)
    features = load_feature_range(
        args.feature_dir,
        args.transition_dir,
        start_year=2016,
        end_year=2026,
    )
    training = features.merge(
        labels,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_v2"),
    )
    if len(training) != len(labels):
        raise ValueError(
            f"Native v2/feature coverage is incomplete: {len(training):,}/{len(labels):,}"
        )
    zone_mismatch = training["model_bucket"].ne(training["zone"])
    if zone_mismatch.any():
        raise ValueError(f"Native v2 zone mismatch on {int(zone_mismatch.sum())} shots")
    made_mismatch = training["is_make"].astype(int).ne(training["made"].astype(int))
    if made_mismatch.any():
        raise ValueError(f"Native v2 outcome mismatch on {int(made_mismatch.sum())} shots")

    _, metrics, decisions = fit_evaluation(
        training,
        max_iter=args.max_iter,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    emission_models = fit_emission_models(
        training,
        decisions,
        max_iter=args.max_iter,
    )
    bridge = score_historical_range(
        feature_dir=args.feature_dir,
        transition_dir=args.transition_dir,
        emission_models=emission_models,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    audit = historical_audit(bridge, args.safe_crossfit)
    seam_audit, seam_summary = season_seam_audit(bridge, labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "russell_v2_historical_bridge_models.joblib"
    metrics_path = args.output_dir / "heldout_overlap_metrics.csv"
    decision_path = args.output_dir / "bridge_selection.json"
    output_path = (
        args.output_dir
        / f"russell_v2_historical_bridge_ev_{args.start_year}_{args.end_year}.parquet"
    )
    audit_path = args.output_dir / "historical_outcome_audit.csv"
    seam_audit_path = args.output_dir / "season_seam_audit.csv"
    manifest_path = args.output_dir / "manifest.json"

    artifact = {
        "schema_version": 1,
        "student_categorical": STUDENT_CATEGORICAL,
        "student_numeric": STUDENT_NUMERIC,
        "evaluation_splits": {
            "distiller_train": list(EVALUATION_TRAIN_YEARS),
            "stack_train": list(EVALUATION_STACK_YEARS),
            "heldout_test": list(EVALUATION_TEST_YEARS),
        },
        "emission_splits": {
            "distiller_train": list(EMISSION_DISTILL_YEARS),
            "stack_train": list(EMISSION_STACK_YEARS),
        },
        "decisions": decisions,
        "models": emission_models,
    }
    joblib.dump(artifact, model_path)
    metrics.to_csv(metrics_path, index=False)
    decision_path.write_text(
        json.dumps(decisions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bridge.to_parquet(output_path, index=False)
    audit.to_csv(audit_path, index=False)
    seam_audit.to_csv(seam_audit_path, index=False)

    manifest = {
        "status": "research_historical_bridge_complete",
        "native_v2_unchanged_from_season_end_year": 2016,
        "historical_scoring_years": [args.start_year, args.end_year],
        "historical_rows": int(len(bridge)),
        "historical_unique_games": int(bridge["game_id"].nunique()),
        "historical_zone_rows": {
            str(zone): int(count)
            for zone, count in bridge["zone"].value_counts().sort_index().items()
        },
        "bridge_sources": {
            str(source): int(count)
            for source, count in bridge["bridge_source"].value_counts().sort_index().items()
        },
        "v2_overlap_audit": duplicate_audit,
        "v2_feature_join_coverage": 1.0,
        "v2_zone_mismatches": 0,
        "v2_outcome_mismatches": 0,
        "decisions": decisions,
        "seam_2015_to_2016": seam_summary,
        "target_identity_max_gap": float(
            (bridge["ev_pts"] - bridge["p_make"] * bridge["shot_value"]).abs().max()
        ),
        "source_hashes": {
            str(path): sha256_file(path) for path in args.v2_parquets
        },
        "artifacts": {
            "model": str(model_path),
            "heldout_metrics": str(metrics_path),
            "selection": str(decision_path),
            "historical_ev": str(output_path),
            "historical_audit": str(audit_path),
            "season_seam_audit": str(seam_audit_path),
        },
        "guardrails": [
            "Native Russell ShotQuality v2 remains authoritative from 2016 onward.",
            "No pre-2016 outcome fits the emitted bridge.",
            "Safe-crossfit probabilities are audit-only and are not emitted as VPM targets.",
            "A failed bucket uses the portable v3 fallback explicitly.",
            "The bridge is retrospective research and is not untouched predictive confirmation.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(decisions, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {len(bridge):,} historical EV rows to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
