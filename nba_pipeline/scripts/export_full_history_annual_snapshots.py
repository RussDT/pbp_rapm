"""Pure dated queries of the independent full-history library on strict masks.

This is parameter/mean TRANSPORT, not a source-clean hyperparameter refit.
The completed library, old CORE models, impact fits and selected VPM are never
modified. Every snapshot is replayed separately: inserting a query into an OU
history can move the season-variance injection and change subsequent states.
Only terminal, missing-measurement query rows are added to each source prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_full_skill_history import Engine
from export_full_skill_history import derived, audit as audit_derived
from full_skill_history_specs import SKILLS, GENERIC_SKILLS

PROFILE = "independent_transport_source_v2"
KEYS = ["query_id", "player_id", "query_date", "query_season", "cohort_vintage"]
BOUND = pd.Timestamp("2026-09-01")
ZONES = ["at_rim", "short_midrange", "long_midrange", "corner3", "arc3"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize_queries(frame):
    q = frame.copy()
    if not set(KEYS) <= set(q) or q["query_id"].isna().any() or q.query_id.duplicated().any():
        raise ValueError("Explicit unique query IDs and complete query keys required")
    q["query_date"] = pd.to_datetime(q.query_date, utc=True).dt.tz_localize(None)
    if (q.query_date.isna().any() or not q.query_date.eq(q.query_date.dt.normalize()).all()
            or not q.query_date.between("2014-10-01", BOUND).all()):
        raise ValueError("Date-only historical queries through September1,2026 required")
    for name in ["player_id", "query_season"]:
        x = pd.to_numeric(q[name], errors="raise")
        if x.isna().any() or not np.equal(x, np.floor(x)).all():
            raise ValueError("Exact integer player/season required")
        q[name] = x.astype(np.int64)
    if q.player_id.le(0).any() or not q.query_season.between(2015, 2026).all():
        raise ValueError("Positive identities and no2027 query/recipe required")
    q["cohort_vintage"] = pd.to_numeric(q.cohort_vintage, errors="raise").astype("Int64")
    if q.duplicated(["player_id", "query_date", "query_season"]).any():
        raise ValueError("One canonical query per player/date/ending season; no duplicate roles")
    if q.groupby("query_date").query_season.nunique().gt(1).any():
        raise ValueError("Ambiguous ending season on one common snapshot")
    return q[KEYS].reset_index(drop=True)


def build_queries(static_root, universe_path, annual_roots, current_manifest):
    """Union actual cohort/annual/current cuts, preserving one baseline universe.

    Cohort-vintage is an annual-start-year identifier, not an ending season.
    Its immutable snapshot contract is ending season = cohort vintage +1;
    actual annual producer manifests independently check this on shared cuts.
    """
    static_root, universe_path = Path(static_root), Path(universe_path)
    universe = pd.read_parquet(universe_path).player_id.to_numpy(np.int64)
    if len(np.unique(universe)) != len(universe) or (universe <= 0).any():
        raise ValueError("Exact original baseline universe required")
    inputs = {str(universe_path.resolve()): sha(universe_path)}
    dates = {}
    for vintage in range(2015, 2026):
        path = static_root / f"{vintage}.parquet"
        f = pd.read_parquet(path, columns=["player_id", "snapshot_date", "cohort_vintage"])
        if f.snapshot_date.nunique() != 1 or not f.cohort_vintage.eq(vintage).all():
            raise ValueError("Immutable annual static snapshot identity differs")
        if not set(f.player_id) <= set(universe):
            raise ValueError("Original static packet is outside the shared universe")
        cut = pd.Timestamp(f.snapshot_date.iloc[0]).normalize()
        dates[cut] = (vintage + 1, vintage)
        inputs[str(path.resolve())] = sha(path)
    for root in annual_roots:
        for path in sorted((Path(root) / "forecasts/annual/dates").glob("*/manifest.json")):
            m = json.loads(path.read_text())
            cut, season = pd.Timestamp(m["estimate_date"]), int(m["query_season"])
            if cut not in dates or dates[cut][0] != season:
                raise ValueError("Actual annual deployment does not match a cohort query cut")
            inputs[str(path.resolve())] = sha(path)
    path = Path(current_manifest)
    m = json.loads(path.read_text())
    cut, season = pd.Timestamp(m["estimate_date"]), int(m["query_season"])
    if cut != BOUND or season != 2026 or int(m["full_queried_players"]) != len(universe):
        raise ValueError("Actual common current cut/universe differs")
    dates[cut] = (season, None)
    inputs[str(path.resolve())] = sha(path)
    rows = []
    for cut, (season, vintage) in sorted(dates.items()):
        for player in universe:
            rows.append(dict(query_id=f"lib:{cut.date()}:{season}:{player}", player_id=int(player),
                             query_date=cut, query_season=season, cohort_vintage=vintage))
    q = normalize_queries(pd.DataFrame(rows))
    q.attrs["input_hashes"] = inputs
    return q


def source_dependencies(column):
    """Counts whose independently established uncertainty reaches this field.

    Report fields retain their own source validity flags. Legacy PF is not the
    all-foul/type definition and therefore cannot gate unrelated new fouls.
    """
    if not column.startswith("core."):
        return []
    n = column[5:]
    if n.startswith(tuple(z + "_" for z in ZONES)):
        zone = n.split("_f")[0]
        tag = "rim" if zone == "at_rim" else "mid" if zone in ZONES[1:3] else "three"
        return [tag + ("_fga" if n.endswith("_fga") else "_fgm")]
    for stem in ["tov", "oreb", "dreb"]:
        if n.startswith(stem):
            return [stem]
    if n.startswith("ast_"):
        return ["ast"]
    if n in ["fta", "ftm", "stl", "blk"]:
        return [n]
    if n == "and_one":
        return ["fgm", "fta"]
    if n.endswith("_points"):
        return ["pts"]
    return []


def strict_observations(panel, validated):
    p, v = panel.copy(), validated.copy()
    p.game_id = p.game_id.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    v.game_id = v.game_id.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    if p.duplicated(["player_id", "game_id"]).any() or v.duplicated(["player_id", "game_id"]).any():
        raise ValueError("Ambiguous observation source identity")
    index = pd.MultiIndex.from_frame(p[["player_id", "game_id"]])
    positions = index.get_indexer(pd.MultiIndex.from_frame(v[["player_id", "game_id"]]))
    if (positions < 0).any():
        raise ValueError("Strict source has a query absent from the original full history")
    if not np.array_equal(pd.to_datetime(p.game_date.iloc[positions]).to_numpy(),
                          pd.to_datetime(v.game_date).to_numpy()):
        raise ValueError("Strict source exact identity/date mismatch")
    # Denominators have the same original admission; verify before relying on it.
    for name in ["off_poss", "def_poss"]:
        a, b = p[name].iloc[positions].to_numpy(), v[name].to_numpy()
        if not np.array_equal(a, b, equal_nan=True):
            raise ValueError("Full-history and validated possession definitions differ")
    changes = {}
    raw = sorted({x for s in SKILLS for x in [s.numerator, s.denominator] if x.startswith("core.")})
    for column in raw:
        dependencies = source_dependencies(column)
        if not dependencies:
            continue
        unsafe = np.zeros(len(v), dtype=bool)
        for name in dependencies:
            unsafe |= v[name].isna().to_numpy()
        selected = positions[unsafe]
        changes[column] = int(p[column].iloc[selected].notna().sum())
        p.loc[p.index[selected], column] = np.nan
    return p, changes


def validate_recipe(recipe, year, cut):
    if year >= 2027 or recipe["year"] != year or recipe["training_end_season"] >= year:
        raise ValueError("Wrong/protected annual parameter vintage")
    if recipe["status"] not in ["bootstrap", "annual_past_only_fit"]:
        raise ValueError("Unknown parameter recipe status")
    last = recipe.get("last_training_observation")
    if last is not None and pd.Timestamp(last) >= pd.Timestamp(cut):
        raise ValueError("Actual parameter training observation reaches query date")
    if last is None and recipe["status"] != "bootstrap":
        raise ValueError("Fitted recipe lacks actual fit-end provenance")
    a = np.asarray(recipe["parameters"], float)
    if a.shape != (6,) or not np.isfinite(a).all() or not np.isfinite(recipe["mean"]):
        raise ValueError("Invalid saved OU recipe")
    if min(a[0], a[2], a[5]) <= 0 or min(a[1], a[3], a[4]) < 0:
        raise ValueError("Invalid OU time/noise/variance parameters")


def replay_snapshot(panel, queries, specs, recipes, library):
    """One independent terminal snapshot; no source update from query rows."""
    q = normalize_queries(queries)
    if q.query_date.nunique() != 1 or q.query_season.nunique() != 1:
        raise ValueError("Replay snapshots separately to avoid query side effects")
    cut, year = q.query_date.iloc[0], int(q.query_season.iloc[0])
    prefix = panel.loc[pd.to_datetime(panel.game_date).lt(cut)].copy()
    prefix["_query_row"] = -1
    query = pd.DataFrame(index=np.arange(len(q)), columns=prefix.columns)
    query["player_id"] = q.player_id.to_numpy()
    query["game_date"] = cut
    query["season"] = year
    query["day"] = int((cut - pd.Timestamp("1970-01-01")).days)
    query["_query_row"] = np.arange(len(q))
    full = pd.concat([prefix, query], ignore_index=True).sort_values(
        ["player_id", "game_date", "_query_row"], kind="stable").reset_index(drop=True)
    take = np.flatnonzero(full._query_row.ge(0).to_numpy())
    order = np.argsort(full._query_row.iloc[take].to_numpy())
    rows, values = [], {}
    for spec in specs:
        recipe = recipes[spec.name]
        validate_recipe(recipe, year, cut)
        engine = Engine(full, spec, library)
        result = engine.run(recipe["parameters"], recipe["mean"])
        prediction = result[0][take][order] * (1 if spec.kind == "pct" else 100)
        exposure = result[3][take][order]
        day = result[5][take][order]
        first = engine.day[engine.exposure > 0].min(initial=np.inf)
        supported = bool(spec.first_season <= year <= spec.last_season and first < query.day.iloc[0])
        observed = (exposure > 0) & (day >= 0)
        legal = supported & observed
        last = pd.to_datetime(np.where(day >= 0, day, np.nan), unit="D", origin="1970-01-01")
        if (pd.Series(last).dropna() >= cut).any():
            raise ValueError("Current/future measurement in common snapshot")
        values[spec.name] = np.where(legal, prediction, np.nan)
        r = q.copy()
        r["head_name"] = spec.name
        r["prediction"] = values[spec.name]
        r["source_supported"] = supported
        r["player_observed"] = observed
        r["prior_exposure"] = exposure
        r["last_observation_date"] = last
        r["parameter_year"] = year
        r["parameter_last_training_observation"] = pd.to_datetime(recipe.get("last_training_observation"))
        r["parameter_bootstrap"] = recipe["status"] == "bootstrap"
        r["model_sha256"] = recipe.get("_sha256", "")
        rows.append(r)
    return pd.DataFrame(values), pd.concat(rows, ignore_index=True)


def derived_dag():
    d = {}
    for z in ZONES:
        d[z + "_fgm100"] = [z + "_fga100", z + "_pct"]
        d[z + "_assisted_fgm100"] = [z + "_fgm100", z + "_assisted_make_share"]
        d[z + "_unassisted_fgm100"] = [z + "_fgm100", z + "_assisted_fgm100"]
    for tag, children in [("mid", ZONES[1:3]), ("three", ZONES[3:]), ("two", ZONES[:3])]:
        for stat in ["fga100", "fgm100", "assisted_fgm100", "unassisted_fgm100"]:
            d[tag + "_" + stat] = [z + "_" + stat for z in children]
        d[tag + "_pct"] = [tag + "_fgm100", tag + "_fga100"]
    d.update(fga100=[z + "_fga100" for z in ZONES], fgm100=[z + "_fgm100" for z in ZONES],
             ftm100=["fta100", "ft_pct"], points100=["two_fgm100", "three_fgm100", "ftm100"],
             efg_pct=["fgm100", "three_fgm100", "fga100"], approx_tsa100=["fga100", "fta100"],
             approx_ts_pct=["points100", "approx_tsa100"], ast100=["assists_" + z + "100" for z in ZONES],
             usage_events100=["fga100", "fta100", "tov100"],
             all_fouls100_total_possessions=["foul_" + k + "100" for k in
                 ["shooting", "offensive", "technical", "flagrant", "loose_ball", "double", "personal_group"]])
    for stem in ["bad_pass_any", "lost_ball_any", "traveling", "remainder"]:
        d["tov_" + stem + "100"] = ["tov100", "tov_" + stem + "_share"]
    for stem in ["bad_pass", "lost_ball"]:
        d["tov_" + stem + "_oob100"] = ["tov_" + stem + "_any100", "tov_" + stem + "_oob_share"]
    for reb in ["oreb", "dreb"]:
        for shot in ["2pt", "3pt", "ft", "other"]:
            d[reb + "_after_" + shot + "100"] = [reb + "100", reb + "_after_" + shot + "_share"]
    for stem in ["catch_shoot", "pull_up"]:
        for side in ["two", "three"]:
            d[stem + "_" + side + "_fgm100"] = [stem + "_" + side + "_fga100", stem + "_" + side + "_pct"]
        for stat in ["fga100", "fgm100"]:
            d[stem + "_" + stat] = [stem + "_two_" + stat, stem + "_three_" + stat]
        d[stem + "_pct"] = [stem + "_fgm100", stem + "_fga100"]
    for stem in ["overall_def", "two_pt_def", "three_pt_def", "less_6ft_def", "less_10ft_def", "more_15ft_def"]:
        d[stem + "_fgm100"] = [stem + "_fga100", stem + "_fg_pct"]
    d["drive_fgm100"] = ["drive_fga100", "drive_pct"]
    d["assist_points_created100"] = ["assists_" + z + "100" for z in ZONES]
    d["unassisted_fg_points100"] = ["two_unassisted_fgm100", "three_unassisted_fgm100"]
    d["defended_fg_points100"] = ["two_pt_def_fgm100", "three_pt_def_fgm100"]
    return d


def primitive_dependencies(name, era):
    generic = {s.name for s in GENERIC_SKILLS}
    names = {s.name for s in SKILLS}
    # These canonical proportions are deterministically renormalized together.
    groups = [["tov_" + x + "_share" for x in ["bad_pass_any", "lost_ball_any", "traveling", "remainder"]]]
    groups += [[r + "_after_" + x + "_share" for x in ["2pt", "3pt", "ft", "other"]] for r in ["oreb", "dreb"]]
    for group in groups:
        if name in group:
            return sorted(group)
    if name in generic:
        if name in names:
            return [name]
        selected = name + ("__v1" if era == "through_2024" else "__v2")
        if selected not in names:
            raise ValueError("No declared source-vintage branch")
        return [selected]
    return sorted({head for dep in derived_dag()[name] for head in primitive_dependencies(dep, era)})


def feature_schema(model_registry=None):
    generic = {s.name: s for s in GENERIC_SKILLS}
    dag = derived_dag()
    names = sorted(generic) + list(dag)
    if len(names) != 203 or len(set(names)) != 203:
        raise ValueError("Fixed203-field inventory changed")
    overlap = {"at_rim_fga100", "at_rim_pct", "fta100", "ft_pct", "tov100", "oreb100", "dreb100", "stl100", "blk100"}
    records = []
    for name in names:
        side = "both"
        if name in generic:
            spec = generic[name]
            block = spec.family
            if block in ["shooting", "assistedness", "passing", "creation", "turnovers", "scoring_context", "shooting_context", "defender_distance"]:
                side = "off"
            elif block == "defended_shooting" or name.startswith(("dreb", "stl", "blk")):
                side = "def"
            elif name.startswith("oreb"):
                side = "off"
        else:
            block = "coherent_products"
            side = "def" if name.startswith(("dreb", "overall_def", "two_pt_def", "three_pt_def", "less_", "more_", "defended_")) else "off"
        kind = "overlap" if name in overlap else "new" if name in generic else "engineered"
        physical = "lib__" + name
        eras = {era: primitive_dependencies(name, era) for era in ["through_2024", "from_2025"]}
        records.append(dict(name=physical, block=block, kind=kind, native_side=side,
            transform="identity", initial_registry=kind != "engineered",
            exposure_column=physical + "__exposure", last_observation_column=physical + "__last_observation_date",
            exposure_semantics="minimum_earlier_dependency_exposure_support_gate_only_not_an_estimator_weight",
            dependencies=sorted(set(eras["through_2024"] + eras["from_2025"])), source_heads_by_era=eras,
            dependency_dag=dag.get(name, []), units=(generic[name].record()["units"] if name in generic else
                "fraction" if name.endswith("_pct") else "per_100_total_poss" if name == "all_fouls100_total_possessions"
                else "per_100_def_poss" if side == "def" else "per_100_off_poss")))
    return dict(schema_version=1, source_profile=PROFILE, records=records, model_registry=model_registry or [],
                head_registry=[s.record() for s in SKILLS],
                metadata=dict(parameter_transport_not_refit=True, cross_stat_used=False,
                    canonical_source_switch_ending_season=2025, missing_v2_never_filled_from_v1=True,
                    old_baseline_inputs_untouched=True, source_measurements_required=True,
                    query_rows_do_not_update_history=True, snapshot_replays_independent=True))


def make_features(values, provenance, queries, schema):
    q = normalize_queries(queries)
    frame = values.copy()
    frame["season"] = q.query_season.to_numpy()
    frame = derived(frame)
    frame["assist_points_created100"] = 2 * sum(frame["assists_" + z + "100"] for z in ZONES[:3]) + 3 * sum(frame["assists_" + z + "100"] for z in ZONES[3:])
    frame["unassisted_fg_points100"] = 2 * frame.two_unassisted_fgm100 + 3 * frame.three_unassisted_fgm100
    frame["defended_fg_points100"] = 2 * frame.two_pt_def_fgm100 + 3 * frame.three_pt_def_fgm100
    audit_derived(frame)
    lookup = {head: f.set_index("query_id").reindex(q.query_id) for head, f in provenance.groupby("head_name", sort=False)}
    columns = {}
    for r in schema["records"]:
        name = r["name"][5:]
        prediction = frame[name].to_numpy(float)
        exposure = np.zeros(len(q))
        last = np.full(len(q), np.datetime64("NaT"), dtype="datetime64[ns]")
        legal = np.zeros(len(q), bool)
        for era, take in [("through_2024", q.query_season.le(2024).to_numpy()), ("from_2025", q.query_season.ge(2025).to_numpy())]:
            heads = r["source_heads_by_era"][era]
            parts = [lookup[h] for h in heads]
            x = np.stack([p.prior_exposure.to_numpy(float) for p in parts], axis=1)
            days = np.stack([p.last_observation_date.to_numpy(dtype="datetime64[ns]") for p in parts], axis=1)
            available = np.stack([(p.source_supported & p.player_observed).to_numpy(bool) for p in parts], axis=1).all(axis=1)
            exposure[take] = x.min(axis=1)[take]
            last[take] = days.max(axis=1)[take]
            legal[take] = available[take]
        columns[r["name"]] = np.where(legal, prediction, np.nan)
        columns[r["exposure_column"]] = exposure
        columns[r["last_observation_column"]] = last
    return pd.concat([q, pd.DataFrame(columns)], axis=1)


def export_queries(full_history_root, validated_source_root, queries, out, *, profile=PROFILE):
    if profile != PROFILE:
        raise ValueError("Only the registered independent transport profile is implemented")
    root, source, out = Path(full_history_root).resolve(), Path(validated_source_root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new immutable export root")
    query_inputs = dict(queries.attrs.get("input_hashes", {}))
    q = normalize_queries(queries)
    lock = json.loads((root / "run_lock.json").read_text())
    for name, key in [("observations.parquet", "observations_sha256"), ("registry.json", "registry_sha256")]:
        if sha(root / name) != lock[key]:
            raise ValueError("Original full-history observation/registry pin changed")
    library = root / "kernel" / ("full_skill_" + lock["kernel_sha256"][:16] + ".so")
    files = [root / "run_lock.json", root / "observations.parquet", root / "registry.json", library,
             source / "manifest.json", source / "player_game_lines.parquet", Path(__file__),
             Path(__file__).with_name("build_full_skill_history.py"), Path(__file__).with_name("full_skill_history_specs.py"),
             Path(__file__).with_name("export_full_skill_history.py"), Path(__file__).with_name("full_skill_history_kernel.cpp"),
             Path(__file__).parents[1] / "config/full_skill_history_sources.json"]
    if sha(files[-2]) != lock["kernel_sha256"]:
        raise ValueError("Original replay kernel source changed")
    inputs = {**query_inputs, **{str(p.resolve()): sha(p) for p in files}}
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError("Query/source input changed")
    panel, masks = strict_observations(pd.read_parquet(root / "observations.parquet"), pd.read_parquet(source / "player_game_lines.parquet"))
    if not panel.season.between(1997, 2026).all() or pd.to_datetime(panel.game_date).ge(BOUND).any():
        raise ValueError("Protected/future observation source")
    registry, recipes = [], {}
    for year in sorted(q.query_season.unique()):
        recipes[int(year)] = {}
        for spec in SKILLS:
            path = root / "models" / str(year) / (spec.name + ".json")
            r = json.loads(path.read_text())
            if r.get("skill") != spec.name or r["run_lock_sha256"] != sha(root / "run_lock.json"):
                raise ValueError("Parameter recipe is not bound to original full-history observations")
            r["_sha256"] = sha(path)
            inputs[str(path)] = r["_sha256"]
            recipes[int(year)][spec.name] = r
            registry.append(dict(head_name=spec.name, parameter_year=int(year), model_path=str(path), model_sha256=r["_sha256"],
                parameter_last_training_observation=r.get("last_training_observation"), parameter_bootstrap=r["status"] == "bootstrap"))
    schema = feature_schema(registry)
    packets, provenance = [], []
    for _, batch in q.groupby("query_date", sort=True):
        batch = batch.reset_index(drop=True)
        values, heads = replay_snapshot(panel, batch, SKILLS, recipes[int(batch.query_season.iloc[0])], library)
        packets.append(make_features(values, heads, batch, schema))
        provenance.append(heads)
        print(json.dumps(dict(stage="pure_snapshot", date=str(batch.query_date.iloc[0].date()), queries=len(batch), heads=len(SKILLS))), flush=True)
    wide = pd.concat(packets, ignore_index=True).set_index("query_id").reindex(q.query_id).reset_index()
    long = pd.concat(provenance, ignore_index=True)
    if len(wide) != len(q) or len(long) != len(q) * len(SKILLS) or not wide[KEYS].equals(q):
        raise ValueError("Exact ordered query census changed")
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError("Input changed during source replay")
    out.mkdir(parents=True)
    wide.to_parquet(out / "feature_packet.parquet", index=False)
    long.to_parquet(out / "head_provenance.parquet", index=False)
    q.to_parquet(out / "queries.parquet", index=False)
    write_json(out / "feature_schema.json", schema)
    coverage = []
    for cut, f in wide.groupby("query_date", sort=True):
        for r in schema["records"]:
            coverage.append(dict(query_date=str(cut.date()), feature=r["name"], finite_rows=int(f[r["name"]].notna().sum()), query_rows=len(f)))
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    receipt = dict(status="passed_dated_feature_queries", profile=PROFILE, feature_schema_sha256=sha(out / "feature_schema.json"),
        query_rows=len(q), query_dates=int(q.query_date.nunique()), source_heads=len(SKILLS), catalog_features=len(schema["records"]),
        parameter_transport_not_refit=True, original_parameter_mean_transport=True, strict_mask_changes=masks,
        strict_mask_source_first_season=2014, pre2014_original_library_admission_retained=True,
        source_observations_used_before_each_query=True, parameter_dates_checked_individually=True,
        cross_stat_coefficients_used=False, old_baseline_features_modified=False, protected_2027_inputs=0,
        missing_measurements_never_zero_filled=True, original_source_files_modified=False,
        input_hashes=inputs, output_hashes={str(p): sha(p) for p in out.iterdir() if p.is_file()})
    write_json(out / "receipt.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-history-root", type=Path, required=True)
    p.add_argument("--validated-source-root", type=Path, required=True)
    p.add_argument("--static-root", type=Path, required=True)
    p.add_argument("--universe", type=Path, required=True)
    p.add_argument("--annual-root", type=Path, action="append", required=True)
    p.add_argument("--current-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    q = build_queries(args.static_root, args.universe, args.annual_root, args.current_manifest)
    result = export_queries(args.full_history_root, args.validated_source_root, q, args.out)
    print(json.dumps({k: result[k] for k in ["status", "query_rows", "query_dates", "catalog_features"]}))


if __name__ == "__main__":
    main()
