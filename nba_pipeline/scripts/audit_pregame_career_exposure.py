"""Pin unresolved career-exposure evidence and common comparison exclusions.

Native target-row appearances are diagnostic counts, not substitutes for the
selected source's possession definition. No exposure is recovered or filled;
whole future games requiring unidentified entrant weights are listed explicitly.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

from audit_pregame_target_scoreboards import normal
from prepare_full_skill_history import read_table, sha
from build_full_skill_history import write_json

ROOT = Path(__file__).resolve().parents[1]
VPM = ROOT.parent.parent / "vpm"
sys.path.insert(0, str(VPM))
from vpm.harness.pregame_entrants import load_inputs, legal_model
from vpm.harness.field_benchmark import snapshot_date

GAMES = ["0021500278", "0021701046", "0021800388", "0021900376", "0021900539", "0022100105", "0022400763"]
SLOTS = [f"{side}{n}" for side in ["O", "D"] for n in range(1, 6)]
SOURCE = ROOT / "results/estimated_skills/pregame_source_validated_v2"
ENTRANTS = VPM / "vpm/results/harness/pregame_entrants_v4"
OUT = ROOT / "results/estimated_skills/pregame_career_exposure_audit_v1"


def required_game_exclusions(queries, exposures, existing_admission):
    """Only actual positive deployment exposure requires an unknown weight."""
    q = queries.loc[queries.background_model_available.eq(True), ["game_id", "player_id"]].drop_duplicates()
    active = exposures.loc[exposures.exposure.gt(0), ["game_id", "player_id"]].drop_duplicates()
    affected = q.merge(active, on=["game_id", "player_id"], validate="one_to_one")
    game = affected.groupby("game_id").player_id.agg(lambda a: ",".join(map(str, sorted(a)))).rename("unknown_player_ids").reset_index()
    game = game.merge(existing_admission[["game_id", "admitted_target_game"]], on="game_id", how="left", validate="one_to_one")
    game["additional_common_exclusion"] = game.admitted_target_game.eq(True)
    game["reason"] = "unidentified_required_entrant_career_weight"
    return game


def run(source=SOURCE, entrants=ENTRANTS, out=OUT):
    source, entrants, out = Path(source).resolve(), Path(entrants).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new immutable career-exposure audit directory")
    pins = {}
    def pin(path):
        path = Path(path).resolve(); pins[str(path)] = sha(path); return path
    pin(__file__)
    pin(Path(__file__).with_name("prepare_full_skill_history.py"))
    pin(Path(__file__).with_name("audit_pregame_target_scoreboards.py"))
    entry_lock = json.loads(pin(entrants / "run_lock.json").read_text())
    for name, digest in entry_lock["source_hashes"].items():
        if sha(name) != digest:
            raise ValueError("Entrant source changed before independent audit")
        pins[name] = digest
    query_path = pin(entrants / "unknown_experience_queries.csv")
    unknown = pd.read_csv(query_path, dtype={"game_id":str})
    unknown.game_id = normal(unknown.game_id)
    admission_path = pin(VPM / "vpm/results/harness/vpm_pregame_neutral_core_v1/native_admission/games.csv")
    admission = pd.read_csv(admission_path, dtype={"game_id":str})
    admission.game_id = normal(admission.game_id)
    core_root = source.parent / "full_history_v2/source"
    core, manifest = read_table(core_root, "player_game")
    pin(core_root / "player_game/manifest.json")
    for item in manifest["objects"]:
        pin(item["local_file"])
    core.game_id = normal(core.game_id)
    unsafe = core.loc[core.game_id.isin(GAMES)].copy()
    if len(unsafe) != 145 or unsafe.game_id.nunique() != 7:
        raise ValueError("Unsafe career source census changed")
    score = pd.read_csv(pin(source / "completeness_games.csv"), dtype={"game_id":str})
    records, controls, native_exposure = [], [], []
    # Every native year is inspected for rolling-query footprint, not only the
    # seven historical source games. The same full-game exclusion serves arms.
    for year in range(2016, 2027):
        for phase in ["RS", "PS"]:
            path = pin(ROOT / f"results/monthly_vpm_targets_clean_sq_v2/atoms_{year}_{phase}.parquet")
            atoms = pd.read_parquet(path, columns=["game_id", "actual_points", *SLOTS])
            atoms.game_id = normal(atoms.game_id)
            if not atoms[SLOTS].gt(0).all().all():
                raise ValueError("Native footprint contains incomplete player slots")
            relevant = set(unknown.game_id)
            queried = atoms.loc[atoms.game_id.isin(relevant)]
            for side in ["O", "D"]:
                melted = queried[["game_id", *[f"{side}{i}" for i in range(1,6)]]].melt(id_vars="game_id", value_name="player_id")
                e = melted.groupby(["game_id", "player_id"]).size().rename("exposure").reset_index()
                native_exposure.append(e)
            if phase != "RS":
                continue
            for gid in GAMES:
                rows = atoms.loc[atoms.game_id.eq(gid)]
                if rows.empty:
                    continue
                source_rows = unsafe.loc[unsafe.game_id.eq(gid)]
                terminal = score.loc[score.game_id.eq(gid)].iloc[0]
                records.append(dict(game_id=gid, season=year, source_rows=len(source_rows),
                    source_off_sum=int(source_rows.off_poss.sum()), source_def_sum=int(source_rows.def_poss.sum()),
                    source_career_mass=float((source_rows.off_poss+source_rows.def_poss).sum()/2),
                    native_rows=len(rows), native_off_sum=5*len(rows), native_def_sum=5*len(rows),
                    native_points=float(rows.actual_points.sum()), terminal_score=float(terminal.terminal_score),
                    native_matches_terminal=bool(abs(rows.actual_points.sum()-terminal.terminal_score)<1e-8),
                    source_lineups_clean=bool(source_rows.lineups_clean.all()),
                    source_provably_wrong_rows=int(source_rows.provably_wrong_rows.max()),
                    source_missing_ten_rows=int(source_rows.rows_without_10_players.max()),
                    recovery_admitted=False, reason="different_possession_definition_and_unsafe_source_lineups"))
            # Clean controls demonstrate the definition difference independently
            # of the seven flagged source games. Keep a small deterministic set.
            for gid in atoms.game_id.drop_duplicates().head(5):
                c = core.loc[core.game_id.eq(gid)]
                if c.empty or not c.lineups_clean.all() or c.provably_wrong_rows.max() or c.rows_without_10_players.max():
                    continue
                n = int(atoms.game_id.eq(gid).sum())
                controls.append(dict(game_id=gid, season=year, source_off_sum=int(c.off_poss.sum()), native_off_sum=5*n,
                    source_def_sum=int(c.def_poss.sum()), native_def_sum=5*n))
    evidence = pd.DataFrame(records).sort_values("game_id")
    if evidence.game_id.tolist() != sorted(GAMES) or not evidence.native_matches_terminal.all():
        raise ValueError("Seven-game independent scoring evidence changed")
    exposures = pd.concat(native_exposure, ignore_index=True).groupby(["game_id","player_id"],as_index=False).exposure.sum()
    rolling = required_game_exclusions(unknown, exposures, admission)
    entrant_root = VPM / "vpm/data/entrants"
    model_root = VPM / "vpm/results/harness/vpm_net_lane_clean_v1/entrant_models"
    lines, sources, index, models = load_inputs(source, entrant_root, model_root, ROOT.parent / "autocomplete_map.csv")
    ids = np.sort(np.unique(np.r_[lines.player_id.to_numpy(), index.old.index.to_numpy()]))
    annual_rows = []
    for season in range(2020, 2027):
        cut = snapshot_date(season)
        state = index.at(ids, cut).reset_index()
        _, model = legal_model(models, cut)
        u = state.loc[~state.background_weight_identified].copy()
        u["snapshot_date"] = cut
        u["evaluation_season"] = season
        u["background_model_available"] = model is not None
        annual_rows.append(u)
    current = index.at(ids, "2026-09-01").reset_index()
    current = current.loc[~current.background_weight_identified].copy()
    current["snapshot_date"] = pd.Timestamp("2026-09-01")
    current["evaluation_season"] = 0
    current["background_model_available"] = legal_model(models, "2026-09-01")[1] is not None
    annual = pd.concat(annual_rows, ignore_index=True)
    minutes_path = pin(VPM / "vpm/results/harness/vpm_field_benchmark_clean_sq_v1/actual_minutes.parquet")
    games_path = pin(VPM / "vpm/results/harness/vpm_net_lane_clean_v1/game_predictions.parquet")
    minutes, benchmark = pd.read_parquet(minutes_path), pd.read_parquet(games_path)
    minutes.game_id, benchmark.game_id = normal(minutes.game_id), normal(benchmark.game_id)
    # Exact existing selected-model benchmark census, with actual positive
    # player minutes. Coverage inspection never treats future minutes as input.
    minutes = minutes.loc[minutes.minutes.gt(0)].merge(benchmark[["game_id","season"]], on=["game_id","season"], validate="many_to_one")
    hit = minutes.merge(annual[["player_id","evaluation_season"]].rename(columns={"evaluation_season":"season"}),
                        on=["player_id","season"], validate="many_to_one")
    static = hit.groupby(["season","game_id"]).player_id.agg(lambda a:",".join(map(str,sorted(set(a))))).rename("unknown_player_ids").reset_index()
    static["reason"] = "unidentified_entrant_weight_at_frozen_annual_board_cut"
    for name, digest in pins.items():
        if sha(name) != digest:
            raise ValueError("A career audit parent changed during the run")
    out.mkdir(parents=True)
    tables = {"unsafe_source_games.csv":evidence, "clean_definition_controls.csv":pd.DataFrame(controls),
        "rolling_game_exclusions.csv":rolling, "annual_unknown_players.csv":annual,
        "current_unknown_players.csv":current, "annual_benchmark_game_exclusions.csv":static}
    for name, table in tables.items():
        table.to_csv(out/name,index=False)
    summary = dict(status="completed_unresolved_career_exposure_gate", unsafe_source_games=7, unsafe_source_rows=145,
        recovered_exposure_rows=0, native_scoring_matches_all_seven=True,
        rolling_unknown_queries=len(unknown), rolling_with_available_background_model=int(unknown.background_model_available.sum()),
        rolling_required_native_games=len(rolling), rolling_additional_common_exclusions=int(rolling.additional_common_exclusion.sum()),
        annual_unknown_player_counts={str(s):int(len(g)) for s,g in annual.groupby("evaluation_season")},
        current_unknown_players=current.player_id.astype(int).tolist(),
        benchmark_reference_games=len(benchmark), annual_benchmark_excluded_games=len(static),
        annual_benchmark_excluded_by_season={str(s):int(len(g)) for s,g in static.groupby("season")},
        career_definition="selected completed1997-2013 exposure plus validated2014+ sum((off_poss+def_poss)/2), strictly before query date",
        policy="do not splice native target-row exposure into a different source possession definition; retain unknown; whole-game common exclusions only where an unidentified entrant weight is required",
        selected_vpm_changed=False, source_v2_changed=False, protected_2027_outcomes=0, model_fitted=False,
        inputs=pins, artifacts={name:sha(out/name) for name in tables})
    write_json(out/"receipt.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ["inputs","artifacts"]}),flush=True)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--entrants",type=Path,default=ENTRANTS)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    run(args.source,args.entrants,args.out)
