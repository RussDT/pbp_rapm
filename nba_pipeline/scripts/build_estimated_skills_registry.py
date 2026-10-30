"""Build the NBA predictive box-score stats registry snapshot.

The curated source is ``nba_pipeline/config/estimated_skills_registry.json``: one
entry per predictive box-score stat across the twenty estimated-skills lanes,
with the model on the table, the dev-selected model, the export that carries the
current per-player value, the receipts that hold the evidence, and plain-English
state notes. This script

1. validates the registry (unique ids, statuses in the vocabulary, every lane
   1-20 present, every ``live`` stat with an export, no provider columns);
2. pulls the headline evidence for each stat from its receipts through small
   lane-specific adapters that tolerate missing files (a missing receipt is
   reported, never invented);
3. joins the current per-player value of every ``live`` stat on the 582-player
   current-PBP universe (missing values stay NaN, never zero);
4. writes the snapshot under ``results/estimated_skills/registry/``: the
   registry JSON with resolved evidence, a status table (CSV + Markdown), the
   values table (wide + long), an evidence table, a manifest with input hashes,
   and a self-contained ``index.html`` reader.

Nothing here scores 2026; the evidence comes from development and frozen
stress receipts only. Provider ratings (EPM, DARKO, DPM, eWINS) are never read.
Generated files stay out of git.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "nba_pipeline/config/estimated_skills_registry.json"
RESULTS = ROOT / "nba_pipeline/results/estimated_skills"
AUTOCOMPLETE = ROOT / "autocomplete_map.csv"
PROVIDER_TOKENS = ("epm", "darko", "dpm", "ewins", "dunks")
LANE_RANGE = range(1, 21)
PBP_SKILLS = ["rim_pct", "mid_pct", "three_pct", "ft_pct", "rim_fga_rate", "mid_fga_rate", "three_fga_rate",
              "fta_rate", "pts_rate", "ast_rate", "tov_rate", "oreb_rate", "usg_pct", "dreb_rate", "stl_rate",
              "blk_rate", "pf_committed_rate"]


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------
def load_registry(path: Path = REGISTRY) -> dict:
    return json.loads(Path(path).read_text())


def validate_registry(reg: dict) -> list[str]:
    """Return a list of problems; an empty list means the registry is valid."""
    problems: list[str] = []
    if reg.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    for key in ("registry_id", "owner", "contract", "status_vocabulary", "families", "pbp_panel", "stats", "lanes"):
        if key not in reg:
            problems.append(f"missing top-level key {key}")
    if problems:
        return problems
    vocab = set(reg["status_vocabulary"])
    families = reg["families"]
    stats = reg["stats"]
    ids = [s.get("stat_id") for s in stats]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate stat_id: {dupes}")
    lanes = {l.get("lane"): l for l in reg["lanes"]}
    missing_lanes = [n for n in LANE_RANGE if n not in lanes]
    if missing_lanes:
        problems.append(f"lanes missing from inventory: {missing_lanes}")
    extra = sorted(n for n in lanes if n not in LANE_RANGE)
    if extra:
        problems.append(f"unexpected lane numbers: {extra}")
    for l in reg["lanes"]:
        if l.get("status") not in vocab:
            problems.append(f"lane {l.get('lane')} status {l.get('status')!r} not in vocabulary")
        if isinstance(l.get("stats"), list):
            unknown = [s for s in l["stats"] if s not in ids]
            if unknown:
                problems.append(f"lane {l.get('lane')} references unknown stats {unknown}")
    pbp_ids = set()
    for s in stats:
        sid = s.get("stat_id", "<missing>")
        for key in ("label", "family", "lane", "side", "status", "notes", "next_step", "evidence"):
            if key not in s:
                problems.append(f"{sid}: missing {key}")
        if s.get("status") not in vocab:
            problems.append(f"{sid}: status {s.get('status')!r} not in vocabulary")
        fam = families.get(s.get("family"))
        if fam is None:
            problems.append(f"{sid}: unknown family {s.get('family')!r}")
        elif fam.get("lane") != s.get("lane"):
            problems.append(f"{sid}: lane {s.get('lane')} disagrees with family lane {fam.get('lane')}")
        if s.get("lane") not in lanes:
            problems.append(f"{sid}: lane {s.get('lane')} not in inventory")
        ev = s.get("evidence") or {}
        if ev.get("kind") == "pbp_skill":
            pbp_ids.add(sid)
            if ev.get("skill") != sid or sid not in PBP_SKILLS:
                problems.append(f"{sid}: pbp_skill evidence must name the skill itself")
            if fam is not None and "pbp_value_pattern" not in fam:
                problems.append(f"{sid}: family {s['family']} lacks pbp_value_pattern")
        elif s.get("status") == "live":
            exp = s.get("export") or {}
            for key in ("path", "id_column", "value_column", "date_column"):
                if key not in exp:
                    problems.append(f"{sid}: live stat export lacks {key}")
            cols = [exp.get("value_column", ""), exp.get("date_column", "")] + list((exp.get("companion_columns") or {}).values())
            bad = [c for c in cols if any(tok in str(c).lower() for tok in PROVIDER_TOKENS)]
            if bad:
                problems.append(f"{sid}: export references provider columns {bad}")
        if s.get("status") == "live" and not s.get("table_model"):
            problems.append(f"{sid}: live stat needs table_model")
    if pbp_ids != set(PBP_SKILLS):
        problems.append(f"PBP panel must list exactly the 17 skills; got {sorted(pbp_ids)}")
    return problems


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def load_json_if_present(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def gain_block(d) -> dict | None:
    """Read a paired-comparison block from a receipt, tolerating the key names each lane used."""
    if not isinstance(d, dict):
        return None
    pct = None
    for k in ("gain_pct", "pct_gain", "gain_percent", "pct", "gain"):
        if k in d:
            pct = _num(d[k])
            break
    ci = None
    if isinstance(d.get("gain_ci95"), (list, tuple)) and len(d["gain_ci95"]) == 2:
        ci = [_num(d["gain_ci95"][0]), _num(d["gain_ci95"][1])]          # already in percent
    elif isinstance(d.get("ci95"), (list, tuple)) and len(d["ci95"]) == 2:
        lo, hi = _num(d["ci95"][0]), _num(d["ci95"][1])
        control = _num(d.get("control_loss"))
        if control and "delta" in d:                                    # shooting-lane receipts: interval on the loss delta
            lo, hi = (lo / control * 100 if lo is not None else None), (hi / control * 100 if hi is not None else None)
        ci = [lo, hi]
    elif "ci_lo" in d and "ci_hi" in d:
        ci = [_num(d["ci_lo"]), _num(d["ci_hi"])]
    wins = None
    for k in ("season_wins", "seasons_won", "wins", "cutoffs_won", "dates_won"):
        if k in d:
            v = d[k]
            wins = len(v) if isinstance(v, (list, tuple)) else _num(v)
            break
    of = None
    for k in ("seasons", "n_seasons", "cutoffs", "dates_total"):
        if k in d:
            v = d[k]
            of = len(v) if isinstance(v, (list, tuple)) else _num(v)
            break
    if pct is None and ci is None:
        return None
    return {"gain_pct": pct, "ci_lo": ci[0] if ci else None, "ci_hi": ci[1] if ci else None,
            "wins": int(wins) if wins is not None else None, "of": int(of) if of is not None else None}


def fmt_gain(g: dict | None, digits=3) -> str:
    if not g or g.get("gain_pct") is None:
        return "no value"
    s = f"{g['gain_pct']:+.{digits}f}%"
    if g.get("ci_lo") is not None and g.get("ci_hi") is not None:
        s += f" [{g['ci_lo']:+.{digits}f}, {g['ci_hi']:+.{digits}f}]"
    if g.get("wins") is not None and g.get("of"):
        s += f" {g['wins']}/{g['of']}"
    return s


def row(stat_id, comparison, window, g, source, note=None):
    out = {"stat_id": stat_id, "comparison": comparison, "window": window, "source": source}
    out.update(g or {"gain_pct": None, "ci_lo": None, "ci_hi": None, "wins": None, "of": None})
    out["display"] = fmt_gain(g)
    if note:
        out["note"] = note
    return out


# --------------------------------------------------------------------------------------
# evidence adapters (development + frozen stress only; never 2026)
# --------------------------------------------------------------------------------------
class Receipts:
    def __init__(self, results: Path):
        self.results = Path(results)
        self.used: dict[str, Path] = {}
        self.missing: list[str] = []
        self._cache: dict[str, object] = {}

    def get(self, rel: str):
        if rel in self._cache:
            return self._cache[rel]
        path = self.results / rel
        data = load_json_if_present(path)
        if data is None:
            if rel not in self.missing:
                self.missing.append(rel)
        else:
            self.used[rel] = path
        self._cache[rel] = data
        return data


def evidence_pbp_skill(stat: dict, reg: dict, rc: Receipts) -> list[dict]:
    skill = stat["evidence"]["skill"]
    panel = reg["pbp_panel"]
    rcp = panel["receipts"]
    rows = []
    ev = rc.get(rcp["eval_report"])
    if ev:
        dev = ((ev.get("per_skill") or {}).get(skill) or {}).get("dev") or {}
        for base, key in (("trail10", "pct_impr_vs_trail10"), ("season_to_date", "pct_impr_vs_season"), ("career", "pct_impr_vs_career")):
            if key in dev:
                rows.append(row(skill, f"v1 (table) vs {base}", "dev 2020-2025", {"gain_pct": _num(dev[key]), "ci_lo": None, "ci_hi": None, "wins": None, "of": None}, rcp["eval_report"]))
    v4 = rc.get(rcp["v4"])
    if v4 and skill in v4:
        blk = v4[skill]
        if "vs_v1_pct" in blk:
            rows.append(row(skill, "v4 (dev-selected) vs v1 (table)", "dev 2020-2025", {"gain_pct": _num(blk["vs_v1_pct"]), "ci_lo": None, "ci_hi": None, "wins": None, "of": None}, rcp["v4"]))
        vs = blk.get("vs") or {}
        if "A" in vs:
            rows.append(row(skill, "v4 vs plain OU (stage A)", "dev 2020-2025", gain_block(vs["A"]), rcp["v4"]))
        if "B" in vs:
            rows.append(row(skill, "v4 context vs OU + corrections (stage B)", "dev 2020-2025", gain_block(vs["B"]), rcp["v4"]))
    recipe = stat.get("lab_recipe")
    r3 = rc.get(rcp["lab_round3"])
    if r3 and recipe:
        variant = panel["lab_recipes"][recipe]["variant"]
        res = (r3.get("results") or {}).get(skill) or {}
        blk = res.get(variant)
        if isinstance(blk, dict):
            rows.append(row(skill, f"lab recipe {recipe} ({variant}) vs re-tuned v4", "dev 2020-2025", gain_block(blk.get("vs_base")), rcp["lab_round3"]))
    if stat["evidence"].get("extra") == "volume_prior":
        rows.extend(evidence_volume_prior(stat, reg, rc))
    return rows


def evidence_volume_prior(stat, reg, rc):
    rep = rc.get("three_pct_volume_prior/report.json")
    if not rep:
        return []
    rows = []
    stress = rep.get("stress") or {}
    for key, label in (("all", "all shooters"), ("low_volume_under_1", "under 1 3PA/100"), ("under_3", "under 3 3PA/100")):
        if key in stress:
            rows.append(row(stat["stat_id"], f"volume prior (strength {rep.get('selected_strength')}) vs v1, {label}", "frozen stress 2023-2025", gain_block(stress[key]), "three_pct_volume_prior/report.json"))
    return rows


def evidence_deflections_next_game(stat, reg, rc):
    rows = []
    frozen = rc.get(stat["receipts"]["frozen"])
    if frozen and isinstance(frozen.get("engine_selection"), dict):
        rows.append(row(stat["stat_id"], "OU vs re-tuned random walk (selection)", "dev 2020-2022", gain_block(frozen["engine_selection"]), stat["receipts"]["frozen"]))
    stress = rc.get(stat["receipts"]["stress"])
    if stress:
        sel = stress.get("selected")
        for m in stress.get("metrics") or []:
            if not isinstance(m, dict) or (sel and m.get("candidate") != sel):
                continue
            rows.append(row(stat["stat_id"], f"{m.get('candidate')} vs {m.get('baseline')}", "frozen stress 2023-2025", gain_block(m), stat["receipts"]["stress"]))
    return rows


def evidence_tng_summary(stat, reg, rc):
    fam = stat["evidence"]["family"]
    summary = rc.get(stat["receipts"]["summary"])
    rows = []
    if not summary:
        return rows
    entries = summary if isinstance(summary, list) else summary.get("stats") or summary.get("results") or []
    entry = next((e for e in entries if isinstance(e, dict) and e.get("stat") == stat["stat_id"]), None)
    if entry is None:
        return rows
    dev_window = "dev 2021-2022" if fam == "fouls_drawn" else "dev 2020-2022"
    if isinstance(entry.get("dev_ou_vs_rw"), dict):
        rows.append(row(stat["stat_id"], "OU vs re-tuned random walk (selection)", dev_window, gain_block(entry["dev_ou_vs_rw"]), stat["receipts"]["summary"]))
    for base, blk in (entry.get("vs") or {}).items():
        rows.append(row(stat["stat_id"], f"{entry.get('selected', 'selected')} vs {base}", "frozen stress 2023-2025", gain_block(blk), stat["receipts"]["summary"]))
    return rows


def evidence_rim_saved(stat, reg, rc):
    rows = []
    stress = rc.get(stat["receipts"]["stress"])
    if stress:
        for key, label in (("saved", "decay model vs prior-mean baseline, points saved"), ("allowed_points", "decay model vs baseline, allowed points")):
            if key in stress:
                rows.append(row(stat["stat_id"], label, "frozen stress 2024-2025", gain_block(stress[key]), stat["receipts"]["stress"]))
    blocks = rc.get(stat["receipts"]["blocks_control"])
    if blocks:
        for key, label in (("rim_vs_blocks", "rim saved vs blocks alone"), ("joint_vs_blocks", "blocks + rim saved vs blocks alone")):
            if key in blocks:
                rows.append(row(stat["stat_id"], label, "frozen stress 2024-2025", gain_block(blocks[key]), stat["receipts"]["blocks_control"]))
    return rows


def evidence_lane(lane: dict, rc: Receipts) -> list[dict]:
    """Evidence rows for non-stat lanes (failed / increment experiments)."""
    kind = (lane.get("evidence") or {}).get("kind")
    rcp = lane.get("receipts") or {}
    lid = f"lane_{lane['lane']}"
    rows = []
    if kind == "playoff_context":
        rep = rc.get(rcp["report"])
        if rep:
            for key, win in (("development", "dev 2020-2022"), ("stress", "frozen stress 2023-2025")):
                blk = (rep.get(key) or {}).get("all")
                rows.append(row(lid, "additive playoff offset vs no offset", win, gain_block(blk), rcp["report"]))
    elif kind == "steals_increment":
        rep = rc.get(rcp["report"])
        if rep:
            rows.append(row(lid, "steals + pregame deflections vs steals alone", "dev 2020-2022", gain_block(rep.get("development")), rcp["report"]))
            rows.append(row(lid, "steals + pregame deflections vs calibrated steals", "frozen stress 2023-2025", gain_block(rep.get("stress_vs_calibrated")), rcp["report"]))
    elif kind == "rim_blocked":
        rep = rc.get(rcp["report"])
        if rep:
            dev = rep.get("development") or {}
            if isinstance(dev.get("pct_mse"), dict):
                rows.append(row(lid, "rim FG% + blocked prior vs rim FG% alone (pct MSE)", "dev 2020-2022", gain_block(dev["pct_mse"]), rcp["report"],
                                note=f"passes_development={rep.get('passes_development')}"))
            blk = (rep.get("stress") or {}).get("all") or {}
            if isinstance(blk.get("pct_mse"), dict):
                rows.append(row(lid, "rim FG% + blocked prior vs rim FG% alone (pct MSE)", "frozen stress 2023-2025", gain_block(blk["pct_mse"]), rcp["report"]))
    elif kind == "fusion":
        rep = rc.get(rcp["report"])
        if rep:
            dg = rep.get("development_gate")
            if isinstance(dg, dict):
                rows.append(row(lid, f"{dg.get('candidate')} vs {dg.get('control')}, {dg.get('metric')} (development gate)", "dev 2020-2022", gain_block(dg), rcp["report"]))
            jg = rep.get("joint_gate")
            if isinstance(jg, dict):
                rows.append(row(lid, f"{jg.get('candidate')} vs {jg.get('control')}, {jg.get('metric')} (joint gate)", "dev 2020-2022", gain_block(jg), rcp["report"]))
            sk = rep.get("stress_vs_kalman") or {}
            for metric in ("pct_mse", "makes_mse"):
                if isinstance(sk.get(metric), dict):
                    rows.append(row(lid, f"selected {sk[metric].get('candidate')} vs v1 Kalman, {metric}", "frozen stress 2023-2025", gain_block(sk[metric]), rcp["report"]))
    elif kind == "seasonality":
        rep = rc.get(rcp["report"])
        if rep:
            dev = rep.get("development") or {}
            for strength, blk in dev.items():
                if isinstance(blk, dict) and str(strength) != "0.0":
                    rows.append(row(lid, f"game-number cubic strength {strength} vs none ({blk.get('metric')})", "dev 2020-2022", gain_block(blk), rcp["report"]))
            rows.append(row(lid, "selected strength", "dev 2020-2022", None, rcp["report"],
                            note=f"selected {rep.get('selected')}; passes_development={rep.get('passes_development')}"))
    if lane["lane"] == 3 and "stress" in rcp:
        rep = rc.get(rcp["stress"])
        if rep:
            for m in rep.get("metrics") or []:
                if isinstance(m, dict) and m.get("baseline") == "random_walk":
                    rows.append(row(lid, "forward-horizon OU vs random walk (rest-of-season objective)", "frozen stress", gain_block(m), rcp["stress"]))
    return rows


ADAPTERS = {"pbp_skill": evidence_pbp_skill, "deflections_next_game": evidence_deflections_next_game,
            "tng_summary": evidence_tng_summary, "rim_saved": evidence_rim_saved}


def collect_evidence(reg: dict, rc: Receipts) -> tuple[list[dict], dict[str, str]]:
    rows, errors = [], {}
    for stat in reg["stats"]:
        kind = stat["evidence"]["kind"]
        fn = ADAPTERS.get(kind)
        if fn is None:
            errors[stat["stat_id"]] = f"no adapter for evidence kind {kind}"
            continue
        try:
            rows.extend(fn(stat, reg, rc))
        except Exception as e:  # a broken receipt must not stop the snapshot
            errors[stat["stat_id"]] = f"{type(e).__name__}: {e}"
    for lane in reg["lanes"]:
        try:
            rows.extend(evidence_lane(lane, rc))
        except Exception as e:
            errors[f"lane_{lane['lane']}"] = f"{type(e).__name__}: {e}"
    return rows, errors


# --------------------------------------------------------------------------------------
# current values
# --------------------------------------------------------------------------------------
def _read_csv(path: Path, **kw) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, **kw)


def load_names(autocomplete: Path = AUTOCOMPLETE) -> pd.Series:
    am = _read_csv(autocomplete)
    if am is None:
        return pd.Series(dtype=object)
    am = am.dropna(subset=["nba_id"]).drop_duplicates("nba_id")
    return pd.Series(am.player_name.values, index=am.nba_id.astype(int).values)


def build_values(reg: dict, results: Path, names: pd.Series, rc: Receipts):
    """Wide and long current-value tables on the current-PBP universe. Missing stays NaN."""
    panel = reg["pbp_panel"]
    prov_path = results / panel["export"]["path"]
    prov = _read_csv(prov_path)
    if prov is None:
        raise FileNotFoundError(f"PBP export missing: {prov_path}")
    rc.used[panel["export"]["path"]] = prov_path
    idc = panel["export"]["id_column"]
    prov[idc] = prov[idc].astype(int)
    universe = prov[[idc]].drop_duplicates().rename(columns={idc: "player_id"})
    universe["player_name"] = universe.player_id.map(names)
    fallback = prov.set_index(idc)
    for col in ("player_name", "Player"):
        if col in fallback:
            universe["player_name"] = universe.player_name.fillna(universe.player_id.map(fallback[col]))
    wide = universe.copy()
    long_rows = []
    coverage = {}
    families = reg["families"]
    for stat in reg["stats"]:
        if stat["status"] != "live":
            continue
        sid = stat["stat_id"]
        fam = families[stat["family"]]
        ev = stat["evidence"]
        if ev["kind"] == "pbp_skill":
            col = fam["pbp_value_pattern"].format(skill=sid)
            date_col = panel["export"]["date_pattern"].format(skill=sid)
            obs_col = panel["export"]["obs_pattern"].format(skill=sid)
            override = stat.get("export_override") or {}
            col = override.get("value_column", col)
            if col not in prov:
                coverage[sid] = {"players": 0, "error": f"column {col} missing"}
                continue
            vals = prov.set_index(idc)[col] * fam["display_scale"]
            dates = prov.set_index(idc)[date_col] if date_col in prov else pd.Series(index=prov[idc], dtype=object)
            obs = prov.set_index(idc)[obs_col] if obs_col in prov else None
            wide[sid] = wide.player_id.map(vals)
            wide[f"{sid}_date"] = wide.player_id.map(dates)
            for cname, ccol in (override.get("companion_columns") or {}).items():
                if ccol in prov:
                    wide[cname] = wide.player_id.map(prov.set_index(idc)[ccol] * fam["display_scale"])
            src = panel["export"]["path"]
            if obs is not None:
                wide[f"{sid}_n_obs"] = wide.player_id.map(obs)
        else:
            exp = stat["export"]
            path = results / exp["path"]
            df = _read_csv(path)
            if df is None:
                coverage[sid] = {"players": 0, "error": f"export missing {exp['path']}"}
                continue
            rc.used[exp["path"]] = path
            eid = exp["id_column"]
            df = df.dropna(subset=[eid]).copy()
            df[eid] = df[eid].astype(int)
            if df[eid].duplicated().any():
                df = df.sort_values(exp["date_column"]).drop_duplicates(eid, keep="last")
            df = df.set_index(eid)
            wide[sid] = wide.player_id.map(df[exp["value_column"]] * fam["display_scale"])
            wide[f"{sid}_date"] = wide.player_id.map(df[exp["date_column"]])
            for cname, ccol in (exp.get("companion_columns") or {}).items():
                if ccol in df:
                    wide[cname] = wide.player_id.map(df[ccol])
            src = exp["path"]
        have = wide[sid].notna()
        coverage[sid] = {"players": int(have.sum()), "universe": int(len(wide)),
                         "state_date_min": str(wide.loc[have, f"{sid}_date"].min()) if have.any() else None,
                         "state_date_max": str(wide.loc[have, f"{sid}_date"].max()) if have.any() else None,
                         "value_mean": float(wide.loc[have, sid].mean()) if have.any() else None,
                         "source": src}
        sub = wide.loc[have, ["player_id", "player_name", sid, f"{sid}_date"]]
        for r in sub.itertuples(index=False):
            long_rows.append({"player_id": int(r[0]), "player_name": r[1], "stat_id": sid, "value": float(r[2]),
                              "state_date": r[3], "units": fam["display_units"], "status": stat["status"], "source": src})
    long = pd.DataFrame(long_rows, columns=["player_id", "player_name", "stat_id", "value", "state_date", "units", "status", "source"])
    return wide, long, coverage


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------
def headline(stat: dict, ev_rows: list[dict]) -> str:
    mine = [r for r in ev_rows if r["stat_id"] == stat["stat_id"]]
    if not mine:
        return "no receipt evidence found"
    pick = []
    kind = stat["evidence"]["kind"]
    if kind == "pbp_skill":
        for r in mine:
            if r["comparison"].startswith("v1 (table) vs trail10") or r["comparison"].startswith("v4 (dev-selected) vs v1") or r["comparison"].startswith("lab recipe"):
                pick.append(r)
    elif kind in ("deflections_next_game", "tng_summary"):
        for r in mine:
            if "random walk" in r["comparison"] or "random_walk" in r["comparison"] or "trail10" in r["comparison"]:
                pick.append(r)
    elif kind == "rim_saved":
        pick = mine[:1] + [r for r in mine if "vs blocks alone" in r["comparison"]]
    else:
        pick = mine[:2]
    return "; ".join(f"{r['comparison']} {r['display']} ({r['window']})" for r in pick)


def write_csv(df: pd.DataFrame, path: Path):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    out.to_csv(path, index=False)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df[cols].itertuples(index=False):
        cells = []
        for v in r:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("")
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_status(reg: dict, ev_rows: list[dict], coverage: dict) -> pd.DataFrame:
    rows = []
    for s in reg["stats"]:
        fam = reg["families"][s["family"]]
        cov = coverage.get(s["stat_id"], {})
        rows.append({
            "stat_id": s["stat_id"], "label": s["label"], "lane": s["lane"], "family": fam["label"], "side": s["side"],
            "units": fam["display_units"], "status": s["status"], "table_model": s.get("table_model"),
            "dev_selected": (f"v4 + lab recipe {s['lab_recipe']}" if s.get("lab_recipe") else ("v4 (no lab gain)" if s["evidence"]["kind"] == "pbp_skill" else "same as table (frozen)")),
            "players_with_value": cov.get("players"), "state_date_latest": cov.get("state_date_max"),
            "state_date_earliest": cov.get("state_date_min"), "headline_evidence": headline(s, ev_rows),
            "notes": s["notes"], "next_step": s["next_step"], "doc": s.get("doc") or reg["pbp_panel"].get("doc") or reg["doc"],
            "export": (s.get("export") or {}).get("path") or reg["pbp_panel"]["export"]["path"],
            "export_error": cov.get("error")})
    return pd.DataFrame(rows)


def build_lanes_table(reg: dict, ev_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for l in reg["lanes"]:
        mine = [r for r in ev_rows if r["stat_id"] == f"lane_{l['lane']}"]
        rows.append({"lane": l["lane"], "name": l["name"], "status": l["status"],
                     "stats": ", ".join(l["stats"]) if isinstance(l["stats"], list) else l["stats"],
                     "state": l["state"], "evidence": "; ".join(f"{r['comparison']} {r['display']} ({r['window']})" + (f" [{r['note']}]" if r.get("note") else "") for r in mine),
                     "doc": l.get("doc"), "scripts": ", ".join(l.get("scripts") or [])})
    return pd.DataFrame(rows)


def render_html(reg: dict, status: pd.DataFrame, lanes: pd.DataFrame, wide: pd.DataFrame, ev: pd.DataFrame, manifest: dict) -> str:
    def esc(v):
        return html.escape("" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    live = [s for s in reg["stats"] if s["status"] == "live"]
    value_cols = [s["stat_id"] for s in live if s["stat_id"] in wide]
    labels = {s["stat_id"]: s["label"] for s in reg["stats"]}
    table = wide[["player_id", "player_name"] + value_cols + [f"{c}_date" for c in value_cols if f"{c}_date" in wide]]
    records = json.loads(table.to_json(orient="records", date_format="iso"))
    status_json = json.loads(status.to_json(orient="records"))
    ev_json = json.loads(ev.to_json(orient="records"))
    lanes_json = json.loads(lanes.to_json(orient="records"))
    contract = reg["contract"]
    vocab = reg["status_vocabulary"]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{esc(reg['title'])}</title>",
        "<style>body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:16px 20px;color:#1a1a1a;background:#fbfaf7;font-size:14px}",
        "h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #999;padding-bottom:4px}",
        ".muted{color:#555}.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;font-weight:600;margin-right:4px}",
        ".live{background:#d9f2dd;color:#14532d}.development{background:#fde9c8;color:#7c3d00}.increment{background:#dbeafe;color:#1e3a8a}",
        ".measurement,.imputation,.diagnostic{background:#e8e8e8;color:#333}.failed{background:#fbd5d5;color:#7f1d1d}",
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}",
        "th{background:#f0ede6;position:sticky;top:0;cursor:pointer;white-space:nowrap}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}",
        ".wrap{overflow-x:auto;max-height:70vh;border:1px solid #ccc}input{font-size:14px;padding:4px 8px;width:260px}",
        ".notes{max-width:520px}.small{font-size:12px}dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px}dt{font-weight:600}",
        "</style></head><body>",
        f"<h1>{esc(reg['title'])}</h1>",
        f"<div class='muted'>registry <code>{esc(reg['registry_id'])}</code> · as of {esc(reg['as_of'])} · built {esc(manifest['built_utc'])} · owner {esc(reg['owner'])} · {len(reg['stats'])} stats across 20 lanes · {sum(1 for s in live)} live on the values table</div>",
        "<h2>Contract</h2><dl>",
    ]
    for k, v in contract.items():
        parts.append(f"<dt>{esc(k)}</dt><dd>{esc(json.dumps(v) if isinstance(v, dict) else v)}</dd>")
    parts.append("</dl><h2>Status vocabulary</h2><dl>")
    for k, v in vocab.items():
        parts.append(f"<dt><span class='pill {esc(k)}'>{esc(k)}</span></dt><dd>{esc(v)}</dd>")
    parts.append("</dl><h2>Stat status</h2><div class='wrap'><table id='status'><thead><tr>")
    for c in ("stat_id", "label", "lane", "side", "status", "table_model", "dev_selected", "players_with_value", "state_date_latest", "headline_evidence", "notes", "next_step"):
        parts.append(f"<th>{esc(c)}</th>")
    parts.append("</tr></thead><tbody>")
    for r in status_json:
        parts.append("<tr>")
        for c in ("stat_id", "label", "lane", "side", "status", "table_model", "dev_selected", "players_with_value", "state_date_latest", "headline_evidence", "notes", "next_step"):
            v = r.get(c)
            if c == "status":
                parts.append(f"<td><span class='pill {esc(v)}'>{esc(v)}</span></td>")
            elif c in ("notes", "next_step", "headline_evidence"):
                parts.append(f"<td class='notes small'>{esc(v)}</td>")
            else:
                parts.append(f"<td>{esc(v)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    parts.append("<h2>Lane inventory</h2><div class='wrap'><table><thead><tr><th>lane</th><th>name</th><th>status</th><th>stats</th><th>state</th><th>evidence</th></tr></thead><tbody>")
    for r in lanes_json:
        parts.append(f"<tr><td>{esc(r['lane'])}</td><td>{esc(r['name'])}</td><td><span class='pill {esc(r['status'])}'>{esc(r['status'])}</span></td><td class='small'>{esc(r['stats'])}</td><td class='notes small'>{esc(r['state'])}</td><td class='notes small'>{esc(r['evidence'])}</td></tr>")
    parts.append("</tbody></table></div>")
    parts.append("<h2>Current values</h2><p class='muted small'>One row per player on the current-PBP universe. Percentages are in %, everything else per 100 possessions of the stat's exposure. Blank means no measured state for that player, never zero. Click a header to sort; type to filter by name.</p>")
    parts.append("<input id='q' placeholder='filter players'> <span id='count' class='muted small'></span>")
    parts.append("<div class='wrap'><table id='values'><thead><tr><th>player</th>")
    for c in value_cols:
        parts.append(f"<th class='num' title='{esc(c)}'>{esc(labels[c])}</th>")
    parts.append("</tr></thead><tbody id='vbody'></tbody></table></div>")
    parts.append("<h2>Evidence rows</h2><p class='muted small'>Every paired comparison pulled from receipts. Percent gain in next-game exposure-weighted MSE versus the named baseline, game-clustered 95% interval, and paired season wins. Development and frozen-stress windows only; nothing here scores 2026.</p>")
    parts.append("<div class='wrap'><table><thead><tr><th>stat</th><th>comparison</th><th>window</th><th>result</th><th>source</th></tr></thead><tbody>")
    for r in ev_json:
        parts.append(f"<tr><td>{esc(r['stat_id'])}</td><td>{esc(r['comparison'])}</td><td>{esc(r['window'])}</td><td>{esc(r['display'])}{(' · ' + esc(r['note'])) if r.get('note') else ''}</td><td class='small'>{esc(r['source'])}</td></tr>")
    parts.append("</tbody></table></div>")
    parts.append("<script>const DATA=" + json.dumps(records) + ";const COLS=" + json.dumps(value_cols) + ";")
    parts.append("""
let sortKey='player_name',sortDir=1;
function fmt(v){if(v===null||v===undefined)return'';return (Math.round(v*100)/100).toFixed(2);}
function render(){const q=(document.getElementById('q').value||'').toLowerCase();
 let rows=DATA.filter(r=>!q||String(r.player_name||'').toLowerCase().includes(q));
 rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];if(x===null||x===undefined)return 1;if(y===null||y===undefined)return -1;if(typeof x==='string')return sortDir*x.localeCompare(y);return sortDir*(x-y);});
 const tb=document.getElementById('vbody');tb.innerHTML=rows.map(r=>'<tr><td>'+(r.player_name||r.player_id)+'</td>'+COLS.map(c=>'<td class="num" title="'+(r[c+'_date']||'')+'">'+fmt(r[c])+'</td>').join('')+'</tr>').join('');
 document.getElementById('count').textContent=rows.length+' players';}
document.getElementById('q').addEventListener('input',render);
document.querySelectorAll('#values th').forEach((th,i)=>th.addEventListener('click',()=>{const k=i===0?'player_name':COLS[i-1];if(sortKey===k)sortDir=-sortDir;else{sortKey=k;sortDir=i===0?1:-1;}render();}));
document.querySelectorAll('#status th').forEach((th,i)=>th.addEventListener('click',()=>{const t=th.closest('table');const rows=[...t.tBodies[0].rows];const dir=th.dataset.dir==='1'?-1:1;th.dataset.dir=dir;rows.sort((a,b)=>{const x=a.cells[i].innerText,y=b.cells[i].innerText;const nx=parseFloat(x),ny=parseFloat(y);if(!isNaN(nx)&&!isNaN(ny))return dir*(nx-ny);return dir*x.localeCompare(y);});rows.forEach(r=>t.tBodies[0].appendChild(r));}));
render();
</script></body></html>""")
    return "".join(parts)


def build(registry_path: Path, results: Path, output_dir: Path, autocomplete: Path = AUTOCOMPLETE) -> dict:
    reg = load_registry(registry_path)
    problems = validate_registry(reg)
    if problems:
        raise SystemExit("registry invalid:\n  " + "\n  ".join(problems))
    rc = Receipts(results)
    ev_rows, errors = collect_evidence(reg, rc)
    names = load_names(autocomplete)
    wide, long, coverage = build_values(reg, results, names, rc)
    if (wide.select_dtypes("number").drop(columns=["player_id"]).fillna(1) == 0).all().any():
        raise SystemExit("a value column is entirely zero; refusing to write a zero-filled table")
    status = build_status(reg, ev_rows, coverage)
    lanes = build_lanes_table(reg, ev_rows)
    ev = pd.DataFrame(ev_rows, columns=["stat_id", "comparison", "window", "gain_pct", "ci_lo", "ci_hi", "wins", "of", "display", "source", "note"])
    output_dir.mkdir(parents=True, exist_ok=True)
    built = datetime.now(timezone.utc).isoformat(timespec="seconds")
    inputs = {rel: {"path": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p), "sha256": sha256_of(p),
                    "modified": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")}
              for rel, p in sorted(rc.used.items())}
    manifest = {"registry_id": reg["registry_id"], "built_utc": built, "registry_path": str(registry_path.relative_to(ROOT)) if str(registry_path).startswith(str(ROOT)) else str(registry_path),
                "registry_sha256": sha256_of(registry_path), "results_root": str(results),
                "n_stats": len(reg["stats"]), "n_live": int((status.status == "live").sum()), "n_lanes": len(reg["lanes"]),
                "universe_players": int(len(wide)), "coverage": coverage, "evidence_rows": len(ev_rows),
                "missing_receipts": rc.missing, "adapter_errors": errors, "inputs": inputs,
                "holdout_note": "No 2026 next-game scores appear in this snapshot. The v1 eval report and the NBA_ESTIMATED_SKILLS.md results table hold the one-shot v1 naive-baseline holdout; nothing here reads them."}
    snapshot = dict(reg)
    snapshot["built_utc"] = built
    snapshot["evidence"] = ev_rows
    snapshot["coverage"] = coverage
    (output_dir / "estimated_skills_registry.json").write_text(json.dumps(snapshot, indent=1))
    write_csv(status, output_dir / "registry_status.csv")
    write_csv(lanes, output_dir / "registry_lanes.csv")
    write_csv(ev, output_dir / "registry_evidence.csv")
    write_csv(wide, output_dir / "current_values.csv")
    write_csv(long, output_dir / "current_values_long.csv")
    md = [f"# {reg['title']} — status snapshot", "", f"Built {built} from `{manifest['registry_path']}`. Generated file; the curated registry and `{reg['doc']}` are the checked-in sources.", "",
          "## Stats", "", md_table(status, ["stat_id", "label", "lane", "side", "status", "table_model", "dev_selected", "players_with_value", "state_date_latest", "headline_evidence"]), "",
          "## Notes", ""]
    for r in status.itertuples(index=False):
        md.append(f"- **{r.stat_id}** ({r.status}): {r.notes} Next: {r.next_step}")
    md += ["", "## Lanes", "", md_table(lanes, ["lane", "name", "status", "stats", "state", "evidence"]), ""]
    (output_dir / "registry_status.md").write_text("\n".join(md))
    (output_dir / "index.html").write_text(render_html(reg, status, lanes, wide, ev, manifest))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--results", type=Path, default=RESULTS, help="estimated_skills results root")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "registry")
    parser.add_argument("--autocomplete", type=Path, default=AUTOCOMPLETE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        problems = validate_registry(load_registry(args.registry))
        if problems:
            print("\n".join(problems)); return 1
        print("registry valid"); return 0
    m = build(args.registry, args.results, args.output_dir, args.autocomplete)
    live = [k for k, v in m["coverage"].items() if v.get("players")]
    print(f"{m['registry_id']}: {m['n_stats']} stats ({m['n_live']} live), {m['n_lanes']} lanes, {m['universe_players']} players, "
          f"{m['evidence_rows']} evidence rows, {len(live)} value columns; missing receipts {len(m['missing_receipts'])}, adapter errors {len(m['adapter_errors'])} -> {args.output_dir}")
    for k, v in m["adapter_errors"].items():
        print(f"  adapter error {k}: {v}")
    for k in m["missing_receipts"]:
        print(f"  missing receipt {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
