#!/usr/bin/env python3
"""Build the self-contained WNBA VPM "rack" reader from the published board.

The rack is a single dark-theme HTML page (leaderboard, player knobs, four
factor bars, eFG zone EQ, prior -> lineup signal chain, team mixer) that embeds
the current-season WNBA VPM box-prior board as JSON. It reads the drop-in
databallr publish CSV written by ``prepare_wnba_vpm_publish.py`` and the daily
``latest_manifest.json``; it never recomputes anything, so every number on the
page is exactly a published field (or an exact CSV sum such as eFG value).

Usage (from the repository root, after ``prepare_wnba_vpm_publish.py``):

    .venv/bin/python wnba_test/build_wnba_vpm_rack.py

Outputs ``wnba_test/Results/current/publish/wnba_vpm_rack.html`` by default.
The template lives at ``wnba_test/wnba_vpm_rack_template.html`` and carries a
single ``__DATA__`` placeholder inside ``<script id="board-data">``.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "Results" / "current" / "publish" / "wnba_vpm_boxprior_current.csv"
DEFAULT_MANIFEST = ROOT / "Results" / "current" / "latest_manifest.json"
DEFAULT_TEMPLATE = ROOT / "wnba_vpm_rack_template.html"
DEFAULT_OUT = ROOT / "Results" / "current" / "publish" / "wnba_vpm_rack.html"
# databallr's brand face (Ballr Sans, OFL-derived, tabular numerals) lives in the sibling
# product checkout; the Arena / NBA RAPM themes embed it as data URIs when it is present.
DEFAULT_FONTS_DIR = ROOT.parents[2] / "products" / "databallr" / "public" / "fonts" / "ballr-sans"
FONT_FACES = (("BallrSans-Regular.ttf", 400), ("BallrSans-Medium.ttf", 500), ("BallrSans-Bold.ttf", 700))

# rack field -> publish CSV column
PLAYER_FIELDS = {
    "net": "net_rapm",
    "off": "off",
    "def": "def",
    "o_efg": "oALT_EFG_VALUE",
    "o_rim": "oALT_EFG_RIM",
    "o_mid": "oALT_EFG_MID",
    "o_three": "oALT_EFG_THREE",
    "o_ft": "oALT_FT",
    "o_tov": "oALT_TOV_VALUE",
    "o_sc": "oSC",
    "d_efg": "dALT_EFG_VALUE",
    "d_rim": "dALT_EFG_RIM",
    "d_mid": "dALT_EFG_MID",
    "d_three": "dALT_EFG_THREE",
    "d_ft": "dALT_FT",
    "d_tov": "dALT_TOV_VALUE",
    "d_sc": "dSC",
    "o_rim_f": "oALT_EFG_RIM_FREQ",
    "o_rim_g": "oALT_EFG_RIM_FG",
    "o_mid_f": "oALT_EFG_MID_FREQ",
    "o_mid_g": "oALT_EFG_MID_FG",
    "o_three_f": "oALT_EFG_THREE_FREQ",
    "o_three_g": "oALT_EFG_THREE_FG",
    "d_rim_f": "dALT_EFG_RIM_FREQ",
    "d_rim_g": "dALT_EFG_RIM_FG",
    "d_mid_f": "dALT_EFG_MID_FREQ",
    "d_mid_g": "dALT_EFG_MID_FG",
    "d_three_f": "dALT_EFG_THREE_FREQ",
    "d_three_g": "dALT_EFG_THREE_FG",
    "box_net": "box_net",
    "prior_net": "prior_net_rapm",
    "delta_net": "delta_net_rapm",
}
INT_FIELDS = {"poss": "possessions", "op": "off_poss", "dp": "def_poss"}


def _num(value: object) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite value in publish CSV: {value!r}")
    return round(out, 3)


def build_payload(csv_path: Path, manifest_path: Path, profile: dict | None = None) -> dict:
    df = pd.read_csv(csv_path)
    required = set(PLAYER_FIELDS.values()) | set(INT_FIELDS.values()) | {
        "player_id", "player_name", "team", "qualified", "box_prior_source", "as_of", "model_variant",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"publish CSV is missing columns: {missing}")
    if df["player_id"].duplicated().any():
        raise SystemExit("publish CSV must have exactly one row per player_id")

    # eFG must be the exact CSV value and the four factors must close to off/def.
    for side, total in (("o", "off"), ("d", "def")):
        gap = (
            df[f"{side}ALT_EFG_VALUE"] + df[f"{side}ALT_FT"] + df[f"{side}ALT_TOV_VALUE"] + df[f"{side}SC"] - df[total]
        ).abs().max()
        if gap > 0.0021:
            raise SystemExit(f"four-factor closure failed on {total}: max gap {gap:.4f}")

    manifest = json.load(open(manifest_path))
    if manifest.get("status") != "complete":
        raise SystemExit(f"latest_manifest.json status is {manifest.get('status')!r}, not 'complete'")
    as_of = str(df["as_of"].iloc[0])
    if manifest.get("as_of") != as_of:
        raise SystemExit(f"as_of mismatch: CSV {as_of} vs manifest {manifest.get('as_of')}")
    games = int(manifest["source"]["after"]["games"])
    as_of_long = dt.date.fromisoformat(as_of).strftime("%b %-d, %Y")

    players = []
    for _, r in df.sort_values("net_rapm", ascending=False).iterrows():
        p = {
            "id": int(r["player_id"]),
            "name": str(r["player_name"]),
            "team": str(r["team"]),
        }
        p.update({k: _num(r[c]) for k, c in PLAYER_FIELDS.items()})
        p.update({k: int(r[c]) for k, c in INT_FIELDS.items()})
        p["q"] = bool(int(r["qualified"]))
        p["box_src"] = str(r["box_prior_source"])
        players.append(p)

    payload = {
        "meta": {
            "as_of": as_of,
            "as_of_long": as_of_long,
            "games": games,
            "model_variant": str(df["model_variant"].iloc[0]),
        },
        "players": players,
    }
    if profile is not None:
        payload["profile"] = profile
    return payload


# ---------------------------------------------------------------- player profile
# The rack's profile modules (career log, shot profile, role rates) are built from the
# repo's own WNBA box cache -- the same pbpstats player-season totals that feed the box
# priors -- joined to the published VPM history. Nothing here is recomputed from PBP and
# nothing is imputed: a season a player did not play simply has no row.
BOX_DATA_DIR = ROOT.parents[1] / "wnba" / "data"
DEFAULT_HISTORY = ROOT / "Results" / "current" / "publish" / "wnba_vpm_boxprior_history_2022_2026.csv"
PROFILE_SEASONS = (2022, 2023, 2024, 2025, 2026)

# Row schema shared by the payload and the page. Keeping it as one ordered tuple lets the
# rows travel as bare arrays, which is what keeps the embedded JSON small.
ROW_KEYS = (
    "y", "st", "tm", "gp", "min",                       # identity / exposure
    "pts", "reb", "ast", "stl", "blk", "tov",           # box counting totals
    "op", "dp",                                          # on-floor possessions
    "fga", "fta", "ftm",                                 # shooting volume
    "rim_a", "rim_m", "mid_a", "mid_m", "tp_a", "tp_m",  # zone volume / makes
    "unast", "fgpts", "apts", "lb", "oreb", "dreb", "fd",  # role inputs
    "v_off", "v_def", "v_net", "v_poss",                 # published VPM impact
)


def _z(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, 0.0), errors="coerce").fillna(0.0)


def _season_frame(path: Path, year: int, split: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "id": pd.to_numeric(raw["EntityId"], errors="coerce").astype("int64"),
        "y": year,
        "st": split,
        "tm": raw["TeamAbbreviation"].astype(str),
        "gp": _z(raw, "GamesPlayed"),
        "min": _z(raw, "Minutes").clip(lower=0.0),
        "pts": _z(raw, "Points"),
        "reb": _z(raw, "Rebounds"),
        "ast": _z(raw, "Assists"),
        "stl": _z(raw, "Steals"),
        "blk": _z(raw, "Blocks"),
        "tov": _z(raw, "Turnovers"),
        "op": _z(raw, "OffPoss"),
        "dp": _z(raw, "DefPoss"),
        "fga": _z(raw, "FG2A") + _z(raw, "FG3A"),
        "fta": _z(raw, "FTA"),
        "ftm": _z(raw, "FtPoints"),
        "rim_a": _z(raw, "AtRimFGA"),
        "rim_m": _z(raw, "AtRimFGM"),
        "mid_a": _z(raw, "ShortMidRangeFGA") + _z(raw, "LongMidRangeFGA"),
        "mid_m": _z(raw, "ShortMidRangeFGM") + _z(raw, "LongMidRangeFGM"),
        "tp_a": _z(raw, "FG3A"),
        "tp_m": _z(raw, "FG3M"),
        "unast": _z(raw, "PtsUnassisted2s") + _z(raw, "PtsUnassisted3s"),
        "apts": _z(raw, "AssistPoints"),
        "lb": _z(raw, "LostBallTurnovers"),
        "oreb": _z(raw, "OffRebounds"),
        "dreb": _z(raw, "DefRebounds"),
        "fd": _z(raw, "FoulsDrawn"),
    })
    out["fgpts"] = out["pts"] - out["ftm"]
    return out[out["gp"] > 0].reset_index(drop=True)


def load_box_rows(box_dir: Path) -> pd.DataFrame:
    frames = []
    for year in PROFILE_SEASONS:
        for split, suffix in (("rs", ""), ("ps", "ps")):
            path = box_dir / f"{year}{suffix}_pbp.csv"
            if path.is_file():
                frames.append(_season_frame(path, year, split))
    if not frames:
        raise SystemExit(f"no WNBA box cache found under {box_dir}")
    return pd.concat(frames, ignore_index=True)


def league_baselines(rows: pd.DataFrame) -> dict:
    """Possession-weighted league rates per season/split.

    A player rate is per 100 of that player's own on-floor possessions, so pooling the
    same ratio over every player gives team volume per 100 divided by five -- the neutral
    per-slot baseline the shot-profile bars diverge from.
    """
    out = {}
    for (year, split), g in rows.groupby(["y", "st"]):
        op, dp = g["op"].sum(), g["dp"].sum()
        tsa = g["fga"].sum() + 0.44 * g["fta"].sum()
        def per100(col: str) -> float:
            return round(100.0 * g[col].sum() / op, 2) if op else 0.0
        def pct(num: str, den: str) -> float:
            d = g[den].sum()
            return round(100.0 * g[num].sum() / d, 2) if d else 0.0
        out[f"{year}|{split}"] = {
            "fta": per100("fta"), "ft": pct("ftm", "fta"),
            "rim_a": per100("rim_a"), "rim_g": pct("rim_m", "rim_a"),
            "mid_a": per100("mid_a"), "mid_g": pct("mid_m", "mid_a"),
            "tp_a": per100("tp_a"), "tp_g": pct("tp_m", "tp_a"),
            "ts": round(100.0 * g["pts"].sum() / (2.0 * tsa), 2) if tsa else 0.0,
            "usg": per100("fga") + round(0.44 * 100.0 * g["fta"].sum() / op, 2) + per100("tov") if op else 0.0,
            "rim_r": pct("rim_a", "fga"), "tp_r": pct("tp_a", "fga"),
            "self": pct("unast", "fgpts"), "apts": per100("apts"),
            "lb": per100("lb"), "oreb": per100("oreb"), "fd": per100("fd"),
            "stops": round(100.0 * (g["dreb"].sum() + g["stl"].sum() + g["blk"].sum()) / dp, 2) if dp else 0.0,
            "stl": round(100.0 * g["stl"].sum() / dp, 2) if dp else 0.0,
            "blk": round(100.0 * g["blk"].sum() / dp, 2) if dp else 0.0,
            "dreb": round(100.0 * g["dreb"].sum() / dp, 2) if dp else 0.0,
        }
    return out


def build_profile(box_dir: Path, history_path: Path, board_ids: set[int]) -> dict:
    rows = load_box_rows(box_dir)
    lg = league_baselines(rows)

    # Published VPM per season. 2022-2025 are ALL (RS+PS) solves and 2026 is RS, so the
    # impact column attaches to the row that solve actually covers rather than to both.
    vpm: dict[tuple[int, str], dict[int, tuple]] = {}
    if history_path.is_file():
        hist = pd.read_csv(history_path)
        for _, h in hist.iterrows():
            split = "ps" if str(h["season_type"]).lower() == "ps" else "rs"
            vpm.setdefault((int(h["season"]), split), {})[int(h["player_id"])] = (
                _num(h["off"]), _num(h["def"]), _num(h["net_rapm"]), int(h["possessions"]),
            )

    players: dict[str, list] = {}
    for pid, g in rows[rows["id"].isin(board_ids)].groupby("id"):
        packed = []
        for _, r in g.sort_values(["y", "st"], ascending=[False, True]).iterrows():
            v = vpm.get((int(r["y"]), str(r["st"])), {}).get(int(pid))
            rec = {k: r[k] for k in ROW_KEYS if k in r}
            rec["v_off"], rec["v_def"], rec["v_net"], rec["v_poss"] = v if v else (None, None, None, None)
            packed.append([
                rec[k] if isinstance(rec[k], (str, type(None))) else round(float(rec[k]), 3)
                for k in ROW_KEYS
            ])
        if packed:
            players[str(int(pid))] = packed
    return {"keys": list(ROW_KEYS), "lg": lg, "rows": players}


def font_face_css(fonts_dir: Path | None) -> str:
    """Return inline ``@font-face`` rules for Ballr Sans, or an empty string when the
    font files are unavailable (the themes then fall back to DM Sans / system UI)."""
    if fonts_dir is None:
        return ""
    files = [(fonts_dir / name, weight) for name, weight in FONT_FACES]
    missing = [str(path) for path, _ in files if not path.is_file()]
    if missing:
        print(f"note: Ballr Sans not embedded, missing {missing[0]}")
        return ""
    rules = []
    for path, weight in files:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        rules.append(
            '@font-face{font-family:"Ballr Sans";font-style:normal;font-weight:%d;font-display:swap;'
            'src:url("data:font/ttf;base64,%s") format("truetype")}' % (weight, b64)
        )
    return "\n  ".join(rules)


PROFILE_START = "<!-- profile:start"
PROFILE_END = "<!-- profile:end -->"


def strip_profile(template: str) -> str:
    """Remove the player-profile markup between the profile markers. The matching JS is
    guarded by ``if ($('profile'))`` so it becomes a no-op once the markup is gone."""
    start = template.find(PROFILE_START)
    end = template.find(PROFILE_END)
    if start < 0 or end < 0 or end < start:
        raise SystemExit("template is missing the profile:start / profile:end markers")
    return template[:start] + template[end + len(PROFILE_END):]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--also-write", type=Path, action="append", default=[], help="extra output copies")
    ap.add_argument(
        "--fonts-dir", type=Path, default=DEFAULT_FONTS_DIR,
        help="directory holding BallrSans-{Regular,Medium,Bold}.ttf to embed for the databallr themes",
    )
    ap.add_argument("--no-fonts", action="store_true", help="do not embed Ballr Sans (system-font fallback)")
    ap.add_argument("--box-dir", type=Path, default=BOX_DATA_DIR, help="WNBA pbpstats box cache (YYYY[ps]_pbp.csv)")
    ap.add_argument("--history", type=Path, default=DEFAULT_HISTORY, help="published multi-season VPM history CSV")
    ap.add_argument(
        "--no-profile", action="store_true",
        help="drop the player-profile section (career log, shot profile, role rates)",
    )
    ap.add_argument(
        "--emit-payload", type=Path, default=None,
        help="also write the embedded payload as a standalone JSON file (the databallr "
             "/wnba/vpm page fetches this instead of the self-contained HTML)",
    )
    ap.add_argument("--no-html", action="store_true", help="only write --emit-payload, skip the HTML page")
    args = ap.parse_args()

    payload = build_payload(args.csv, args.manifest)
    template = args.template.read_text(encoding="utf-8")
    if args.no_profile:
        template = strip_profile(template)
    else:
        board_ids = {p["id"] for p in payload["players"]}
        payload["profile"] = build_profile(args.box_dir, args.history, board_ids)
        covered = len(payload["profile"]["rows"])
        print(f"profile: {covered} of {len(board_ids)} board players have box rows")
    for placeholder in ("__DATA__", "__FONTS__"):
        if template.count(placeholder) != 1:
            raise SystemExit(f"template must contain exactly one {placeholder} placeholder")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    if args.emit_payload is not None:
        args.emit_payload.parent.mkdir(parents=True, exist_ok=True)
        # The standalone payload is fetched, not inlined in a <script> block, so it keeps
        # real "</" sequences; only the embedded copy needs the escape above.
        args.emit_payload.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        print(f"wrote {args.emit_payload} ({args.emit_payload.stat().st_size:,} bytes)")
    if args.no_html:
        return
    fonts = font_face_css(None if args.no_fonts else args.fonts_dir)
    html = template.replace("__DATA__", data).replace("__FONTS__", fonts)
    for out in [args.out, *args.also_write]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(payload['players'])} players, as of {payload['meta']['as_of']})")


if __name__ == "__main__":
    main()
