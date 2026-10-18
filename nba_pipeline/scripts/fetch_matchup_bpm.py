"""Cache public BRef advanced BPM for lagged opponent-strength research.

Use exact season/BRef-slug to NBA-id joins only. Current BPM 2.0 historical
values are retrospective definitions, not certified publication vintages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external/matchup_bpm_v1"
INDEX_URL = "https://raw.githubusercontent.com/gabriel1200/site_Data/refs/heads/master/index_master.csv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AdvancedParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.active = False
        self.row = None
        self.cell = None
        self.text = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.active = attrs.get("id") in ("advanced", "advanced_stats")
        elif self.active and tag == "tr":
            self.row = {}
        elif self.active and self.row is not None and tag in ("td", "th"):
            self.cell = attrs.get("data-stat")
            self.text = []
            if attrs.get("data-append-csv"):
                self.row["bref_id"] = attrs["data-append-csv"]
        elif self.active and self.row is not None and tag == "a" and self.cell in ("name_display", "player"):
            href = attrs.get("href", "")
            if href.startswith("/players/"):
                self.row["bref_id"] = href.rsplit("/", 1)[-1].removesuffix(".html")

    def handle_data(self, data):
        if self.active and self.cell:
            self.text.append(data)

    def handle_endtag(self, tag):
        if self.active and tag in ("td", "th") and self.cell:
            self.row[self.cell] = "".join(self.text).strip()
            self.cell = None
        elif self.active and tag == "tr" and self.row is not None:
            if self.row.get("bref_id") and self.row.get("bpm") not in (None, ""):
                self.rows.append(self.row)
            self.row = None
        elif tag == "table":
            self.active = False


def parse_advanced(html, year):
    if year not in range(2017, 2027):
        raise ValueError("Only registered2017–2026 source seasons")
    parser = AdvancedParser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError("No regular-season BRef advanced rows")
    raw = pd.DataFrame(parser.rows)
    raw["year"] = year
    raw["player_name"] = raw["name_display"] if "name_display" in raw else raw["player"]
    raw["team"] = raw["team_name_abbr"] if "team_name_abbr" in raw else raw["team_id"]
    raw["position"] = raw["pos"]
    for key in ["bpm", "obpm", "dbpm", "mp", "games"]:
        raw[key] = pd.to_numeric(raw[key].str.replace(",", "", regex=False), errors="raise")
    raw["aggregate"] = raw.team.isin(["TOT", "2TM", "3TM", "4TM", "5TM", "6TM"])
    selected = []
    for bref, group in raw.groupby("bref_id", sort=True):
        if len(group) == 1:
            selected.append(group.iloc[0])
        else:
            total = group.loc[group["aggregate"]]
            if len(total) != 1:
                raise ValueError(f"No unique whole-season aggregate for {year}/{bref}")
            selected.append(total.iloc[0])
    return pd.DataFrame(selected).rename(columns={"mp": "minutes", "games": "games_played"})[
        ["year", "bref_id", "player_name", "team", "position", "minutes", "games_played", "bpm", "obpm", "dbpm", "aggregate"]]


def exact_id_join(stats, index):
    index = index[["year", "bref_id", "nba_id"]].drop_duplicates().copy()
    index["nba_id"] = pd.to_numeric(index.nba_id, errors="raise")
    if index.nba_id.isna().any() or (index.nba_id <= 0).any():
        raise ValueError("Invalid NBA-id crosswalk")
    if (index.groupby(["year", "bref_id"]).nba_id.nunique().gt(1).any()
            or index.groupby(["year", "nba_id"]).bref_id.nunique().gt(1).any()):
        raise ValueError("Ambiguous exact BRef/NBA identity mapping")
    joined = stats.merge(index, on=["year", "bref_id"], how="left", validate="one_to_one")
    rejected = joined.loc[joined.nba_id.isna()].copy()
    joined = joined.loc[joined.nba_id.notna()].copy()
    joined["nba_id"] = joined.nba_id.astype(np.int64)
    if joined.duplicated(["year", "nba_id"]).any():
        raise ValueError("Duplicate joined player-season")
    return joined, rejected


def fetch(output=OUT, years=range(2017, 2027), interval=1.25):
    years = list(years)
    if not years or any(y not in range(2017, 2027) for y in years):
        raise ValueError("Only registered2017–2026 source seasons")
    if interval < 1:
        raise ValueError("At least one second between public-page requests")
    output = Path(output)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; local basketball research)"
    index = output / "index_master.csv"
    if not index.exists():
        response = session.get(INDEX_URL, timeout=30)
        response.raise_for_status()
        index.write_bytes(response.content)
        time.sleep(interval)
    frames, source_hashes = [], {str(index.resolve()): sha(index)}
    for year in years:
        path = raw / f"NBA_{year}_advanced.html"
        if not path.exists():
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            path.write_bytes(response.content)
            time.sleep(interval)
        frames.append(parse_advanced(path.read_text(), year))
        source_hashes[str(path.resolve())] = sha(path)
        print(json.dumps(dict(year=year, players=len(frames[-1]))), flush=True)
    stats = pd.concat(frames, ignore_index=True)
    joined, rejected = exact_id_join(stats, pd.read_csv(index).loc[lambda d: d.year.isin(years)])
    lines_path = ROOT / "results/estimated_skills/monthly_source_2014_2026/player_game_lines.parquet"
    lines = pd.read_parquet(lines_path, columns=["season", "season_type", "game_date"])
    last = lines.loc[lines.season_type.eq("rs")].groupby("season").game_date.max()
    joined["source_last_game_date"] = pd.to_datetime(joined.year.map(last))
    joined["earliest_use_date"] = joined.source_last_game_date + pd.Timedelta(days=1)
    joined["opponent_join_rule"] = "year == matchup ending season - 1; earlier complete RS only"
    joined["historical_publication_vintage_certified"] = False
    source_hashes[str(lines_path.resolve())] = sha(lines_path)
    source_hashes[str(Path(__file__).resolve())] = sha(__file__)
    joined.to_parquet(output / "player_seasons.parquet", index=False)
    rejected.to_csv(output / "unmatched_exact_ids.csv", index=False)
    manifest = dict(status="research_lagged_opponent_bpm_not_player_impact_prior",
        fetched_at=datetime.now(timezone.utc).isoformat(), years=years, rows=len(joined), unmatched_rows=len(rejected),
        by_year=joined.groupby("year").size().to_dict(),
        sources=source_hashes, index_url=INDEX_URL, stat_url="https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html",
        artifacts={name: sha(output / name) for name in ["player_seasons.parquet", "unmatched_exact_ids.csv"]},
        identity="exact year+bref_id; bidirectionally unambiguous crosswalk; no fuzzy joins",
        trade_rows="unique whole-season aggregate preferred; ambiguous aggregates rejected",
        time_rule="Use only immediately preceding complete ending-season row; never current-season final BPM before completion",
        limitation="Current BPM2.0 historical backhistory is retrospectively redefined; not certified pre2020 publication vintage",
        protected_outcomes=0, forbidden_use="Never substitute this opponent context into the player's own BPM prior or complete EPM")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ["rows", "unmatched_rows", "by_year"]}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--years", default="2017,2018,2019,2020,2021,2022,2023,2024,2025,2026")
    parser.add_argument("--interval", type=float, default=1.25)
    args = parser.parse_args()
    fetch(args.out, map(int, args.years.split(",")), args.interval)
