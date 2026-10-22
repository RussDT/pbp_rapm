"""Per (game_id, player_id): home flag, team id, opponent team id, from raw PBP.

Side comes from the on-court lineup columns (home_player1-5 / away_player1-5);
team ids come from the modal player1_team_id among each side's players. Used only
as next-game context for the estimated-skills filter. The frozen holdout season
is not built unless --include-holdout is passed.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

RAW = Path(__file__).resolve().parents[1] / "raw_data"
OUT = Path(__file__).resolve().parents[1] / "results" / "estimated_skills" / "game_context.parquet"
H = [f"home_player{i}" for i in range(1, 6)]
A = [f"away_player{i}" for i in range(1, 6)]


def one_file(path):
    df = pd.read_parquet(path, columns=["game_id", "player1_id", "player1_team_id"] + H + A)
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    rows = []
    for side, cols, home in (("h", H, 1), ("a", A, 0)):
        m = df[["game_id"] + cols].melt("game_id", value_name="player_id").drop(columns="variable")
        m = m[m.player_id.notna() & (m.player_id != 0)].drop_duplicates()
        m["player_id"] = m.player_id.astype(np.int64); m["is_home"] = home
        rows.append(m)
    side = pd.concat(rows, ignore_index=True)
    # a player seen on both sides in one game is a lineup error: drop
    amb = side.groupby(["game_id", "player_id"]).is_home.nunique()
    side = side.set_index(["game_id", "player_id"]).drop(amb[amb > 1].index).reset_index()
    ev = df[df.player1_id.notna() & df.player1_team_id.notna()][["game_id", "player1_id", "player1_team_id"]]
    ev = ev.rename(columns={"player1_id": "player_id", "player1_team_id": "team_id"})
    ev["player_id"] = ev.player_id.astype(np.int64); ev["team_id"] = ev.team_id.astype(np.int64)
    ev = ev[ev.team_id > 0].merge(side, on=["game_id", "player_id"])
    team = ev.groupby(["game_id", "is_home"]).team_id.agg(lambda s: s.mode().iloc[0]).unstack()
    team.columns = ["away_team", "home_team"]
    side = side.merge(team.reset_index(), on="game_id")
    side["team_id"] = np.where(side.is_home == 1, side.home_team, side.away_team)
    side["opp_team_id"] = np.where(side.is_home == 1, side.away_team, side.home_team)
    return side[["game_id", "player_id", "is_home", "team_id", "opp_team_id"]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--include-holdout", action="store_true")
    a = ap.parse_args()
    yrs = list(range(19, 26)) + ([26] if a.include_holdout else [])
    parts = [one_file(p) for y in yrs for p in (RAW / f"NBA{y}.parquet", RAW / f"NBA{y}_PS.parquet") if p.exists()]
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["game_id", "player_id"])
    out.to_parquet(OUT, index=False)
    print(f"{len(out):,} player-games, {out.game_id.nunique():,} games -> {OUT}")


if __name__ == "__main__":
    main()
