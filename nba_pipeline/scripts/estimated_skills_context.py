"""Next-game context terms for the NBA estimated-skills filter (research lane).

Every term is a per-row additive offset on the one-step-ahead prediction that is
also removed from the innovation (pass it through ``run_filter_v2(transient=)``),
so the latent state tracks the player net of tonight's context.

Date safety:
- opponent / own-team terms are shrunk running means of BASE-filter one-step
  residuals from strictly earlier dates in the same season (same-date rows are
  excluded), plus the previous season's final opponent value;
- teammate absence uses only rosters of games already played: a teammate is a
  candidate if they missed the team's most recent game (streak = consecutive
  recent team games missed, up to MAX_STREAK). Each candidate's minutes share is
  weighted by the estimated chance they sit tonight, P(out | streak, minutes
  tier), fit on earlier seasons. Tonight's own roster is never read, so the
  feature is reproducible before tip. Each absent teammate's state is taken
  strictly before the date;
- coefficients are fit on seasons < S for target season S (expanding window).

Fairness rule (owner, 2026-09-14): every feature must be constructible before
tip from information that will exist in live use. An earlier variant read
tonight's realised PBP roster to define absence; it was removed because it is
not reproducible going forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_WINDOW = 10
RATE_WINDOW = 30
MIN_ABSENT_MINUTES = 10.0
MAX_STREAK = 20            # consecutive team games missed still treated as a candidate
STREAK_EDGES = np.array([2, 3, 4, 6, 11])   # buckets 1, 2, 3, 4-5, 6-10, 11-20
ROTATION_MINUTES = 20.0
FRESH_STREAK = 5           # absence terms split fresh (<=5) vs long (6-20)


def schedule_features(sub: pd.DataFrame) -> list[np.ndarray]:
    hp = (sub["step_k"] > 0).to_numpy()
    home = sub["is_home"].fillna(0.5).to_numpy() - 0.5
    gap = sub["days_gap"].to_numpy(); sc = sub["season_cross"].to_numpy() > 0
    b2b = ((gap == 1) & ~sc & hp).astype(float)
    rest3 = ((gap >= 3) & ~sc & hp).astype(float)
    return [home, b2b, rest3]


def _per_game_weight(d: pd.DataFrame, key: str, reference_end=None) -> float:
    if reference_end is not None:
        d = d.loc[d.season.le(reference_end)]
    return float(d.groupby(["season", key, "gid"]).w.sum().mean())


def running_group_residual(sub, resid, expo, c, key="opp_team_id", reference_end=None):
    """Shrunk season-to-date mean residual for the row's group, strictly earlier dates."""
    d = pd.DataFrame({"season": sub["season"].values, key: sub[key].values,
                      "date": sub["game_date"].values, "wr": resid * expo, "w": expo,
                      "gid": sub["game_id"].values})
    d = d[np.isfinite(d.wr) & d[key].notna()]
    g = (d.groupby(["season", key, "date"]).agg(wr=("wr", "sum"), w=("w", "sum"))
         .reset_index().sort_values(["season", key, "date"]))
    k = g.groupby(["season", key])
    g["cwr"] = k.wr.cumsum() - g.wr
    g["cw"] = k.w.cumsum() - g.w
    g["val"] = g.cwr / (g.cw + c * _per_game_weight(d, key, reference_end))
    rows = pd.DataFrame({"season": sub["season"].values, key: sub[key].values,
                         "date": sub["game_date"].values})
    return rows.merge(g[["season", key, "date", "val"]], how="left").val.fillna(0.0).to_numpy()


def previous_season_group_residual(sub, resid, expo, c, key="opp_team_id", reference_end=None):
    d = pd.DataFrame({"season": sub["season"].values, key: sub[key].values,
                      "wr": resid * expo, "w": expo, "gid": sub["game_id"].values})
    d = d[np.isfinite(d.wr) & d[key].notna()]
    g = d.groupby(["season", key]).agg(wr=("wr", "sum"), w=("w", "sum")).reset_index()
    g["val"] = g.wr / (g.w + c * _per_game_weight(d, key, reference_end))
    g["season"] = g["season"] + 1
    rows = pd.DataFrame({"season": sub["season"].values, key: sub[key].values})
    return rows.merge(g[["season", key, "val"]], how="left").val.fillna(0.0).to_numpy()


def absent_teammates(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """(season, game_id, game_date, team_id, player_id, streak, tmin, p_play): teammates who
    missed the team's most recent completed game, with an estimated chance they play tonight.

    ``streak`` counts consecutive most-recent team games missed (1..MAX_STREAK) and uses only
    rosters of games completed before tip. ``p_play`` = P(plays | streak bucket, minutes tier)
    estimated from earlier seasons only (the first panel season uses its own rates; it is a
    warm-up that is never scored). ``played`` is the realised label, kept only to build that
    table for later seasons and never used as a feature."""
    tg = ctx.merge(df[["game_id", "player_id", "season", "game_date"]].drop_duplicates(),
                   how="inner", on=["game_id", "player_id"])
    games = (tg[["season", "team_id", "game_id", "game_date"]].drop_duplicates()
             .sort_values(["season", "team_id", "game_date", "game_id"]))
    games["n"] = games.groupby(["season", "team_id"]).cumcount()
    ng = games.groupby(["season", "team_id"]).n.max().rename("n_last").reset_index()
    app = (tg.merge(games, on=["season", "team_id", "game_id", "game_date"])[
        ["season", "team_id", "n", "player_id"]].drop_duplicates()
        .sort_values(["season", "team_id", "player_id", "n"]))
    app["n_next"] = app.groupby(["season", "team_id", "player_id"]).n.shift(-1)
    app = app.merge(ng, on=["season", "team_id"])
    # appearance at L, next appearance N: games n = L+2..N have last appearance L and missed n-1
    app["stop"] = np.minimum(app.n_next.fillna(app.n_last + 1).clip(upper=app.n_last + 1) - 1
                             + app.n_next.notna(), app.n + 1 + MAX_STREAK)
    app = app[app.stop >= app.n + 2]
    reps = (app.stop - app.n - 1).astype(int).to_numpy()
    ab = app.loc[app.index.repeat(reps), ["season", "team_id", "player_id", "n", "n_next"]].copy()
    ab["streak"] = ab.groupby(level=0).cumcount().to_numpy() + 1
    ab["played"] = (ab.n + 1 + ab.streak == ab.n_next).astype(float)
    ab["n"] = ab.n + 1 + ab.streak
    ab = ab.merge(games, on=["season", "team_id", "n"])[
        ["season", "game_id", "game_date", "team_id", "player_id", "streak", "played"]]
    mst = df.sort_values(["player_id", "game_date"], kind="stable")[["player_id", "game_date", "minutes"]].copy()
    mst["tmin"] = mst.groupby("player_id").minutes.transform(
        lambda x: x.rolling(MIN_WINDOW, min_periods=1).mean())
    ab = asof_state(ab, mst, ["tmin"])
    ab = ab[ab.tmin.notna() & (ab.tmin >= MIN_ABSENT_MINUTES)].reset_index(drop=True)
    ab["cell"] = (np.searchsorted(STREAK_EDGES, ab.streak, side="right") * 2
                  + (ab.tmin >= ROTATION_MINUTES))
    ab["p_play"] = np.nan
    seasons = sorted(ab.season.unique())
    for i, S in enumerate(seasons):
        hist = ab[ab.season < S] if i else ab[ab.season == S]
        tab = hist.groupby("cell").played.agg(["sum", "count"])
        p = (tab["sum"] + 1.0) / (tab["count"] + 2.0)
        m = ab.season == S
        ab.loc[m, "p_play"] = ab.loc[m, "cell"].map(p).fillna(0.5).to_numpy()
    return ab


def asof_state(rows: pd.DataFrame, state: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    state = state.sort_values("game_date", kind="stable"); rows = rows.sort_values("game_date", kind="stable")
    return pd.merge_asof(rows, state[["player_id", "game_date"] + cols], on="game_date",
                         by="player_id", allow_exact_matches=False)


def teammate_absence_features(sub, spec, expo, centers_asof, pred_level, absent, reference_end=None):
    """Expected absent-teammate terms: each candidate's minutes share is weighted by the
    estimated chance they sit tonight (1 - p_play), split fresh (streak <= FRESH_STREAK) vs
    long absences, whose effect the filter state has partly absorbed:
    [E out share fresh, E out skill excess fresh, same for long, total E share x focal excess]."""
    hp = (sub["step_k"] > 0).to_numpy()
    st = pd.DataFrame({"player_id": sub["player_id"].values, "game_date": sub["game_date"].values,
                       "nu": sub[spec.numer].astype(float).values, "de": expo, "cen": centers_asof})
    g = st.groupby("player_id")
    st["rn"] = g.nu.transform(lambda x: x.rolling(RATE_WINDOW, min_periods=1).sum())
    st["rd"] = g.de.transform(lambda x: x.rolling(RATE_WINDOW, min_periods=1).sum())
    reference_expo = expo if reference_end is None else expo[sub.season.le(reference_end).to_numpy()]
    k0 = 5.0 * float(np.median(reference_expo))
    st["excess"] = (st.rn + k0 * st.cen) / (st.rd + k0) - st.cen
    a = asof_state(absent, st, ["excess"]).fillna({"excess": 0.0})
    a["eshare"] = (a.tmin / 48.0) * (1.0 - a.p_play)
    a["ex"] = a.eshare * a.excess
    fr = (a.streak <= FRESH_STREAK).astype(float)
    agg = (a.assign(m1=a.eshare * fr, x1=a.ex * fr, m2=a.eshare * (1 - fr), x2=a.ex * (1 - fr))
           .groupby(["game_id", "team_id"])[["m1", "x1", "m2", "x2"]].sum().reset_index())
    km = pd.DataFrame({"game_id": sub["game_id"].values, "team_id": sub["team_id"].values}).merge(
        agg, how="left", on=["game_id", "team_id"]).fillna(0.0)
    f = [km[c].to_numpy() * hp for c in ("m1", "x1", "m2", "x2")]
    return f + [(f[0] + f[2]) * (pred_level - centers_asof)]


def expanding_linear_context(X, resid, expo, seasons, has_prior, target_seasons, lam=1e-6):
    """ctx[S] = X[S] @ beta fit on rows with season < S. Returns (ctx, betas)."""
    ctx = np.zeros(len(seasons)); betas = {}
    for S in target_seasons:
        m = (seasons < S) & has_prior & np.isfinite(resid)
        if not m.any():
            continue
        Xw = X[m] * expo[m, None]
        b = np.linalg.solve(X[m].T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ resid[m])
        te = seasons == S
        ctx[te] = X[te] @ b; betas[int(S)] = b.tolist()
    return ctx, betas
