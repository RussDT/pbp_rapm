"""Stage-B deterministic corrections, each fit on strictly PRIOR seasons."""
import numpy as np, pandas as pd

AGE_BINS = [17, 21, 23, 25, 27, 29, 31, 33, 50]
GIDX_BINS = [-1, 2, 5, 10, 20, 1000]


def _wmean(v, w):
    w = np.asarray(w, float); v = np.asarray(v, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.sum(w[m] * v[m]) / np.sum(w[m])) if m.any() else np.nan


def player_season_rates(sub, spec):
    """Exposure-weighted player-season rate + exposure."""
    g = sub.groupby(["player_id", "season"], as_index=False).agg(
        num=(spec.numer, "sum"), den=("exposure", "sum"))
    g = g[g.den > 0].copy()
    g["rate"] = g.num / g.den
    return g


def fit_debut(ps, age, train_seasons, mean_rate):
    """Ridge of debut-season rate on repo-owned biographical covariates."""
    first = ps.groupby("player_id", as_index=False)["season"].min().rename(
        columns={"season": "first_season"})
    d = ps.merge(first, on="player_id")
    d = d[(d.season == d.first_season) & d.season.isin(train_seasons)]
    d = d.merge(age, on=["player_id", "season"], how="left")
    if len(d) < 40:
        return None, _wmean(d.rate, d.den) if len(d) else mean_rate
    X, y, w = _design(d), d.rate.values, d.den.values
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y, w = X[ok], y[ok], w[ok]
    Xm, Xs = X.mean(0), X.std(0); Xs[Xs == 0] = 1.0
    Z = np.c_[np.ones(len(X)), (X - Xm) / Xs]
    W = np.diag(w / w.mean())
    lam = np.eye(Z.shape[1]) * 5.0; lam[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ W @ Z + lam, Z.T @ W @ y)
    return {"beta": beta, "Xm": Xm, "Xs": Xs}, _wmean(d.rate, d.den)


def _design(d):
    dn = pd.to_numeric(d.get("draft_number"), errors="coerce")
    und = dn.isna().astype(float)
    dn = dn.fillna(61.0).clip(1, 61)
    dy = pd.to_numeric(d.get("draft_year"), errors="coerce")
    ysd = (d.season - dy).clip(-1, 20).fillna(1.0)
    return np.c_[d.age.fillna(24.0).values, ysd.values, dn.values, und.values,
                 d.get("height", pd.Series(78.0, index=d.index)).fillna(78.0).values,
                 d.get("weight", pd.Series(215.0, index=d.index)).fillna(215.0).values]


def apply_debut(model, fallback, rows_df):
    if model is None:
        return np.full(len(rows_df), fallback)
    X = _design(rows_df)
    Z = np.c_[np.ones(len(X)), (X - model["Xm"]) / model["Xs"]]
    out = Z @ model["beta"]
    return np.where(np.isfinite(out), out, fallback)


def fit_age_delta(ps, age, train_seasons):
    """Mean season-over-season rate change by age bin, on prior seasons only."""
    p = ps[ps.season.isin(train_seasons)].copy().sort_values(["player_id", "season"])
    p["prev_rate"] = p.groupby("player_id")["rate"].shift()
    p["prev_den"] = p.groupby("player_id")["den"].shift()
    p["prev_season"] = p.groupby("player_id")["season"].shift()
    p = p[(p.season - p.prev_season) == 1].merge(age, on=["player_id", "season"], how="left")
    if len(p) < 60:
        return {}
    p["delta"] = p.rate - p.prev_rate
    p["w"] = np.minimum(p.den, p.prev_den)
    p["ab"] = pd.cut(p.age, AGE_BINS)
    return {str(k): _wmean(g.delta, g.w) for k, g in p.groupby("ab", observed=True)}


def fit_early_season(sub, spec, train_seasons):
    """League within-season game-index effect (rate minus that season's mean)."""
    t = sub[sub.season.isin(train_seasons)].copy()
    if not len(t):
        return {}
    t["gidx"] = t.groupby(["player_id", "season"]).cumcount()
    t["gb"] = pd.cut(t.gidx, GIDX_BINS)
    smean = t.groupby("season").apply(
        lambda g: g[spec.numer].sum() / g["exposure"].sum())
    t["dev"] = t["obs_rate"] - t["season"].map(smean)
    return {str(k): _wmean(g.dev, g.exposure) for k, g in t.groupby("gb", observed=True)}
