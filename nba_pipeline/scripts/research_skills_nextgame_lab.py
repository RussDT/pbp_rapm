"""Next-game estimated-skills lab: paired variant tests on top of the v4 pipeline.

Purpose: a fast, reproducible harness for testing modelling theories against the
selected next-game candidate (v4 = OU state + prior-season corrections + pre-tip
context) on the owner's target, one-step-ahead exposure-weighted game MSE on dev
2020-2025. 2026 is dropped at load and never scored here; 2027 is untouched.

Every variant is scored on the identical eligible rows and exposure weights as
its baseline, with a game-clustered paired bootstrap and per-season wins. All
variants are re-tuned with the same DE seed/budget as v4 (plus an identical
Nelder-Mead polish), so the comparison is symmetric: baseline and challenger
get the same optimiser.

Variants (``--variants``), each a single change on top of ``base``:

- ``base``      exact v4 reconstruction (closure-checked against v4_nextgame.json).
- ``fixr``      fix ``r`` to its natural scale and tune 5 OU params + lam. The
                six OU params are jointly scale-invariant for predictions
                (Kalman gain depends only on ratios), so v4's DE spends budget on
                a flat direction; this is an optimiser change, not a model change.
- ``noise``     heteroskedastic observation noise: binomial ``r*p(1-p)/n`` for
                pct skills, Poisson ``r*level/n`` for rate skills, with the level
                taken from the causal one-step prediction.
- ``ctxplus``   extra pre-tip context columns: playoff flag, playoff x focal
                excess, minutes trend (mean of last 3 prior games minus last 30).
- ``ctxmult``   rate skills: context fit on relative residuals and applied
                multiplicatively (``level * (1 + lam*ctx)``) so opponent/team
                effects scale with the player's own level.
- ``center``    rate skills: as-of league centering (obs minus running league
                rate), so the OU attractor follows league drift.
- ``clip``      robust update: innovations clipped at ``k`` predictive SD, ``k``
                tuned in [1.5, 8] (8 == no clipping in practice).
- ``xusg``      cross-skill form: focal player's shrunk trailing-5-game usage
                residual (base usage filter, strictly prior games) as a context
                column for every non-usage skill.
- ``ctxps``     playoff flag + playoff x focal excess only (ctxplus ablation).
- ``ctxmin``    minutes trend only (ctxplus ablation).
- ``xall``      cross-skill form: shrunk trailing-5 base-filter residuals of every
                other skill (strictly prior games) as context columns.
- ``jump``      team-change variance inflation: extra process noise ``q_jump``
                (tuned) on the slow component when the player's team id differs
                from his previous game (DARKO's "learning rate" bump).
- ``a+b+c``     any ``+``-joined list composes options (e.g.
                ``ctxplus+ctxmult+xusg``); ``--combo a,b,c`` is the legacy form.
                Pairwise paired comparisons between all run variants are stored
                under ``pairs`` so increments (A minus B) carry their own CI.

Run: ``.venv/bin/python nba_pipeline/scripts/research_skills_nextgame_lab.py --variants base,fixr,noise``
Outputs: ``results/estimated_skills/nextgame_lab/<variant>.json`` (ignored).
"""
from __future__ import annotations

import argparse, json, multiprocessing, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if __name__ == "__main__" and "--full-history" in sys.argv:
    # Dispatch before loading legacy lab adapters: the new profile has its own
    # pinned observations and never requires the old research caches.
    from full_skill_cross_talk import main as full_history_main
    full_history_main([arg for arg in sys.argv[1:] if arg != "--full-history"])
    raise SystemExit(0)

import numpy as np, pandas as pd
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nba_estimated_skills import (  # noqa: E402
    SKILLS, load_panel, skill_subpanel, league_center, weighted_mse,
    OUT_DIR, DEV_SEASONS, HOLDOUT_SEASON, SEED)
from build_nba_estimated_skills_v2 import LayoutV2, bounds_v2, V2_PARAMS  # noqa: E402
from build_nba_estimated_skills_v2_stageb import build_corrections, AGE_PATH  # noqa: E402
from build_nba_estimated_skills_v4 import CTX_PATH, game_bootstrap  # noqa: E402
import estimated_skills_context as X  # noqa: E402

LAB_DIR = OUT_DIR / "nextgame_lab"
V4_PATH = OUT_DIR / "v4_nextgame.json"
STAGE_A_PATH = OUT_DIR / "v2_stage_a.json"
KNOWN = ("base", "fixr", "noise", "ctxplus", "ctxmult", "center", "clip", "xusg",
         "ctxps", "ctxmin", "xall", "jump", "combo")


# --------------------------------------------------------------------------
# Engine: v2 OU filter with lab options (const/binom/poisson noise, multiplicative
# context, innovation clipping). With all options off it is run_filter_v2 exactly.
# --------------------------------------------------------------------------


def run_filter_lab(layout, obs, params, mean_rate, mu0=None, transient=None,
                   level_shift=None, centers=None, noise="const", noise_floor=1e-3,
                   mult=None, clip_k=None, jump=None, q_jump=0.0):
    tau_mu, s_mu, tau_d, s_d, q_off, r = params[:6]
    n = layout.n_players
    mu = np.zeros(n); d = np.zeros(n)
    Pmm = np.zeros(n); Pmd = np.zeros(n); Pdd = np.zeros(n)
    pred = np.empty(layout.n_rows); pred_var = np.empty(layout.n_rows)
    post = np.empty(layout.n_rows); post_var = np.empty(layout.n_rows)
    for k, rows in enumerate(layout.step_rows):
        pl = layout.player_idx[rows]
        if k == 0:
            pm = (mean_rate * np.ones(rows.shape[0])) if mu0 is None else mu0[rows]
            pd_ = np.zeros(rows.shape[0])
            amm = np.full(rows.shape[0], s_mu); amd = np.zeros(rows.shape[0])
            add = np.full(rows.shape[0], s_d)
        else:
            dt = layout.dt[rows]; sc = layout.season_cross[rows]
            am = np.exp(-dt / tau_mu); ad = np.exp(-dt / tau_d)
            anchor = mean_rate + (0.0 if level_shift is None else level_shift[rows] * sc)
            pm = anchor + am * (mu[pl] - anchor)
            pd_ = ad * d[pl]
            amm = am * am * Pmm[pl] + s_mu * (1.0 - am * am) + q_off * sc
            if jump is not None:
                amm = amm + q_jump * jump[rows]
            amd = am * ad * Pmd[pl]
            add = ad * ad * Pdd[pl] + s_d * (1.0 - ad * ad)
        base = pm + pd_
        cen = 0.0 if centers is None else centers[rows]
        p_row = base
        if mult is not None:
            p_row = p_row + mult[rows] * (base + cen)
        if transient is not None:
            p_row = p_row + transient[rows]
        hph = amm + 2.0 * amd + add
        if noise == "const":
            obs_var = r / layout.exposure[rows]
        else:
            lvl = p_row + cen
            if noise == "binom":
                lvl = np.clip(lvl, 0.02, 0.98); scale = lvl * (1.0 - lvl)
            else:  # poisson
                scale = np.maximum(lvl, noise_floor)
            obs_var = r * scale / layout.exposure[rows]
        S = hph + obs_var
        c0 = amm + amd; c1 = amd + add
        Km = c0 / S; Kd = c1 / S
        innov = obs[rows] - p_row
        if clip_k is not None:
            lim = clip_k * np.sqrt(S)
            innov = np.clip(innov, -lim, lim)
        mu[pl] = pm + Km * innov
        d[pl] = pd_ + Kd * innov
        Pmm[pl] = amm - Km * c0
        Pmd[pl] = amd - Km * c1
        Pdd[pl] = add - Kd * c1
        pred[rows] = p_row; pred_var[rows] = hph
        post[rows] = mu[pl] + d[pl]
        post_var[rows] = Pmm[pl] + 2.0 * Pmd[pl] + Pdd[pl]
    return pred, pred_var, post, post_var


# --------------------------------------------------------------------------
# Extra pre-tip context columns
# --------------------------------------------------------------------------


def playoff_features(sub, pred_level, centers):
    hp = (sub["step_k"] > 0).to_numpy()
    ps = (sub["season_type"].to_numpy() == "ps").astype(float) * hp
    return [ps, ps * (pred_level - centers)]


def minutes_trend_feature(sub, short=3, long=30):
    hp = (sub["step_k"] > 0).to_numpy()
    g = sub.groupby("player_id")["minutes"]
    prev = g.shift()
    m_s = prev.groupby(sub["player_id"]).transform(lambda x: x.rolling(short, min_periods=1).mean())
    m_l = prev.groupby(sub["player_id"]).transform(lambda x: x.rolling(long, min_periods=1).mean())
    f = ((m_s - m_l) / 48.0).fillna(0.0).to_numpy()
    return [f * hp]


def team_change_flag(sub):
    """1.0 when the player's team id differs from his previous game (both known)."""
    hp = (sub["step_k"] > 0).to_numpy()
    tid = sub["team_id"].to_numpy(dtype=float)
    prev = sub.groupby("player_id")["team_id"].shift().to_numpy(dtype=float)
    return (hp & np.isfinite(tid) & np.isfinite(prev) & (tid != prev)).astype(float)


def usage_form_feature(sub, usg_resid_tab, c_games=5.0):
    """Shrunk trailing-5 base-filter residual of the focal player in another
    skill, strictly prior games (used for ``xusg`` with usage and ``xall`` with
    every other skill)."""
    hp = (sub["step_k"] > 0).to_numpy()
    t = usg_resid_tab.sort_values(["player_id", "game_date"]).copy()
    g = t.groupby("player_id")
    t["wr"] = g["wr"].transform(lambda x: x.rolling(5, min_periods=1).sum())
    t["w"] = g["w"].transform(lambda x: x.rolling(5, min_periods=1).sum())
    t["val"] = t.wr / (t.w + c_games * float(np.median(usg_resid_tab.w)))
    rows = pd.DataFrame({"player_id": sub["player_id"].values, "game_date": sub["game_date"].values,
                         "_i": np.arange(len(sub))}).sort_values("game_date")
    m = pd.merge_asof(rows, t[["player_id", "game_date", "val"]].sort_values("game_date"),
                      on="game_date", by="player_id", allow_exact_matches=False)
    f = m.sort_values("_i")["val"].fillna(0.0).to_numpy()
    return [f * hp]


# --------------------------------------------------------------------------
# Per-skill preparation (shared by all variants) and variant assembly
# --------------------------------------------------------------------------


def prepare(spec, df, ctx_tab, age, absent, stage_a, v4, usg_resid_tab=None, form_tabs=None, context_reference_end=None):
    sub = skill_subpanel(df, spec)
    obs = sub["obs_rate"].to_numpy(); se = sub["season"].to_numpy()
    hp = (sub["step_k"] > 0).to_numpy(); dev = np.isin(se, DEV_SEASONS) & hp
    pre = se < HOLDOUT_SEASON
    lay = LayoutV2(sub); expo = lay.exposure
    mean_rate = float(sub.loc[pre, spec.numer].sum() / sub.loc[pre, "exposure"].sum())
    sa = stage_a[spec.name]
    cen_asof, _ = league_center(sub, spec)
    seasons_all = sorted(df["season"].unique())
    if sa["centered"]:
        centers = cen_asof; obs_fit = obs - centers; mr = 0.0
    else:
        centers = np.zeros(len(sub)); obs_fit = obs; mr = mean_rate
    bv = float(np.sum(expo[pre] * (obs_fit[pre] - mr) ** 2) / np.sum(expo[pre]))
    # base filter (stage A params) -> residuals feeding the context terms
    pA, *_ = run_filter_lab(lay, obs_fit, sa["v2"]["params"], mr); pA = pA + centers
    resid0 = np.where(hp, obs - pA, np.nan)
    c_sel = v4[spec.name]["context_c"]
    sub_team = sub.drop(columns="opp_team_id").assign(opp_team_id=sub["team_id"].values)
    absence = ([] if absent is None or spec.kind != "rate" else
               X.teammate_absence_features(sub, spec, expo, cen_asof, pA, absent, reference_end=context_reference_end))
    cols = X.schedule_features(sub) + [
        X.running_group_residual(sub, resid0, expo, c_sel, reference_end=context_reference_end),
        X.previous_season_group_residual(sub, resid0, expo, c_sel, reference_end=context_reference_end),
        X.running_group_residual(sub_team, resid0, expo, c_sel, reference_end=context_reference_end)] + absence
    mu0_raw, shift, trans, _ = build_corrections(sub, spec, age, mean_rate, seasons_all)
    return dict(spec=spec, sub=sub, obs=obs, se=se, hp=hp, dev=dev, pre=pre, lay=lay, expo=expo,
                mean_rate=mean_rate, centered=bool(sa["centered"]), cen_asof=cen_asof,
                centers=centers, obs_fit=obs_fit, mr=mr, bv=bv, pA=pA, resid0=resid0,
                c_sel=c_sel, ctx_cols=cols, mu0_raw=mu0_raw, shift=shift, trans=trans,
                seasons_all=seasons_all, games=sub["game_id"].to_numpy(),
                v4_params=v4[spec.name]["params"], v4_obj=v4[spec.name]["obj_v4"],
                usg_resid_tab=usg_resid_tab, form_tabs=form_tabs or {},
                team_change=team_change_flag(sub))


def assemble(prep, opts):
    """Build (run, bounds, x0, meta) for a variant described by ``opts`` (set of names)."""
    spec = prep["spec"]; sub = prep["sub"]; expo = prep["expo"]; se = prep["se"]; hp = prep["hp"]
    is_rate = spec.kind == "rate"
    centers = prep["centers"]; obs_fit = prep["obs_fit"]; mr = prep["mr"]; bv = prep["bv"]
    if "center" in opts and is_rate and not prep["centered"]:
        centers = prep["cen_asof"]; obs_fit = prep["obs"] - centers; mr = 0.0
        bv = float(np.sum(expo[prep["pre"]] * obs_fit[prep["pre"]] ** 2) / np.sum(expo[prep["pre"]]))
    cols = list(prep["ctx_cols"])
    if "ctxplus" in opts or "ctxps" in opts:
        cols += playoff_features(sub, prep["pA"], prep["cen_asof"])
    if "ctxplus" in opts or "ctxmin" in opts:
        cols += minutes_trend_feature(sub)
    if "xusg" in opts and spec.name != "usg_pct" and prep["usg_resid_tab"] is not None:
        cols += usage_form_feature(sub, prep["usg_resid_tab"])
    if "xall" in opts:
        for other, tab in prep["form_tabs"].items():
            if other != spec.name:
                cols += usage_form_feature(sub, tab)
    Xd = np.column_stack(cols)
    resid = prep["resid0"]
    mult = None
    if "ctxmult" in opts and is_rate:
        lvl = np.maximum(prep["pA"], 0.05 * prep["mean_rate"])
        rel = resid / lvl
        ctx_rel, betas = X.expanding_linear_context(Xd, rel, expo * lvl ** 2, se, hp,
                                                    prep["seasons_all"][1:])
        ctx = np.zeros(len(sub)); mult_base = ctx_rel
    else:
        ctx, betas = X.expanding_linear_context(Xd, resid, expo, se, hp, prep["seasons_all"][1:])
        mult_base = None
    mu0 = prep["mu0_raw"] - centers
    shift = prep["shift"]; trans = prep["trans"]
    noise = "const"
    if "noise" in opts:
        noise = "poisson" if is_rate else "binom"
    floor = 0.05 * prep["mean_rate"]
    mean_expo = float(expo[prep["pre"]].mean())
    bounds = bounds_v2(bv, mean_expo)
    r_nat = bv * mean_expo
    if noise != "const":
        bounds[5] = (0.01, 100.0)          # dimensionless overdispersion
        r_nat = 1.0
    names = list(V2_PARAMS)
    if "fixr" in opts:
        bounds = bounds[:5]; names = names[:5]
    bounds = bounds + [(0.0, 2.0)]; names = names + ["ctx_lambda"]
    if "clip" in opts:
        bounds = bounds + [(1.5, 8.0)]; names = names + ["clip_k"]
    jump = prep["team_change"] if "jump" in opts else None
    if "jump" in opts:
        bounds = bounds + [(0.0, 2.0 * bv)]; names = names + ["q_jump"]

    def unpack(p):
        p = list(p); i = 0
        ou = p[:5] + [r_nat] if "fixr" in opts else p[:6]
        i = 5 if "fixr" in opts else 6
        lam = p[i]; i += 1
        ck = None
        if "clip" in opts:
            ck = p[i]; i += 1
        qj = p[i] if "jump" in opts else 0.0
        return ou, lam, ck, qj

    def run(p):
        ou, lam, ck, qj = unpack(p)
        tr = trans + (0.0 if mult_base is not None else lam * ctx)
        mu_ = None if mult_base is None else lam * mult_base
        pred, pvar, post, postv = run_filter_lab(
            prep["lay"], obs_fit, ou, mr, mu0=mu0, transient=tr, level_shift=shift,
            centers=centers, noise=noise, noise_floor=floor, mult=mu_, clip_k=ck,
            jump=jump, q_jump=qj)
        return pred + centers, pvar, post + centers, postv, ou

    meta = dict(noise=noise, centered=bool(centers is not prep["centers"] or prep["centered"]),
                n_ctx=Xd.shape[1], mult=mult_base is not None, names=names, bounds=bounds,
                betas_last=(betas[max(betas)] if betas else None))
    return run, bounds, names, meta


def tune(run, bounds, obs, expo, dev, maxiter, popsize, polish_fev, seed=SEED):
    def obj(p):
        return weighted_mse(run(p)[0], obs, expo, dev)
    res = differential_evolution(obj, bounds=bounds, seed=seed, maxiter=maxiter,
                                 popsize=popsize, tol=1e-7, polish=False)
    x = np.array(res.x, float); f_de = float(res.fun)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    if polish_fev > 0:
        # log-space Nelder-Mead inside the box (positive params); clip at bounds
        def tobj(z):
            p = np.clip(np.exp(z), np.maximum(lo, 1e-15), hi)
            return obj(p)
        z0 = np.log(np.maximum(x, 1e-15))
        r2 = minimize(tobj, z0, method="Nelder-Mead",
                      options=dict(maxfev=polish_fev, xatol=1e-4, fatol=1e-12))
        xp = np.clip(np.exp(r2.x), np.maximum(lo, 1e-15), hi)
        if float(r2.fun) < f_de:
            x = xp
    return [float(v) for v in x], f_de, float(obj(x)), bool(res.success), int(res.nit)


def evaluate(pred_ref, pred, obs, expo, se, dev, games, boot, rng):
    gain, ci = game_bootstrap(obs - pred_ref, obs - pred, expo, games, dev, boot, rng)
    per = {}
    for S in DEV_SEASONS:
        m = dev & (se == S)
        a = weighted_mse(pred_ref, obs, expo, m); b = weighted_mse(pred, obs, expo, m)
        per[int(S)] = 100 * (a - b) / a
    return dict(gain_pct=gain, ci95=ci, seasons_won=int(sum(v > 0 for v in per.values())),
                per_season_pct=per)


def worker(args):
    spec_name, variants, combo, maxiter, popsize, polish_fev, boot = args
    t0 = time.time()
    spec = next(s for s in SKILLS if s.name == spec_name)
    G = _globals
    prep = prepare(spec, G["df"], G["ctx"], G["age"], G["absent"], G["stage_a"], G["v4"],
                   G["usg_resid_tab"], G.get("form_tabs"))
    obs, expo, se, dev, games = prep["obs"], prep["expo"], prep["se"], prep["dev"], prep["games"]
    rng = np.random.default_rng(SEED)
    out = {}
    # closure: v4 params through the lab engine must reproduce obj_v4
    run_b, bounds_b, names_b, meta_b = assemble(prep, set())
    pred_v4 = run_b(prep["v4_params"])[0]
    obj_v4 = weighted_mse(pred_v4, obs, expo, dev)
    out["closure"] = dict(obj_v4_json=prep["v4_obj"], obj_v4_lab=obj_v4,
                          abs_gap=abs(obj_v4 - prep["v4_obj"]))
    # baseline: v4 reconstruction re-tuned with the lab optimiser (symmetric protocol)
    pb, f_de, f_pol, ok, nit = tune(run_b, bounds_b, obs, expo, dev, maxiter, popsize, polish_fev)
    pred_base = run_b(pb)[0]
    out["base"] = dict(params=dict(zip(names_b, pb)), obj_de=f_de, obj=f_pol, converged=ok, nit=nit,
                       vs_v4=evaluate(pred_v4, pred_base, obs, expo, se, dev, games, boot, rng),
                       meta={k: v for k, v in meta_b.items() if k != "bounds"})
    preds = {}
    for v in variants:
        if v == "base":
            continue
        opts = set(combo) if v == "combo" else set(v.split("+"))
        if ("center" in opts or "ctxmult" in opts) and spec.kind != "rate" and len(opts) == 1:
            out[v] = dict(skipped="pct skill"); continue
        if ("xusg" in opts and spec.name == "usg_pct" and len(opts) == 1):
            out[v] = dict(skipped="own skill"); continue
        t1 = time.time()
        run_v, bounds_v, names_v, meta_v = assemble(prep, opts)
        pv, f_de, f_pol, ok, nit = tune(run_v, bounds_v, obs, expo, dev, maxiter, popsize, polish_fev)
        pred_v = run_v(pv)[0]; preds[v] = pred_v
        out[v] = dict(params=dict(zip(names_v, pv)), obj_de=f_de, obj=f_pol, converged=ok, nit=nit,
                      vs_base=evaluate(pred_base, pred_v, obs, expo, se, dev, games, boot, rng),
                      vs_v4=evaluate(pred_v4, pred_v, obs, expo, se, dev, games, boot, rng),
                      meta={k: v for k, v in meta_v.items() if k != "bounds"},
                      seconds=round(time.time() - t1, 1))
    # pairwise paired comparisons between run variants (row = challenger, col = reference)
    names = list(preds)
    out["pairs"] = {f"{a}|{b}": evaluate(preds[b], preds[a], obs, expo, se, dev, games, boot, rng)
                    for a in names for b in names if a != b}
    out["seconds"] = round(time.time() - t0, 1); out["n_dev"] = int(dev.sum())
    return spec_name, out


_globals = {}


def _form_table(df, spec, stage_a):
    sub = skill_subpanel(df, spec); lay = LayoutV2(sub)
    obs = sub["obs_rate"].to_numpy(); pre = sub["season"].to_numpy() < HOLDOUT_SEASON
    sa = stage_a[spec.name]
    if sa["centered"]:
        cen, _ = league_center(sub, spec); obs_fit = obs - cen; mr = 0.0
    else:
        cen = 0.0; obs_fit = obs
        mr = float(sub.loc[pre, spec.numer].sum() / sub.loc[pre, "exposure"].sum())
    pA, *_ = run_filter_lab(lay, obs_fit, sa["v2"]["params"], mr); pA = pA + cen
    hp = (sub["step_k"] > 0).to_numpy()
    return pd.DataFrame({"player_id": sub["player_id"].values,
                         "game_date": sub["game_date"].values,
                         "wr": np.where(hp, (obs - pA) * lay.exposure, 0.0),
                         "w": np.where(hp, lay.exposure, 0.0)})


def load_all(need_usg, need_all=False):
    df = load_panel()
    df = df[df["season"] < HOLDOUT_SEASON].copy()      # 2026 never enters
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    ctx = pd.read_parquet(CTX_PATH)
    assert int((ctx["game_id"].str.len() != 10).sum()) == 0
    df = df.merge(ctx, on=["game_id", "player_id"], how="left")
    age = pd.read_parquet(AGE_PATH)
    stage_a = json.load(open(STAGE_A_PATH)); v4 = json.load(open(V4_PATH))
    absent = X.absent_teammates(df, ctx)
    usg_tab = None; form_tabs = {}
    if need_all:
        form_tabs = {s.name: _form_table(df, s, stage_a) for s in SKILLS}
        usg_tab = form_tabs["usg_pct"]
    elif need_usg:
        usg_tab = _form_table(df, next(s for s in SKILLS if s.name == "usg_pct"), stage_a)
    _globals.update(df=df, ctx=ctx, age=age, stage_a=stage_a, v4=v4, absent=absent,
                    usg_resid_tab=usg_tab, form_tabs=form_tabs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default="")
    ap.add_argument("--variants", default="base,fixr,noise")
    ap.add_argument("--combo", default="", help="comma list of option names for the combo variant")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--polish", type=int, default=600, help="Nelder-Mead maxfev after DE (0=off)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=17)
    ap.add_argument("--tag", default="")
    ap.add_argument("--serial", action="store_true", help="run skills in-process (debug)")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    for v in variants:
        for part in v.split("+"):
            assert part in KNOWN, part
    combo = [c for c in args.combo.split(",") if c]
    specs = [s.name for s in SKILLS if not args.skills or s.name in args.skills.split(",")]
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    parts = {p for v in variants for p in v.split("+")} | set(combo)
    need_usg = "xusg" in parts; need_all = "xall" in parts
    t0 = time.time(); load_all(need_usg, need_all); print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    jobs = [(s, variants, combo, args.maxiter, args.popsize, args.polish, args.boot) for s in specs]
    results = {}

    def _report(name, out):
        results[name] = out
        c = out["closure"]
        print(f"{name:<18} closure gap {c['abs_gap']:.2e}  base vs v4 {out['base']['vs_v4']['gain_pct']:+.3f}%"
              f"  ({out['seconds']:.0f}s)", flush=True)

    if args.serial or args.workers <= 1:
        for j in jobs:
            _report(*worker(j))
    else:
        # spawn + per-worker initializer: fork on macOS crashes inside numpy/pandas
        # after the parent has touched BLAS/objc; load_all is ~0.5 s so re-loading
        # in each worker is cheap and keeps every worker identical.
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=load_all, initargs=(need_usg, need_all)) as ex:
            for name, out in ex.map(worker, jobs):
                _report(name, out)
    tag = args.tag or "_".join(variants if "combo" not in variants else variants[:-1] + ["combo-" + "-".join(combo)])
    out_path = LAB_DIR / f"{tag}.json"
    out_path.write_text(json.dumps(dict(
        protocol=dict(variants=variants, combo=combo, maxiter=args.maxiter, popsize=args.popsize,
                      polish=args.polish, boot=args.boot, seed=SEED, dev=list(DEV_SEASONS),
                      holdout_excluded=HOLDOUT_SEASON),
        results=results), indent=1))
    # summary table
    print(f"\n{'skill':<18}" + "".join(f"{v:>22}" for v in variants if v != "base"))
    for s in specs:
        row = f"{s:<18}"
        for v in variants:
            if v == "base":
                continue
            r = results[s].get(v, {})
            if "skipped" in r:
                row += f"{'-':>22}"
            else:
                g = r["vs_base"]
                row += f"{g['gain_pct']:+8.3f} [{g['ci95'][0]:+.3f},{g['ci95'][1]:+.3f}] {g['seasons_won']}/6".rjust(22)
        print(row)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
