"""Predictor-only, training-prefix positional rim-quality decomposition.

This is optional feature plumbing, not an impact model or a position adjustment
to ratings. Keep the raw feature control. Its physical identity is
``saved100 = 2 * contests100 * suppression``: contest volume is per 100 team
defensive possessions, whereas suppression is saved makes per rim contest.

The role baseline is fitted to each player's latest pre-fit-cut predictive
packet, using its past rim-attempt support. Earlier query rows remain hash-bound
for training transforms but have zero baseline-fit weight: adding query dates
must not count overlapping cumulative history again. These are weights on
snapshot predictors, not independent shot counts. The caller must freeze
``pooling_attempts`` explicitly; this module neither chooses it nor reads future
target exposure/outcomes. The frozen role archive's first availability metadata
is2017-04-13; its strict availability<query rule first permits2017-04-14 queries.
Historical position publication vintages remain uncertified.

Missing roles, missing quality history and inactive early-era baselines yield
missing role coordinates and a disjoint raw fallback. They never erase or
impute the raw control. Final
ratings/position means are outside this module. Calibration may call training
mode only on an exact subset of the fitted predictor rows; deployment requires
an estimate date on/after the baseline fit cut.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


RECIPE = 'training_prefix_role_rim_quality_decomposition_v1'
REGISTERED_POOLING_ATTEMPTS = 500.0  # Unoptimized engineering constant, rim-contest units.
DEFAULT_COLUMNS = dict(
    player='player_id', date='estimate_date', saved='own_rim_saved100',
    volume='own_rim_contests100', quality='own_rim_suppression',
    exposure='own_rim_suppression__prior_exposure',
    last_observation='rim_last_observation_date',
    role_available='role_available_at', role_last='role_source_last_date',
    roles=['role_guard', 'role_wing', 'role_big'])
OUTPUT_COLUMNS = ['rim_saved100_raw_control', 'rim_role_quality_baseline_saved100',
                  'rim_position_relative_quality_saved100',
                  'rim_saved100_raw_when_role_unknown']
CANDIDATE_COLUMNS = OUTPUT_COLUMNS[1:]


def _cut(value):
    date = pd.Timestamp(value)
    if pd.isna(date) or date.tz is not None or date != date.normalize():
        raise ValueError('A finite naive whole query date is required')
    if not pd.Timestamp('1996-11-01') <= date <= pd.Timestamp('2026-10-01'):
        raise ValueError('Query outside admitted ending1997–2026 history')
    return date


def _columns(columns):
    result = dict(DEFAULT_COLUMNS if columns is None else columns)
    if set(result) != set(DEFAULT_COLUMNS) or len(result['roles']) != 3:
        raise ValueError('Explicit complete physical-column contract required')
    names = [v for k, v in result.items() if k != 'roles'] + list(result['roles'])
    if len(set(names)) != len(names):
        raise ValueError('Physical-column aliases must be distinct')
    return result


def _packet(frame, columns):
    """Validate supplied past-state provenance; never invent role/history."""
    c = columns
    names = [v for k, v in c.items() if k != 'roles'] + list(c['roles'])
    d = frame.loc[:, names].copy()
    numeric_id = pd.to_numeric(d[c['player']], errors='raise').to_numpy(float)
    if (not np.isfinite(numeric_id).all() or (numeric_id <= 0).any()
            or (numeric_id != np.floor(numeric_id)).any()
            or (numeric_id > np.iinfo(np.int64).max).any()):
        raise ValueError('Exact positive canonical player IDs required')
    d[c['player']] = numeric_id.astype(np.int64)
    d[c['date']] = pd.to_datetime(d[c['date']], errors='raise')
    if any(_cut(value) != value for value in d[c['date']]):
        raise ValueError('Invalid query date')
    if d.duplicated([c['player'], c['date']]).any():
        raise ValueError('Duplicate player/query predictor row')
    for name in [c['last_observation'], c['role_available'], c['role_last']]:
        d[name] = pd.to_datetime(d[name], errors='raise')
        dated = d[name].notna()
        if (d.loc[dated, name] >= d.loc[dated, c['date']]).any():
            raise ValueError('Same-day/future predictor provenance: ' + name)
    q = d[c['roles']].to_numpy(float)
    known = np.isfinite(q).all(axis=1)
    if np.isinf(q).any() or (np.isfinite(q).any(axis=1) & ~known).any():
        raise ValueError('Role shares must be all observed or all missing')
    if ((q[known] < 0).any() or (q[known] > 1).any()
            or not np.allclose(q[known].sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError('Observed role shares must sum to one')
    if (known & (d[c['role_available']].isna() | d[c['role_last']].isna())).any():
        raise ValueError('Observed role requires dated source availability')
    if ((d[c['role_last']] > d[c['role_available']])
            & d[c['role_available']].notna()).any():
        raise ValueError('Role source ends after stated availability')
    if (known & d[c['role_last']].lt(d[c['date']]-pd.Timedelta(days=730))).any():
        raise ValueError('Role source exceeds the admitted730-day staleness cap')
    x = d[[c['saved'], c['volume'], c['quality'], c['exposure']]].to_numpy(float)
    if np.isinf(x).any() or (x[np.isfinite(x[:, 1]), 1] < 0).any():
        raise ValueError('Invalid raw predictive rim values')
    if (x[np.isfinite(x[:, 3]), 3] < 0).any():
        raise ValueError('Negative predictive rim-attempt support')
    if (np.abs(x[np.isfinite(x[:, 2]), 2]) > 1).any():
        raise ValueError('Suppression must be saved makes per contest')
    complete = np.isfinite(x).all(axis=1)
    if not np.allclose(x[complete, 0], 2*x[complete, 1]*x[complete, 2],
                       rtol=1e-12, atol=1e-12):
        raise ValueError('Saved100/volume/quality physical identity mismatch')
    supported = complete & (x[:, 3] > 0) & d[c['last_observation']].notna().to_numpy()
    return d, q, known, x, supported


def _row_hashes(frame, columns):
    """Bind training subsets to the exact supplied predictor/provenance rows."""
    c = columns
    result = {}
    keys = frame[[c['player'], c['date']]].itertuples(index=False, name=None)
    for (player, date), row in zip(keys, frame.itertuples(index=False, name=None)):
        values = []
        for value in row:
            if pd.isna(value):
                values.append(None)
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                values.append(str(pd.Timestamp(value).date()))
            elif isinstance(value, (np.integer, int)):
                values.append(int(value))
            elif isinstance(value, (float, np.floating)):
                values.append(float(value))
            else:
                values.append(str(value))
        key = str(int(player)) + '|' + str(pd.Timestamp(date).date())
        result[key] = hashlib.sha256(json.dumps(values, separators=(',', ':'),
                                               allow_nan=False).encode()).hexdigest()
    return result


def fit_positional_baseline(training_frame, fit_cut, *, pooling_attempts, columns=None):
    """Learn shrinkage means from predictors strictly before the fit cut.

    ``pooling_attempts`` is an explicit fixed research choice. Only each player's
    latest prefix packet can receive positive weight; no backward search for a
    more convenient earlier nonmissing packet occurs. No target,
    future-side possession eligibility, final rating or current cohort mean is
    consulted. Rows at/after fit_cut are discarded before predictor validation,
    except the protected2027 boundary is rejected for the entire supplied set.
    """
    c = _columns(columns); cut = _cut(fit_cut)
    if not np.isfinite(pooling_attempts) or pooling_attempts <= 0:
        raise ValueError('Freeze a positive finite pooling_attempts explicitly')
    dates = pd.to_datetime(training_frame[c['date']], errors='raise')
    if dates.isna().any() or (dates.dt.year > 2026).any():
        raise ValueError('Invalid/protected training query date')
    prefix = training_frame.loc[dates.lt(cut)].reset_index(drop=True)
    d, q, known, x, supported = _packet(prefix, c)
    # Sort normalized identities/dates, not their possible string spellings.
    order = np.lexsort((d[c['date']].to_numpy(), d[c['player']].to_numpy()))
    d = d.iloc[order].reset_index(drop=True)
    q, known, x, supported = q[order], known[order], x[order], supported[order]
    latest = ~d[c['player']].duplicated(keep='last').to_numpy()
    fit_supported = supported & latest
    w = np.where(fit_supported, x[:, 3], 0.)
    # Compact summation makes parameters invariant even to floating-point
    # reduction regrouping when extra zero-weight earlier queries are added.
    mass = float(w[fit_supported].sum())
    pooled = float(np.dot(w[fit_supported], x[fit_supported, 2])/mass) if mass > 0 else None
    role_mass = np.zeros(3); means = np.full(3, np.nan)
    if pooled is not None:
        for g in range(3):
            take = fit_supported & known
            wg = w[take]*q[take, g]
            role_mass[g] = wg.sum()
            means[g] = (np.dot(wg, x[take, 2]) + pooling_attempts*pooled)/(
                role_mass[g] + pooling_attempts)
    active = bool(role_mass.sum() > 0)
    return dict(recipe=RECIPE, fit_cut=str(cut.date()), columns=c,
                pooling_attempts=float(pooling_attempts),
                weight_definition='past predictive rim-attempt support at latest pre-fit-cut packet perplayer only; earlier query packets have zero fitweight; not independent unique shots',
                pooled_suppression=pooled,
                role_mean_suppression=means.tolist() if pooled is not None else [None]*3,
                role_support=role_mass.tolist(), active=active,
                inactive_reason=None if active else 'no_observed_position_training_support',
                training_rows=len(d), quality_supported_training_rows=int(supported.sum()),
                latest_player_packets=int(latest.sum()),
                quality_supported_fit_players=int(fit_supported.sum()),
                supported_fit_attempt_mass=mass,
                known_role_training_rows=int((fit_supported & known).sum()),
                last_training_query=None if d.empty else str(d[c['date']].max().date()),
                training_row_hashes=_row_hashes(d, c),
                candidate_columns=list(CANDIDATE_COLUMNS),
                raw_control_column=OUTPUT_COLUMNS[0],
                candidate_rule='replace raw with baseline+relative on supported roles and disjoint raw fallback elsewhere; retain contest volume; raw control not an additional fitted coordinate',
                mapper_admission_rule='use original raw basis wholesale until matured mapper admits the role-decomposition block; same rule at calibration and deployment',
                uses_outcomes=False, uses_future_lineup_exposure=False,
                modifies_ratings=False, source_publication_vintages_certified=False)


def transform_positional_baseline(packet, model, *, mode='deployment'):
    """Return raw control and an exact role-quality split, without imputation."""
    if model['recipe'] != RECIPE or mode not in ['training', 'deployment']:
        raise ValueError('Unknown model/transform mode')
    c = _columns(model['columns']); cut = _cut(model['fit_cut'])
    d, q, known, x, supported = _packet(packet, c)
    if mode == 'deployment' and d[c['date']].lt(cut).any():
        raise ValueError('Baseline fitted after deployment query')
    if mode == 'training':
        if not d[c['date']].lt(cut).all():
            raise ValueError('Training rows must precede fit cut')
        hashes = _row_hashes(d, c)
        if any(model['training_row_hashes'].get(k) != h for k, h in hashes.items()):
            raise ValueError('Training transform requires exact fitted predictor rows')
    out = d[[c['player'], c['date']]].copy()
    out['rim_saved100_raw_control'] = x[:, 0]
    out['rim_role_quality_baseline_saved100'] = np.nan
    out['rim_position_relative_quality_saved100'] = np.nan
    out['rim_saved100_raw_when_role_unknown'] = np.nan
    usable = supported & known & bool(model['active'])
    if usable.any():
        baseline = 2*x[usable, 1]*(q[usable]@np.asarray(model['role_mean_suppression']))
        out.loc[usable, 'rim_role_quality_baseline_saved100'] = baseline
        out.loc[usable, 'rim_position_relative_quality_saved100'] = x[usable, 0]-baseline
    # A computationally disjoint fallback, not an invented observed role zero.
    # An observed raw zero remains zero; unknown raw measurements remain NaN.
    out.loc[~usable, 'rim_saved100_raw_when_role_unknown'] = x[~usable, 0]
    out['position_relative_supported'] = usable
    out['position_known'] = known
    return out


def mapper_rim_basis(audit_packet, *, role_decomposition_admitted):
    """Apply one model-aware block rule at both fit and deployment.

    The caller determines admission from the matured mapper's serialized
    feature schema, never raw tracking/role arrival. This function does not
    infer admission from the currently queried players.
    """
    if not isinstance(role_decomposition_admitted, (bool, np.bool_)):
        raise ValueError('Explicit matured-mapper block admission required')
    names = CANDIDATE_COLUMNS if role_decomposition_admitted else OUTPUT_COLUMNS[:1]
    return audit_packet.loc[:, names].copy()
