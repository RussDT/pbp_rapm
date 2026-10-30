"""Step 1 of the dTS prior gate: split-half selection inside 2024-25.

`test_sq_defender_dts_prior.py` found that a compact outcome spec
(`rim_suppression` + `pts_allowed`) beats the prior-dTS incumbent on the only
available forward fold. That spec was chosen *after* reading the univariate
correlation table on that same fold, which makes it the best of eleven tried
rather than a pre-registered result. Its bootstrap interval barely excludes zero
and would not survive a multiplicity correction.

With no third SQ season on this API key, the cheapest way to convert post-hoc
selection into selection-then-confirmation is to split 2024-25 on its median
possession date and run a faithful miniature of the real fold:

    SELECT   H1 2024-25 features  ->  H2 2024-25 dTS   (this script)
    CONFIRM     2024-25 features  ->     2025-26 dTS   (test_sq_defender_dts_prior.py)

The miniature preserves the real fold's structure: features and the incumbent
prior both come from the earlier window, the target comes from a strictly later
one, and generalization is estimated by K-fold CV across players inside that
fixed split. Minimums are halved (500 def poss, 75 shots) to match the halved
exposure.

A spec that wins here and again on the confirm fold has been selected on data it
was not then evaluated on. A spec that wins only on the confirm fold is a
selection artifact. This script does not itself promote anything.

Prerequisites:
    python nba_pipeline/scripts/solve_ts_split_half.py 25
    python nba_pipeline/scripts/build_sq_defender_features.py features \
        --min-shots 60 --half h1 --cut-date 2025-01-25 \
        --out sq_defender_h1_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from test_sq_defender_dts_prior import (
    GEOM, EFFORT, OUTCOME, GEOM_C, OUT_C, evaluate, paired_boot, SEED)

RES = Path(__file__).resolve().parents[1] / 'results'
MIN_POSS = 500      # half of the full-season 1000
MIN_SHOTS = 75      # half of the full-season 150


def load():
    a = pd.read_csv(RES / 'ts_25_h1_results.csv')
    b = pd.read_csv(RES / 'ts_25_h2_results.csv')
    a = a[a['def_poss'] >= MIN_POSS][['player_id', 'def', 'def_poss']]
    a.columns = ['player_id', 'dts_prev', 'poss_prev']
    b = b[b['def_poss'] >= MIN_POSS][['player_id', 'def', 'def_poss']]
    b.columns = ['player_id', 'dts_next', 'poss_next']
    d = a.merge(b, on='player_id')
    d['dts_prev'] *= -1.0        # solver def is negative-is-better; flip
    d['dts_next'] *= -1.0

    f = pd.read_csv(RES / 'sq_defender_h1_features.csv')
    f = f[f['shots_defended'] >= MIN_SHOTS]
    d = d.merge(f.rename(columns={'nba_id': 'player_id'}), on='player_id', how='left')
    d['_sq'] = d['shots_defended'].notna()
    print(f'{len(d)} players with dTS in both halves (>= {MIN_POSS} def poss each)')
    print(f'{d["_sq"].sum()} with H1 SQ features (>= {MIN_SHOTS} shots defended)')
    return d[d['_sq']].reset_index(drop=True)


if __name__ == '__main__':
    d = load()
    print(f'\nselection fold: H1 2024-25 features -> H2 2024-25 dTS, n={len(d)}\n')
    print('out-of-sample (5-fold across players, weighted by H2 def possessions):')
    p_mean, _ = evaluate(d, [], 'league mean')
    p_base, _ = evaluate(d, ['dts_prev'], 'prior dTS (incumbent)')
    specs = [
        (['dts_prev'] + GEOM, 'prior dTS + SQ geometry', 'geometry '),
        (['dts_prev'] + EFFORT, 'prior dTS + effort', 'effort   '),
        (['dts_prev'] + OUTCOME, 'prior dTS + SQ allowed/suppression', 'outcome  '),
        (['dts_prev'] + GEOM + EFFORT + OUTCOME, 'prior dTS + all three arms', 'all three'),
        (['dts_prev'] + GEOM_C, 'prior dTS + compact geometry', 'geom cmpt'),
        (['dts_prev'] + OUT_C, 'prior dTS + compact outcome', 'out cmpt '),
        (['dts_prev', 'rim_suppression'], 'prior dTS + rim_suppression', 'rim_supp '),
    ]
    preds = []
    for cols, label, short in specs:
        p, _ = evaluate(d, cols, label)
        preds.append((p, short))

    print('\npaired bootstrap vs incumbent (negative = SQ helps):')
    for p, short in preds:
        paired_boot(d, p, p_base, short, 'prior dTS')
    print()
    paired_boot(d, p_base, p_mean, 'prior dTS', 'league mean')

    print('\nunivariate correlation with H2 dTS (weighted):')
    y, w = d['dts_next'].values, d['poss_next'].values.astype(float)
    rows = []
    for c in ['dts_prev'] + GEOM + EFFORT + OUTCOME:
        v = d[c].values.astype(float)
        m = ~np.isnan(v)
        if m.sum() < 50:
            continue
        vv = v[m] - np.average(v[m], weights=w[m])
        yy = y[m] - np.average(y[m], weights=w[m])
        r = np.average(vv * yy, weights=w[m]) / np.sqrt(
            np.average(vv ** 2, weights=w[m]) * np.average(yy ** 2, weights=w[m]))
        rows.append((c, r))
    for c, r in sorted(rows, key=lambda x: -abs(x[1]))[:12]:
        print(f'  {c:24s} {r:+.3f}')
