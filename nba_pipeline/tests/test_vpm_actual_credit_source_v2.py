import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vpm_actual_credit_source_v2 as S
from test_vpm_actual_credit_source import source_game


def packet(actions):
    raw,d,q,slots,g=source_game(actions)
    raw=raw.drop(columns=['_possession_row'])
    reference=pd.DataFrame(dict(game_date=[pd.Timestamp('2025-11-01')],source_season_end_year=[2026],actual_points=[q.observed_points.sum()]))
    raw,native,slots=S.possession_game(raw,reference,q)
    return S.reconstruct_game(raw,native,q,slots)[0]


def test_unlabeled_retained_one_shot_penalty_adds_no_extra_possession():
    d=packet([(3,1,'P1 Free Throw 1 of 1 (1 PTS)','home',55),
              (1,2,'P2 Layup (2 PTS)','home',40)])
    assert d.actual_points.sum()==3
    assert d.possession_exposure.sum()==1
    assert d.loc[d.possession_exposure.eq(0),'ft_reference_weight'].sum()==0


def test_and_one_clock_drift_keeps_basket_and_ft_together():
    d=packet([(1,1,'P1 Layup (2 PTS)','home',55),
              (6,6,'P6 S.FOUL','away',55),
              (3,1,'P1 Free Throw 1 of 1 (3 PTS)','home',54)])
    assert len(d)==1
    assert d.possession_exposure.sum()==1
    assert d.actual_points.sum()==3


def test_retained_penalty_trip_uses_foul_context_for_both_attempts():
    d=packet([(6,6,'P6 FLAGRANT.FOUL','away',55),
              (3,1,'P1 Free Throw 1 of 2 (1 PTS)','home',55),
              (3,1,'P1 Free Throw 2 of 2 (2 PTS)','home',55),
              (1,2,'P2 Layup (2 PTS)','home',40)])
    assert d.possession_exposure.sum()==1
    assert d.loc[d.possession_exposure.eq(0),'ft_reference_weight'].sum()==0
    assert d.actual_points.sum()==4
