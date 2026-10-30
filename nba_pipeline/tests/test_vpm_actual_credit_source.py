import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vpm_actual_credit_source as S
from vpm_net_lane_targets import targets


def source_game(actions):
    players=pd.DataFrame(dict(player_id=np.arange(1,11),team_id=[100]*5+[200]*5,
                              is_home=[True]*5+[False]*5,native_off_poss=[10]*10,
                              native_def_poss=[10]*10,observed_points=[0.]*10))
    rows=[]
    for j,(typ,pid,description,side,clock) in enumerate(actions):
        row={c:None for c in S.RAW_COLUMNS}
        row.update(game_id=22500111,event_num=j+1,event_type=typ,event_action_type=0,
                   period=1,minute_remaining_quarter=10,seconds_remaining_quarter=clock,
                   player1_id=pid,player1_name='P'+str(pid),player1_team_id=100 if pid<=5 else 200,
                   home_description=description if side=='home' else None,
                   visitor_description=description if side=='away' else None)
        row.update({f'home_player{i}':i for i in range(1,6)})
        row.update({f'away_player{i}':i+5 for i in range(1,6)})
        rows.append(row)
    raw=S.normalize(pd.DataFrame(rows));points=raw.groupby('player1_id').points.sum()
    players.observed_points=players.player_id.map(points).fillna(0.)
    reference=pd.DataFrame(dict(game_date=[pd.Timestamp('2025-11-01')],source_season_end_year=[2026],actual_points=[players.observed_points.sum()]))
    raw,native,slots=S.possession_game(raw,reference,players)
    atoms,counts=S.reconstruct_game(raw,native,players,slots)
    return raw,atoms,players,slots,counts


def test_and_one_and_retained_penalty_have_correct_score_and_exposure():
    raw,d,q,slots,g=source_game([(1,1,'P1 Layup (2 PTS)','home',55),
                                (3,6,'P6 Technical Free Throw 1 of 1 (1 PTS)','away',55),
                                (3,1,'P1 Free Throw 1 of 1 (3 PTS)','home',55),
                                (5,6,'P6 Turnover','away',40)])
    assert d.possession_exposure.sum()==2
    assert len(d)==3
    assert d.actual_points.sum()==4
    assert sorted(d.actual_points)==[0.,1.,3.]
    assert d.loc[d.possession_exposure.eq(0),'ft_reference_weight'].sum()==0


def test_rebound_chain_uses_actual_events_and_telescopes():
    raw,d,q,slots,g=source_game([(2,1,'MISS P1 3PT Jump Shot','home',55),
                                (4,2,'P2 REBOUND (Off:1 Def:0)','home',54),
                                (1,3,'P3 Layup (2 PTS)','home',50)])
    # Fixed accounting tariffs in this small worked example, no fitted slope.
    pricing=dict(net_zone_reference={'rim':1.6,'mid':1.1,'three':1.3},
                 kappa={'rim':.4,'mid':.3,'three':.25},
                 net_turnover_price=1.2,gross_fga_reference=1.1)
    events,checks=S.game_events(raw,d,q,slots,pricing)
    orb=events.loc[events.kind.eq('offensive_rebound_transition')].iloc[0]
    shot=events.loc[events.kind.eq('continuation_shot')].iloc[0]
    assert orb.player_id==2 and orb.value==.95
    assert shot.player_id==3 and shot.value==.8
    assert abs(events.loc[events.lane.eq('second_chance'),'value'].sum()-1.75)<1e-14
    assert checks.sc_state_gap.abs().max()<1e-14
    assert d.three_attempts.sum()==1 and d.rim_attempts.sum()==0
    assert d.Second_Chance_Diff.sum()==2


def test_defensive_rebound_has_only_second_chance_actor_credit():
    raw,d,q,slots,g=source_game([(2,1,'MISS P1 3PT Jump Shot','home',55),
                                (4,6,'P6 REBOUND (Off:0 Def:1)','away',54)])
    tariff=dict(net_zone_reference={'rim':1.6,'mid':1.1,'three':1.3},
                kappa={'rim':.4,'mid':.3,'three':.25},net_turnover_price=1.2,gross_fga_reference=1.1)
    ev,check=S.game_events(raw,d,q,slots,tariff)
    rebound=ev.loc[ev.player_id.eq(6)]
    assert rebound.lane.tolist()==['second_chance']
    assert rebound.value.tolist()==[.25]


def test_dead_ball_rebound_between_free_throws_does_not_create_possession():
    raw,d,q,s,g=source_game([(3,1,'MISS P1 Free Throw 1 of 2','home',55),
                            (4,200,'Away Team Rebound','away',55),
                            (3,1,'P1 Free Throw 2 of 2 (1 PTS)','home',55)])
    assert d.possession_exposure.sum()==1
    assert len(d)==1 and d.ft_points.iloc[0]==1
    assert d.ft_reference_weight.iloc[0]==1


def test_retained_flagrant_fts_do_not_erase_following_turnover():
    raw,d,q,s,g=source_game([(3,1,'P1 Free Throw Flagrant 1 of 2 (1 PTS)','home',55),
                            (3,1,'P1 Free Throw Flagrant 2 of 2 (2 PTS)','home',55),
                            (5,1,'P1 Turnover','home',50)])
    assert d.possession_exposure.sum()==1
    assert d.Is_Turnover.sum()==1
    assert d.loc[d.Is_Turnover.eq(1),'actual_points'].sum()==0
    assert d.actual_points.sum()==2


def test_personal_scoring_mismatch_blocks_game():
    raw,d,q,s,g=source_game([(1,1,'P1 Layup (2 PTS)','home',55)])
    q.loc[q.player_id.eq(1),'observed_points']=3
    reference=pd.DataFrame(dict(game_date=[pd.Timestamp('2025-11-01')],source_season_end_year=[2026],actual_points=[2.]))
    import pytest
    with pytest.raises(ValueError,match='personal scoring'):
        S.possession_game(raw,reference,q)


def test_turnover_after_missed_ft_does_not_charge_second_first_chance_price():
    raw,d,q,s,g=source_game([(3,1,'MISS P1 Free Throw 1 of 2','home',55),
                            (3,1,'MISS P1 Free Throw 2 of 2','home',55),
                            (4,2,'P2 REBOUND (Off:1 Def:0)','home',54),
                            (5,2,'P2 Turnover','home',50)])
    assert d.possession_exposure.sum()==1
    assert d.Is_Turnover.sum()==0
    assert d.ft_reference_weight.sum()==1
