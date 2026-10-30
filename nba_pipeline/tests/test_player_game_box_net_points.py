import numpy as np
import pandas as pd
import pytest
from nba_pipeline.scripts import build_player_game_box_net_points as b
from nba_pipeline.scripts.player_game_playtype import shrink_difficulty


def test_raw_count_terms_equal_per100_prediction_without_extra_100():
    model=dict(factor='oTS',features=['PTS','FGA'],coefficients=[.6,-.7],intercept=-2.)
    row=pd.Series({'PTS':30.,'FGA':20.})
    for exposure in [30.,70.,100.]:
        terms=b.score_terms(row,model,exposure)
        expected=(-2.+.6*30*100/exposure-.7*20*100/exposure)*exposure/100
        assert sum(x['points'] for x in terms)==pytest.approx(expected)
        assert terms[1]['points']==18.
        assert terms[2]['points']==-14.


def test_exported_coefficients_recover_known_unstandardized_equation():
    n=30
    frame=pd.DataFrame({'off_poss':np.linspace(500,5000,n),'PTS':np.linspace(40,1700,n)**1.1,'season':[2020]*n})
    frame['oTS']=1.25+.3*100*frame.PTS/frame.off_poss
    model=b.fit_linear(frame,'oTS',['PTS'],0.)
    assert model['intercept']==pytest.approx(1.25)
    assert model['coefficients'][0]==pytest.approx(.3)
    np.testing.assert_allclose(b.predict_rate(frame,model),frame.oTS)


def test_game_boxes_terms_and_final_reconciliation_close():
    base=pd.DataFrame({'game_id':['g','g'],'game_date':['2026-01-01']*2,'player_id':[1,2],'team':['A','A'],'opponent':['B','B'],'off_possessions':[40.,60.],'def_possessions':[39.,61.]})
    boxes=pd.DataFrame({'PLAYER_ID':[1,2],'PLAYER_NAME':['One','Two'],'TEAM_ABBREVIATION':['A','A'],'game_date':['2026-01-01']*2,'POSS':[40.,60.]})
    for c in b.BOX:
        boxes[c]=[1.,2.]
    boxes['MISSED_FG']=[0.,0.]
    target=pd.DataFrame({'game_id':['g'],'team':['A'],'points':[3.],**{f:[float(i)] for i,f in enumerate(b.FACTORS)}})
    models={f:dict(factor=f,features=['PTS'],coefficients=[.3],intercept=-1.) for f in b.FACTORS}
    players,terms,rec=b.score_games(base,target,boxes,models)
    np.testing.assert_allclose(rec.final_sum,rec.team_target)
    for row in players.itertuples():
        assert terms.loc[terms.player_id.eq(row.player_id),'points'].sum()==pytest.approx(row.game_net_points)
    assert players.box_net_points.sum()+players.team_adjustment.sum()==pytest.approx(players.game_net_points.sum())
    # No historical player or arbitrary event pricing field is required.
    assert 'oTS_prior_points' not in players
    boxes.loc[0,'PTS']=50.
    with pytest.raises(ValueError,match='sum to team score'):
        b.score_games(base,target,boxes,models)


def test_playtype_prior_only_shrinks_difficulty():
    value,weight=shrink_difficulty(8.,10.,2.,20.)
    assert weight==pytest.approx(1/3)
    assert value==pytest.approx(4.)
    raw_ts=70.;league_ts=58.
    assert raw_ts-league_ts+value==16.
    assert shrink_difficulty(8.,10.,2.,0.)==(8.,1.)
    assert shrink_difficulty(np.nan,0.,2.,20.)==(2.,0.)
    with pytest.raises(ValueError):
        shrink_difficulty(1.,10.,0.,-1.)


def test_offd_includes_disjoint_charge_and_other_offensive_fouls():
    from nba_pipeline.scripts.audit_player_game_five_year_coefficients import attach_offd
    box=pd.DataFrame({'PLAYER_ID':[1,2,3]})
    source=pd.DataFrame({'EntityId':[1,2],'Charge Fouls Drawn':[2.,np.nan],'Offensive Fouls Drawn':[3.,np.nan]})
    result=attach_offd(box,source).set_index('PLAYER_ID')
    assert result.loc[1,'OFFD']==5
    assert result.loc[2,'OFFD']==0
    assert result.loc[2,'offd_covered']
    assert np.isnan(result.loc[3,'OFFD'])
    assert not result.loc[3,'offd_covered']


def test_five_year_fit_recovers_coefficients_without_second_shrinkage():
    from nba_pipeline.scripts.audit_player_game_five_year_coefficients import fit_window
    rng=np.random.default_rng(42)
    frame=pd.DataFrame({'def_poss':rng.uniform(500,30000,100),'STL':rng.uniform(5,300,100),'OFFD':rng.uniform(0,50,100)})
    frame['dTOV']=-2+100*(1.2*frame.STL+.9*frame.OFFD)/frame.def_poss
    model=fit_window(frame,'dTOV',['STL','OFFD'])
    np.testing.assert_allclose(model['coefficients'],[1.2,.9])
    assert model['intercept']==pytest.approx(-2)
