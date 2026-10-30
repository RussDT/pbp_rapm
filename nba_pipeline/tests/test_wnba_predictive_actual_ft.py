import numpy as np
import pandas as pd
from wnba_test import clean_decomp_wnba as clean


def test_actual_ft_path_never_loads_full_season_percentages(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('full-season FT lookup')
    monkeypatch.setattr(clean,'_attach_wnba_ft_pct',forbidden)
    frame=pd.DataFrame({'home_description':['A Free Throw 1 of 2','MISS A Free Throw 2 of 2','A 3PT Jump Shot (3 PTS)'], 'visitor_description':['','',''], 'event_type':['FreeThrow','FreeThrow','MAKE'], 'Home_Action_Score':[1,0,3], 'Away_Action_Score':[0,0,0]})
    result=clean._attach_clean_scoring(frame,'test',2024,False,actual_fts=True)
    np.testing.assert_array_equal(result.ActualNet,[1.,0.,3.])
    np.testing.assert_array_equal(result.ExpFT,[1.,0.,0.])


def test_legacy_expected_ft_default_is_preserved(monkeypatch):
    monkeypatch.setattr(clean,'_attach_wnba_ft_pct',lambda df,*a:df.assign(FTPerc=.8))
    frame=pd.DataFrame({'home_description':['MISS A Free Throw 1 of 2'],'visitor_description':[''],'event_type':['FreeThrow'],'Home_Action_Score':[0],'Away_Action_Score':[0]})
    result=clean._attach_clean_scoring(frame,'test',2024,False)
    assert result.ActualNet.iloc[0]==.8
