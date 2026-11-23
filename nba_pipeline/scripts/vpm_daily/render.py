"""Fill the date, game-count and same-run pins in a nightly VPM template."""
import re

import pandas as pd

NOV3 = dict(TAG='20261103', CUT8='20261103', CUT='2026-11-03', CUT_WORDS='November3',
            FENCE='2026-11-02', LAST_GAME='2026-11-02', GAMES='107', SQ_ROWS='12846',
            SQ_RECEIPT_SHA='92cc585f4bd19fd36cb99664a8ab4bd62bc4303abc6568d0aa7994f137652211',
            SQ_EV_SHA='7b01981a89fdfb54390ac88dce15a30153869402d0b838809e7fa4404266fe3d')


def date_values(cut, tag=None):
    cut = pd.Timestamp(cut)
    return dict(TAG=tag or cut.strftime('%Y%m%d'), CUT8=cut.strftime('%Y%m%d'), CUT=str(cut.date()),
                CUT_WORDS=cut.strftime('%B')+str(cut.day), FENCE=str((cut-pd.Timedelta(days=1)).date()))


def render(template, values):
    out = re.sub(r'@@([A-Z0-9_]+)@@', lambda m: str(values[m.group(1)]), template)
    if '@@' in out:
        raise ValueError('Unfilled template token')
    return out
