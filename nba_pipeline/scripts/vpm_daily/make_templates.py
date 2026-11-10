#!/usr/bin/env python3
"""Regenerate the nightly VPM templates from the Nov 3 code that made the live board.

Run only to re-derive the templates. Every template must render back to its
source byte for byte at the Nov 3 values; the one deliberate exception is the
season-year rule in cloud_refresh.py (see PATCHES).
"""
from pathlib import Path
import re

from render import NOV3, render

HERE = Path(__file__).resolve().parent
E = HERE.parents[1] / 'results/estimated_skills'
B = E / 'current_refresh_20261103_v1/pipeline_code'
SOURCES = {name: B / name for name in [
    'cloud_refresh.py', 'refresh_vpm_native_source.py', 'native_atom_adapter.py',
    'prepare_current_vpm_features.py', 'prepare_current_vpm_tracking.py', 'run_refresh.py']}
SOURCES['score_current_sq.py'] = E / 'current_sq_20261103_v1/score_current_sq.py'
SOURCES['emit_current_vpm_sq_refresh.py'] = E / 'current_refresh_20261103_sq_v1/emit_current_vpm_sq_refresh.py'
# The original only accepted autumn cuts (season == cut.year + 1).
PATCHES = {'cloud_refresh.py': [(
    "season != cut.year+1:", "season != cut.year+(cut.month>=7):")]}


def tokenize(text):
    text = text.replace(NOV3['SQ_RECEIPT_SHA'], '@@SQ_RECEIPT_SHA@@').replace(NOV3['SQ_EV_SHA'], '@@SQ_EV_SHA@@')
    text = re.sub(r'\b12846\b', '@@SQ_ROWS@@', text)
    text = text.replace('<20261103', '<@@CUT8@@').replace('20261103', '@@TAG@@')
    text = text.replace('2026-11-03', '@@CUT@@').replace('November3', '@@CUT_WORDS@@')
    text = text.replace("current_season_last_game='2026-11-02'", "current_season_last_game='@@LAST_GAME@@'")
    text = text.replace('2026-11-02', '@@FENCE@@')
    return re.sub(r'\b107\b', '@@GAMES@@', text)


for name, path in SOURCES.items():
    source = path.read_text()
    template = tokenize(source)
    if render(template, NOV3) != source:
        raise SystemExit('Template does not round-trip: ' + name)
    for old, new in PATCHES.get(name, []):
        if template.count(old) != 1:
            raise SystemExit('Patch seam differs: ' + name)
        template = template.replace(old, new)
    (HERE / 'templates' / (name + '.tmpl')).write_text(template)
    print(name, template.count('@@') // 2, 'tokens')
