"""Comparable per-miss continuation / opponent-premium accounting.

The opponent term only prices the observed immediate defensive-rebound branch.
This is descriptive downstream accounting, not total shot or full game value.
"""

def add_miss_balance(payload):
    reference = next(r for r in payload['groups']['All']['rows'] if r['category'] == 'Missed FG')
    own_reference = reference['own']
    opponent_reference = reference['paired_n'] / reference['misses'] * (reference['paired_ppp'] - payload['baseline'])
    payload['miss_balance_reference'] = {
        'own_points_per_miss': own_reference,
        'opponent_excess_per_miss': opponent_reference,
        'opponent_ppp': payload['baseline'],
        'misses': reference['misses'],
    }
    for group in payload['groups'].values():
        for row in group['rows']:
            if row['own'] is None or not row['misses']:
                continue
            q = row['paired_n'] / row['misses']
            if not 0 <= q <= 1:
                raise ValueError('Defensive-rebound branch outside miss denominator')
            cost = q * (row['paired_ppp'] - payload['baseline']) if row['paired_n'] else 0.0
            row['dreb_branch_fraction'] = q
            row['opponent_excess_per_miss'] = cost
            row['relative_own_continuation'] = row['own'] - own_reference
            row['relative_opponent_effect'] = opponent_reference - cost
            row['relative_miss_balance'] = row['relative_own_continuation'] + row['relative_opponent_effect']
    return payload
