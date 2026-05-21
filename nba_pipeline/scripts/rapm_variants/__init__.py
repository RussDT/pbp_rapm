"""
RAPM Variants Package

Contains multiple RAPM implementations with forward-looking backtesting:
- baseline: Standard ridge (lambda_off=3000 fixed, tune lambda_def)
- timedecay: Time-decay row weights
- possridge: Possession-weighted ridge penalties
- timedecay_possridge: Combined time-decay + possession-weighted ridge
"""

from .base_rapm import (
    load_rapm_data,
    prepare_data,
    build_design_matrix,
    get_player_possessions_fast,
    LAMBDA_OFF_BASE,
    LAMBDA_DEF_GRID,
    BACKTEST_FOLDS,
    RESULTS_DIR,
)
