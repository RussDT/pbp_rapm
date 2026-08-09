pbp_rapm DATA HANDOFF — 20260610
============================================

WHERE TO EXTRACT (from the pbp_rapm repo root, after cloning RussDT/pbp_rapm):
  tar xf pbp_rapm_data_handoff_20260610.tar
  -> nba_pipeline/raw_data/   (historical raw PBP, 1997-2013)
  -> nba_pipeline/processed/  (historical processed metric parquets, 1997-2013)
  -> rapms_master_results/    (alt3 EFG-value weighted-factor parquets;
       copy these into the RussDT/rapms repo's master_results/ — they are
       ALSO already committed in RussDT/rapms git, so cloning that repo works too)

NOT INCLUDED (still in progress): modern 2014-2025 raw + processed.
  Backfill is running; modern data ships once it completes.

CONTENTS:
  raw_data 1997-2013 : 34 files, 225M
  processed          : 612 parquets, 880M
  alt3 efg parquets  : 11 files, 4.8M
