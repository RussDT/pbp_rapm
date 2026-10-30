-- Cover the latest-version foreign key used by current-result reads.
create index analytics_datasets_latest_version_idx
  on public.analytics_datasets (latest_version_id);
