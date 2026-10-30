create table if not exists public.nba_no_prior_vpm_td550_snapshots (
    player_id bigint not null,
    player_name text not null,
    season_end_year smallint not null check (season_end_year between 2020 and 2026),
    snapshot_date date not null,
    snapshot_year smallint not null,
    snapshot_month smallint not null check (snapshot_month in (1, 5, 8, 10)),
    checkpoint text not null check (checkpoint in ('oct01', 'jan01', 'may01', 'aug30')),
    pricing text not null,
    prior_scale double precision not null check (prior_scale = 0),
    time_decay_half_life_days smallint not null default 550 check (time_decay_half_life_days = 550),
    rubberband_model text not null default 'smooth_clock_aware' check (rubberband_model = 'smooth_clock_aware'),
    season_effects text not null default 'rs_ps' check (season_effects = 'rs_ps'),
    component_alphas jsonb not null,
    possessions double precision not null check (possessions >= 0),
    off_poss double precision not null check (off_poss >= 0),
    def_poss double precision not null check (def_poss >= 0),
    o_decomp_net_rim_freq double precision not null,
    d_decomp_net_rim_freq double precision not null,
    o_decomp_net_rim_fg double precision not null,
    d_decomp_net_rim_fg double precision not null,
    o_decomp_net_mid_fg double precision not null,
    d_decomp_net_mid_fg double precision not null,
    o_decomp_net_three_freq double precision not null,
    d_decomp_net_three_freq double precision not null,
    o_decomp_net_three_fg double precision not null,
    d_decomp_net_three_fg double precision not null,
    o_alt_ft double precision not null,
    d_alt_ft double precision not null,
    o_alt_tov_net_value double precision not null,
    d_alt_tov_net_value double precision not null,
    o_decomp_net_mid_value double precision not null,
    d_decomp_net_mid_value double precision not null,
    o_decomp_net_efg double precision not null,
    d_decomp_net_efg double precision not null,
    o_second_chance_clean double precision not null,
    d_second_chance_clean double precision not null,
    vpm_off double precision not null,
    vpm_def double precision not null,
    vpm_net double precision not null,
    ots double precision not null,
    otov double precision not null,
    osc double precision not null,
    dts double precision not null,
    dtov double precision not null,
    dsc double precision not null,
    published_at timestamptz not null default now(),
    primary key (player_id, snapshot_date),
    check (snapshot_year = extract(year from snapshot_date)),
    check (snapshot_month = extract(month from snapshot_date)),
    check (
        (checkpoint = 'oct01' and extract(month from snapshot_date) = 10 and extract(day from snapshot_date) = 1 and season_end_year = extract(year from snapshot_date) + 1)
        or (checkpoint = 'jan01' and extract(month from snapshot_date) = 1 and extract(day from snapshot_date) = 1 and season_end_year = extract(year from snapshot_date))
        or (checkpoint = 'may01' and extract(month from snapshot_date) = 5 and extract(day from snapshot_date) = 1 and season_end_year = extract(year from snapshot_date))
        or (checkpoint = 'aug30' and extract(month from snapshot_date) = 8 and extract(day from snapshot_date) = 30 and season_end_year = extract(year from snapshot_date))
    )
);

comment on table public.nba_no_prior_vpm_td550_snapshots is
    'Research-only no-prior net-continuation VPM: TD550, smooth clock-aware rubberband, RS/PS season effects, frozen component alphas; four calendar snapshots per season-end year.';

create index if not exists nba_no_prior_vpm_td550_player_date_idx
    on public.nba_no_prior_vpm_td550_snapshots (player_id, snapshot_date desc);
create index if not exists nba_no_prior_vpm_td550_player_name_date_idx
    on public.nba_no_prior_vpm_td550_snapshots (lower(player_name), snapshot_date desc);
create index if not exists nba_no_prior_vpm_td550_calendar_idx
    on public.nba_no_prior_vpm_td550_snapshots (snapshot_year, snapshot_month, vpm_net desc);
create index if not exists nba_no_prior_vpm_td550_season_checkpoint_idx
    on public.nba_no_prior_vpm_td550_snapshots (season_end_year, checkpoint, vpm_net desc);

alter table public.nba_no_prior_vpm_td550_snapshots enable row level security;

revoke all on table public.nba_no_prior_vpm_td550_snapshots from anon, authenticated;
grant select on table public.nba_no_prior_vpm_td550_snapshots to anon, authenticated;
grant all on table public.nba_no_prior_vpm_td550_snapshots to service_role;

create policy "Public read no-prior VPM snapshots"
    on public.nba_no_prior_vpm_td550_snapshots
    for select
    to anon, authenticated
    using (true);
