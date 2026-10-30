-- Versioned, RLS-protected publication surface for curated analytics results.
create table public.analytics_datasets (
  slug text primary key,
  title text not null,
  description text not null,
  status text not null,
  is_public boolean not null default false,
  schema_version integer not null default 1,
  row_key_fields text[] not null,
  source_path text not null,
  latest_version_id uuid,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint analytics_datasets_slug_format
    check (slug ~ '^[a-z0-9]+(?:_[a-z0-9]+)*$'),
  constraint analytics_datasets_status
    check (status in ('production', 'research', 'diagnostic', 'archived')),
  constraint analytics_datasets_schema_version_positive
    check (schema_version > 0),
  constraint analytics_datasets_row_key_fields_nonempty
    check (cardinality(row_key_fields) > 0)
);

create table public.analytics_dataset_versions (
  id uuid primary key,
  dataset_slug text not null
    references public.analytics_datasets (slug) on delete cascade,
  content_sha256 text not null,
  as_of_date date,
  row_count bigint not null,
  column_names text[] not null,
  source_path text not null,
  metadata jsonb not null default '{}'::jsonb,
  published_at timestamptz not null default now(),
  constraint analytics_dataset_versions_content_sha256_format
    check (content_sha256 ~ '^[0-9a-f]{64}$'),
  constraint analytics_dataset_versions_row_count_nonnegative
    check (row_count >= 0),
  constraint analytics_dataset_versions_columns_nonempty
    check (cardinality(column_names) > 0),
  constraint analytics_dataset_versions_dataset_hash_unique
    unique (dataset_slug, content_sha256)
);

alter table public.analytics_datasets
  add constraint analytics_datasets_latest_version_fk
  foreign key (latest_version_id)
  references public.analytics_dataset_versions (id)
  on delete set null;

create table public.analytics_dataset_rows (
  version_id uuid not null
    references public.analytics_dataset_versions (id) on delete cascade,
  row_key text not null,
  row_ordinal integer not null,
  payload jsonb not null,
  primary key (version_id, row_key),
  constraint analytics_dataset_rows_row_key_nonempty
    check (length(row_key) > 0),
  constraint analytics_dataset_rows_ordinal_nonnegative
    check (row_ordinal >= 0),
  constraint analytics_dataset_rows_payload_object
    check (jsonb_typeof(payload) = 'object')
);

create index analytics_dataset_versions_dataset_published_idx
  on public.analytics_dataset_versions (dataset_slug, published_at desc);

create index analytics_dataset_rows_version_ordinal_idx
  on public.analytics_dataset_rows (version_id, row_ordinal);

alter table public.analytics_datasets enable row level security;
alter table public.analytics_dataset_versions enable row level security;
alter table public.analytics_dataset_rows enable row level security;

revoke all on table public.analytics_datasets from anon, authenticated;
revoke all on table public.analytics_dataset_versions from anon, authenticated;
revoke all on table public.analytics_dataset_rows from anon, authenticated;

grant select on table public.analytics_datasets to anon, authenticated;
grant select on table public.analytics_dataset_versions to anon, authenticated;
grant select on table public.analytics_dataset_rows to anon, authenticated;

grant select, insert, update, delete on table public.analytics_datasets to service_role;
grant select, insert, update, delete on table public.analytics_dataset_versions to service_role;
grant select, insert, update, delete on table public.analytics_dataset_rows to service_role;

create policy "Public analytics datasets are readable"
  on public.analytics_datasets
  for select
  to anon, authenticated
  using (is_public);

create policy "Public analytics dataset versions are readable"
  on public.analytics_dataset_versions
  for select
  to anon, authenticated
  using (
    exists (
      select 1
      from public.analytics_datasets datasets
      where datasets.slug = analytics_dataset_versions.dataset_slug
        and datasets.is_public
    )
  );

create policy "Public analytics dataset rows are readable"
  on public.analytics_dataset_rows
  for select
  to anon, authenticated
  using (
    exists (
      select 1
      from public.analytics_dataset_versions versions
      join public.analytics_datasets datasets
        on datasets.slug = versions.dataset_slug
      where versions.id = analytics_dataset_rows.version_id
        and datasets.is_public
    )
  );

create view public.analytics_current_rows
with (security_invoker = true)
as
select
  datasets.slug as dataset_slug,
  datasets.title,
  datasets.status,
  versions.id as version_id,
  versions.as_of_date,
  versions.content_sha256,
  rows.row_key,
  rows.row_ordinal,
  rows.payload
from public.analytics_datasets datasets
join public.analytics_dataset_versions versions
  on versions.id = datasets.latest_version_id
join public.analytics_dataset_rows rows
  on rows.version_id = versions.id;

revoke all on table public.analytics_current_rows from anon, authenticated;
grant select on table public.analytics_current_rows to anon, authenticated;
grant select on table public.analytics_current_rows to service_role;

create function public.promote_analytics_dataset_version(
  p_dataset_slug text,
  p_version_id uuid,
  p_expected_rows bigint
)
returns void
language plpgsql
security invoker
set search_path = ''
as $$
declare
  actual_rows bigint;
begin
  if not exists (
    select 1
    from public.analytics_dataset_versions versions
    where versions.id = p_version_id
      and versions.dataset_slug = p_dataset_slug
  ) then
    raise exception 'version % does not belong to dataset %', p_version_id, p_dataset_slug;
  end if;

  select count(*)
  into actual_rows
  from public.analytics_dataset_rows rows
  where rows.version_id = p_version_id;

  if actual_rows <> p_expected_rows then
    raise exception 'dataset % expected % rows but found %',
      p_dataset_slug, p_expected_rows, actual_rows;
  end if;

  update public.analytics_datasets
  set latest_version_id = p_version_id,
      updated_at = now()
  where slug = p_dataset_slug;

  if not found then
    raise exception 'dataset % does not exist', p_dataset_slug;
  end if;
end;
$$;

revoke all on function public.promote_analytics_dataset_version(text, uuid, bigint)
  from public, anon, authenticated;
grant execute on function public.promote_analytics_dataset_version(text, uuid, bigint)
  to service_role;

comment on table public.analytics_datasets is
  'Stable registry for versioned analytics result datasets published from the RAPM workspace.';
comment on table public.analytics_dataset_versions is
  'Immutable content-addressed publication receipts for analytics datasets.';
comment on table public.analytics_dataset_rows is
  'Versioned result rows stored as JSON objects; query analytics_current_rows for each dataset latest version.';
comment on view public.analytics_current_rows is
  'Site-facing latest-version rows for public analytics datasets; underlying RLS is enforced with security_invoker.';
