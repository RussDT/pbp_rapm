#!/usr/bin/env python3
"""Build a standalone, metadata-only Supabase database catalog.

The script uses the authenticated Supabase CLI and a temporary linked project.
It never reads table records, auth identities, storage object names, or secrets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "templates" / "supabase_catalog.html"
DEFAULT_HTML = PIPELINE_ROOT / "docs" / "supabase_database_catalog.html"
DEFAULT_JSON = PIPELINE_ROOT / "docs" / "supabase_database_catalog.json"


METADATA_SQL = r"""
with relation_base as (
  select
    n.nspname as schema_name,
    c.relname as relation_name,
    c.oid as relation_oid,
    c.relkind,
    c.reltuples,
    c.relrowsecurity as rls_enabled,
    c.relforcerowsecurity as rls_forced,
    obj_description(c.oid, 'pg_class') as comment,
    case when c.relkind in ('r', 'p', 'm') then pg_total_relation_size(c.oid) else 0 end as total_bytes,
    case when c.relkind in ('r', 'p', 'm') then pg_relation_size(c.oid) else 0 end as table_bytes,
    case when c.relkind in ('r', 'p', 'm') then pg_indexes_size(c.oid) else 0 end as index_bytes,
    stats.n_live_tup,
    stats.n_dead_tup,
    stats.last_analyze,
    stats.last_autoanalyze,
    stats.last_vacuum,
    stats.last_autovacuum,
    parent_ns.nspname || '.' || parent.relname as partition_parent
  from pg_class c
  join pg_namespace n on n.oid = c.relnamespace
  left join pg_stat_all_tables stats on stats.relid = c.oid
  left join pg_inherits inh on inh.inhrelid = c.oid
  left join pg_class parent on parent.oid = inh.inhparent
  left join pg_namespace parent_ns on parent_ns.oid = parent.relnamespace
  where c.relkind in ('r', 'p', 'v', 'm', 'f')
    and n.nspname not like 'pg_%'
    and n.nspname <> 'information_schema'
),
relation_rows as (
  select
    'relation'::text as object_type,
    rb.schema_name || '.' || rb.relation_name as sort_key,
    jsonb_build_object(
      'schema', rb.schema_name,
      'name', rb.relation_name,
      'qualified_name', rb.schema_name || '.' || rb.relation_name,
      'kind', case rb.relkind
        when 'r' then 'table'
        when 'p' then 'partitioned table'
        when 'v' then 'view'
        when 'm' then 'materialized view'
        when 'f' then 'foreign table'
      end,
      'estimated_rows', greatest(coalesce(rb.n_live_tup, rb.reltuples::bigint, 0), 0),
      'dead_rows', coalesce(rb.n_dead_tup, 0),
      'total_bytes', rb.total_bytes,
      'table_bytes', rb.table_bytes,
      'index_bytes', rb.index_bytes,
      'rls_enabled', rb.rls_enabled,
      'rls_forced', rb.rls_forced,
      'comment', rb.comment,
      'partition_parent', rb.partition_parent,
      'last_analyze', coalesce(rb.last_autoanalyze, rb.last_analyze),
      'last_vacuum', coalesce(rb.last_autovacuum, rb.last_vacuum),
      'view_definition', case when rb.relkind in ('v', 'm') then pg_get_viewdef(rb.relation_oid, true) end,
      'columns', coalesce((
        select jsonb_agg(jsonb_build_object(
          'position', a.attnum,
          'name', a.attname,
          'type', format_type(a.atttypid, a.atttypmod),
          'nullable', not a.attnotnull,
          'default', pg_get_expr(ad.adbin, ad.adrelid),
          'identity', nullif(a.attidentity, ''),
          'generated', nullif(a.attgenerated, ''),
          'comment', col_description(a.attrelid, a.attnum)
        ) order by a.attnum)
        from pg_attribute a
        left join pg_attrdef ad on ad.adrelid = a.attrelid and ad.adnum = a.attnum
        where a.attrelid = rb.relation_oid and a.attnum > 0 and not a.attisdropped
      ), '[]'::jsonb),
      'constraints', coalesce((
        select jsonb_agg(jsonb_build_object(
          'name', con.conname,
          'type', case con.contype
            when 'p' then 'primary key'
            when 'f' then 'foreign key'
            when 'u' then 'unique'
            when 'c' then 'check'
            when 'x' then 'exclusion'
            else con.contype::text
          end,
          'definition', pg_get_constraintdef(con.oid, true),
          'columns', coalesce((
            select jsonb_agg(att.attname order by key_cols.ordinality)
            from unnest(con.conkey) with ordinality as key_cols(attnum, ordinality)
            join pg_attribute att on att.attrelid = con.conrelid and att.attnum = key_cols.attnum
          ), '[]'::jsonb),
          'references', case when con.contype = 'f' then ref_ns.nspname || '.' || ref.relname end,
          'reference_columns', case when con.contype = 'f' then coalesce((
            select jsonb_agg(att.attname order by key_cols.ordinality)
            from unnest(con.confkey) with ordinality as key_cols(attnum, ordinality)
            join pg_attribute att on att.attrelid = con.confrelid and att.attnum = key_cols.attnum
          ), '[]'::jsonb) end
        ) order by con.contype, con.conname)
        from pg_constraint con
        left join pg_class ref on ref.oid = con.confrelid
        left join pg_namespace ref_ns on ref_ns.oid = ref.relnamespace
        where con.conrelid = rb.relation_oid
      ), '[]'::jsonb),
      'indexes', coalesce((
        select jsonb_agg(jsonb_build_object(
          'name', idx.relname,
          'primary', ind.indisprimary,
          'unique', ind.indisunique,
          'valid', ind.indisvalid,
          'definition', pg_get_indexdef(ind.indexrelid),
          'bytes', pg_relation_size(ind.indexrelid)
        ) order by ind.indisprimary desc, ind.indisunique desc, idx.relname)
        from pg_index ind
        join pg_class idx on idx.oid = ind.indexrelid
        where ind.indrelid = rb.relation_oid
      ), '[]'::jsonb),
      'policies', coalesce((
        select jsonb_agg(jsonb_build_object(
          'name', pol.policyname,
          'permissive', pol.permissive,
          'roles', pol.roles,
          'command', pol.cmd,
          'using', pol.qual,
          'with_check', pol.with_check
        ) order by pol.policyname)
        from pg_policies pol
        where pol.schemaname = rb.schema_name and pol.tablename = rb.relation_name
      ), '[]'::jsonb),
      'triggers', coalesce((
        select jsonb_agg(jsonb_build_object(
          'name', tg.tgname,
          'enabled', tg.tgenabled,
          'definition', pg_get_triggerdef(tg.oid, true)
        ) order by tg.tgname)
        from pg_trigger tg
        where tg.tgrelid = rb.relation_oid and not tg.tgisinternal
      ), '[]'::jsonb)
    ) as payload
  from relation_base rb
),
schema_rows as (
  select
    'schema'::text as object_type,
    n.nspname as sort_key,
    jsonb_build_object(
      'name', n.nspname,
      'owner', pg_get_userbyid(n.nspowner),
      'comment', obj_description(n.oid, 'pg_namespace')
    ) as payload
  from pg_namespace n
  where n.nspname not like 'pg_%' and n.nspname <> 'information_schema'
),
function_rows as (
  select
    'function'::text as object_type,
    n.nspname || '.' || p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')' as sort_key,
    jsonb_build_object(
      'schema', n.nspname,
      'name', p.proname,
      'identity_arguments', pg_get_function_identity_arguments(p.oid),
      'result_type', pg_get_function_result(p.oid),
      'language', lang.lanname,
      'kind', case p.prokind when 'f' then 'function' when 'p' then 'procedure' when 'a' then 'aggregate' when 'w' then 'window' end,
      'volatility', case p.provolatile when 'i' then 'immutable' when 's' then 'stable' when 'v' then 'volatile' end,
      'security_definer', p.prosecdef,
      'comment', obj_description(p.oid, 'pg_proc')
    ) as payload
  from pg_proc p
  join pg_namespace n on n.oid = p.pronamespace
  join pg_language lang on lang.oid = p.prolang
  where n.nspname not like 'pg_%'
    and n.nspname not in ('information_schema', 'extensions')
),
enum_rows as (
  select
    'enum'::text as object_type,
    n.nspname || '.' || t.typname as sort_key,
    jsonb_build_object(
      'schema', n.nspname,
      'name', t.typname,
      'values', jsonb_agg(e.enumlabel order by e.enumsortorder)
    ) as payload
  from pg_type t
  join pg_namespace n on n.oid = t.typnamespace
  join pg_enum e on e.enumtypid = t.oid
  where n.nspname not like 'pg_%' and n.nspname <> 'information_schema'
  group by n.nspname, t.typname
),
extension_rows as (
  select
    'extension'::text as object_type,
    ext.extname as sort_key,
    jsonb_build_object(
      'name', ext.extname,
      'version', ext.extversion,
      'schema', n.nspname,
      'comment', obj_description(ext.oid, 'pg_extension')
    ) as payload
  from pg_extension ext
  join pg_namespace n on n.oid = ext.extnamespace
),
bucket_rows as (
  select
    'bucket'::text as object_type,
    b.id::text as sort_key,
    jsonb_build_object(
      'id', b.id,
      'name', b.name,
      'public', b.public,
      'file_size_limit', b.file_size_limit,
      'allowed_mime_types', b.allowed_mime_types,
      'created_at', b.created_at,
      'updated_at', b.updated_at
    ) as payload
  from storage.buckets b
)
select object_type, payload
from (
  select * from schema_rows
  union all select * from relation_rows
  union all select * from function_rows
  union all select * from enum_rows
  union all select * from extension_rows
  union all select * from bucket_rows
) catalog
order by object_type, sort_key;
"""


SYSTEM_SCHEMAS = {
    "auth",
    "cron",
    "extensions",
    "graphql",
    "graphql_public",
    "net",
    "realtime",
    "storage",
    "supabase_migrations",
    "vault",
}


def classify_relation(relation: dict[str, Any]) -> str:
    """Assign one practical owner-facing domain to a relation."""
    schema = relation["schema"].lower()
    name = relation["name"].lower()
    qualified = f"{schema}.{name}"

    if schema in SYSTEM_SCHEMAS:
        return "Supabase platform"
    if schema == "history":
        return "Historical NBA archive"
    if schema in {"mp", "im"}:
        return "Games & multiplayer"
    if "wnba" in qualified:
        return "WNBA data & metrics"
    if any(token in name for token in (
        "draft", "prospect", "scout", "lottery", "redraft", "combine", "college"
    )):
        return "Draft & scouting"
    if name.startswith("v3_") or any(token in name for token in (
        "contract", "free_agent", "free_agency", "cap_hold", "dead_money", "salary"
    )):
        return "Contracts & free agency"
    if any(token in name for token in (
        "sixrings", "impact_monster", "solo_cube", "teammate_race", "arena", "mp_",
        "laker_game", "ranked_player", "darko_team_game", "team_game_result"
    )):
        return "Games & multiplayer"
    if any(token in name for token in (
        "playlist", "mixtape", "video_link", "videolink", "shotchart"
    )):
        return "Video & media"
    if any(token in name for token in (
        "profile", "favorite", "comment", "reaction", "feedback", "search_log", "follow",
        "message", "friendship", "forum", "click_event", "beta_signup", "note"
    )):
        return "Users, community & product"
    if any(token in name for token in (
        "stage", "ingest", "publication", "snapshot", "override", "unknown_player", "config"
    )):
        return "Pipelines, staging & operations"
    if any(token in name for token in (
        "player", "team", "nba", "rapm", "wowy", "playtype", "factor", "metric", "shot",
        "matchup", "stats", "game", "index_master", "current_comp", "measurements"
    )):
        return "NBA data & metrics"
    return "Other application data"


def run_checked(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def query_metadata(project_ref: str) -> list[dict[str, Any]]:
    if shutil.which("supabase") is None:
        raise SystemExit("Supabase CLI is required. Install it, authenticate, and retry.")

    with tempfile.TemporaryDirectory(prefix="supabase-catalog-") as temp_name:
        temp_dir = Path(temp_name)
        run_checked(["supabase", "init", "--workdir", str(temp_dir), "--yes"])
        run_checked([
            "supabase", "link", "--workdir", str(temp_dir),
            "--project-ref", project_ref, "--yes",
        ])
        sql_path = temp_dir / "catalog.sql"
        sql_path.write_text(METADATA_SQL, encoding="utf-8")
        raw = run_checked([
            "supabase", "db", "query", "--workdir", str(temp_dir), "--linked",
            "--file", str(sql_path), "--output-format", "json",
        ])

    response = json.loads(raw)
    rows = response.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Supabase CLI returned no metadata rows")
    return rows


def build_catalog(rows: list[dict[str, Any]], project_ref: str, project_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        object_type = row.get("object_type")
        payload = row.get("payload")
        if object_type and isinstance(payload, dict):
            grouped[object_type].append(payload)

    relations = grouped["relation"]
    for relation in relations:
        relation["domain"] = classify_relation(relation)
        constraints = relation.get("constraints", [])
        relation["primary_key"] = next(
            (constraint.get("columns", []) for constraint in constraints if constraint.get("type") == "primary key"),
            [],
        )
        relation["foreign_key_count"] = sum(
            constraint.get("type") == "foreign key" for constraint in constraints
        )
        relation["risk_flags"] = []
        if relation["kind"] in {"table", "partitioned table"}:
            if not relation.get("rls_enabled"):
                relation["risk_flags"].append("RLS off")
            if not relation["primary_key"]:
                relation["risk_flags"].append("No primary key")
            if relation.get("estimated_rows", 0) == 0:
                relation["risk_flags"].append("Empty")
        if relation.get("total_bytes", 0) >= 1_000_000_000:
            relation["risk_flags"].append("1 GB+")

    relations.sort(key=lambda item: (item["domain"], item["schema"], item["name"]))
    domain_counts = Counter(item["domain"] for item in relations)
    domain_bytes: Counter[str] = Counter()
    for item in relations:
        domain_bytes[item["domain"]] += int(item.get("total_bytes", 0) or 0)

    application_relations = [item for item in relations if item["schema"] not in SYSTEM_SCHEMAS]
    table_relations = [item for item in relations if item["kind"] in {"table", "partitioned table"}]
    application_tables = [item for item in application_relations if item["kind"] in {"table", "partitioned table"}]

    catalog = {
        "catalog_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": project_name,
            "ref": project_ref,
            "scope": "metadata only; no table records, auth identities, storage object names, or secrets",
        },
        "summary": {
            "schemas": len(grouped["schema"]),
            "relations": len(relations),
            "tables": len(table_relations),
            "application_relations": len(application_relations),
            "application_tables": len(application_tables),
            "views": sum(item["kind"] == "view" for item in relations),
            "materialized_views": sum(item["kind"] == "materialized view" for item in relations),
            "functions": len(grouped["function"]),
            "enums": len(grouped["enum"]),
            "extensions": len(grouped["extension"]),
            "total_relation_bytes": sum(int(item.get("total_bytes", 0) or 0) for item in relations),
            "application_relation_bytes": sum(int(item.get("total_bytes", 0) or 0) for item in application_relations),
            "rls_off_tables": sum(not item.get("rls_enabled") for item in table_relations),
            "application_rls_off_tables": sum(not item.get("rls_enabled") for item in application_tables),
            "tables_without_primary_key": sum(not item.get("primary_key") for item in table_relations),
        },
        "domains": [
            {"name": name, "relations": count, "bytes": domain_bytes[name]}
            for name, count in sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "schemas": sorted(grouped["schema"], key=lambda item: item["name"]),
        "relations": relations,
        "functions": sorted(grouped["function"], key=lambda item: (item["schema"], item["name"], item["identity_arguments"])),
        "enums": sorted(grouped["enum"], key=lambda item: (item["schema"], item["name"])),
        "extensions": sorted(grouped["extension"], key=lambda item: item["name"]),
        "buckets": sorted(grouped["bucket"], key=lambda item: str(item["name"])),
    }
    return catalog


def write_catalog(catalog: dict[str, Any], template_path: Path, html_path: Path, json_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")
    marker = "__SUPABASE_CATALOG_DATA__"
    if marker not in template:
        raise RuntimeError(f"Template marker {marker!r} is missing from {template_path}")

    serialized = json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    rendered = template.replace(marker, serialized)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(rendered, encoding="utf-8")
    json_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-ref", required=True, help="Supabase project ref")
    parser.add_argument("--project-name", default="databallr", help="Reader-facing project name")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = query_metadata(args.project_ref)
    catalog = build_catalog(rows, args.project_ref, args.project_name)
    write_catalog(catalog, args.template, args.html, args.json)
    summary = catalog["summary"]
    print(
        f"Wrote {args.html} and {args.json}: "
        f"{summary['schemas']} schemas, {summary['relations']} relations, "
        f"{summary['functions']} functions"
    )


if __name__ == "__main__":
    main()
