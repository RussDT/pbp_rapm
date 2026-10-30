from nba_pipeline.scripts.build_supabase_catalog import build_catalog, classify_relation


def test_classify_relation_prioritizes_working_domains() -> None:
    assert classify_relation({"schema": "public", "name": "wnba_playtype_role"}) == "WNBA data & metrics"
    assert classify_relation({"schema": "public", "name": "v3_contract_salaries"}) == "Contracts & free agency"
    assert classify_relation({"schema": "mp", "name": "matchmaking_attempts"}) == "Games & multiplayer"
    assert classify_relation({"schema": "auth", "name": "users"}) == "Supabase platform"


def test_build_catalog_derives_keys_risks_and_counts() -> None:
    rows = [
        {"object_type": "schema", "payload": {"name": "public", "owner": "postgres", "comment": None}},
        {
            "object_type": "relation",
            "payload": {
                "schema": "public",
                "name": "player_stats",
                "qualified_name": "public.player_stats",
                "kind": "table",
                "estimated_rows": 12,
                "total_bytes": 2048,
                "rls_enabled": False,
                "columns": [{"name": "nba_id", "type": "integer"}],
                "constraints": [
                    {"type": "primary key", "columns": ["nba_id"]},
                    {"type": "foreign key", "columns": ["nba_id"], "references": "public.players"},
                ],
            },
        },
    ]

    catalog = build_catalog(rows, "project-ref", "databallr")
    relation = catalog["relations"][0]

    assert relation["domain"] == "NBA data & metrics"
    assert relation["primary_key"] == ["nba_id"]
    assert relation["foreign_key_count"] == 1
    assert relation["risk_flags"] == ["RLS off"]
    assert catalog["summary"]["application_tables"] == 1
    assert catalog["summary"]["application_rls_off_tables"] == 1
