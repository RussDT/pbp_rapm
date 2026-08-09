"""Normalize Gabriel's modern WNBA PBP into the repo's lineup-aware raw schema."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.fetch_wnba_cdn import (
    BOXSCORE_URL,
    RAW_COLUMNS,
    fetch_json,
    normalize_actions,
)


DEFAULT_SOURCE = Path("nba_pipeline/external/gabriel_wnba/pbp_modern")
DEFAULT_OUTPUT_DIR = Path("wnba_test/GabrielNormalized")
DEFAULT_BOXSCORE_CACHE = Path("nba_pipeline/external/gabriel_wnba_boxscores")


class IncompleteModernGame(RuntimeError):
    """Raised when Gabriel's modern source ends before a terminal Game End."""


def game_metadata(game_id: str) -> tuple[int, str]:
    value = str(game_id).strip()
    if len(value) != 10 or not value.isdigit():
        raise ValueError(f"invalid WNBA game id: {game_id!r}")
    season = 2000 + int(value[3:5])
    season_type = {"102": "RS", "104": "PS"}.get(value[:3])
    if season_type is None:
        raise ValueError(f"unsupported WNBA game-id prefix: {value[:3]}")
    return season, season_type


def modern_game_is_complete(path: Path) -> bool:
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in {"actionType", "subType", "description"},
        low_memory=False,
    )
    if frame.empty:
        return False
    last = frame.iloc[-1].fillna("")
    return (
        str(last.get("actionType", "")).lower() == "game"
        and str(last.get("subType", "")).lower() == "end"
    ) or str(last.get("description", "")).strip().lower() == "game end"


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def load_boxscore(
    game_id: str,
    cache_dir: Path,
    allow_network: bool = True,
) -> dict[str, Any]:
    path = cache_dir / f"{game_id}.json"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if not allow_network:
        raise FileNotFoundError(f"boxscore cache miss for {game_id}: {path}")

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            boxscore = fetch_json(BOXSCORE_URL.format(game_id=game_id))
            break
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(0.5 * (2**attempt))
    else:
        raise RuntimeError(f"boxscore fetch failed for {game_id}: {last_error}")
    returned_game_id = str(boxscore.get("game", {}).get("gameId") or game_id)
    if returned_game_id != game_id:
        raise RuntimeError(
            f"boxscore game id mismatch: requested {game_id}, received {returned_game_id}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cache_dir,
        prefix=f".{game_id}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(boxscore, handle, separators=(",", ":"))
        temp_name = handle.name
    os.replace(temp_name, path)
    return boxscore


def normalize_local_game(
    path: Path,
    cache_dir: Path,
    allow_network: bool = True,
) -> pd.DataFrame:
    game_id = path.stem
    game_metadata(game_id)
    if not modern_game_is_complete(path):
        raise IncompleteModernGame(f"refusing incomplete modern PBP game: {path}")
    source = pd.read_csv(path, low_memory=False)
    if source["actionNumber"].duplicated().any():
        raise RuntimeError(f"duplicate actionNumber values in {path}")
    boxscore = load_boxscore(game_id, cache_dir, allow_network=allow_network)
    normalized = normalize_actions(
        game_id=game_id,
        actions=json_safe_records(source),
        boxscore=boxscore,
    )
    if len(normalized) != len(source):
        raise RuntimeError(
            f"row-count mismatch for {game_id}: {len(source)} source vs {len(normalized)} normalized"
        )
    return normalized


def source_paths(
    source_dir: Path,
    seasons: set[int],
    season_types: set[str],
) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(source_dir.glob("*.csv")):
        season, season_type = game_metadata(path.stem)
        if season in seasons and season_type in season_types:
            selected.append(path)
    return selected


def output_path(output_dir: Path, season: int, season_type: str) -> Path:
    suffix = "_PS" if season_type == "PS" else ""
    return output_dir / f"WNBA{season % 100:02d}{suffix}.csv"


def import_games(
    source_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    seasons: set[int],
    season_types: set[str],
    workers: int,
    allow_network: bool,
    overwrite: bool,
    exclude_incomplete: bool,
    limit: int | None = None,
) -> dict[tuple[int, str], pd.DataFrame]:
    paths = source_paths(source_dir, seasons, season_types)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError("no Gabriel modern PBP games matched the requested seasons")

    frames: dict[tuple[int, str], list[pd.DataFrame]] = {}
    failures: list[str] = []
    skipped_incomplete: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                normalize_local_game,
                path,
                cache_dir,
                allow_network,
            ): path
            for path in paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                frame = future.result()
                frames.setdefault(game_metadata(path.stem), []).append(frame)
            except IncompleteModernGame as exc:
                if exclude_incomplete:
                    skipped_incomplete.append(f"{path.name}: {exc}")
                else:
                    failures.append(f"{path.name}: {exc}")
            except Exception as exc:
                failures.append(f"{path.name}: {exc}")
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(f"{len(failures)} game imports failed:\n{preview}")

    outputs: dict[tuple[int, str], pd.DataFrame] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, game_frames in sorted(frames.items()):
        season, season_type = key
        target = output_path(output_dir, season, season_type)
        if target.exists() and not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {target}")
        output = pd.concat(game_frames, ignore_index=True)
        output = output[RAW_COLUMNS]
        output["game_id"] = output["game_id"].astype(str).str.zfill(10)
        output = output.sort_values(["game_id", "period", "event_num"], kind="mergesort")
        output.to_csv(target, index=False)
        outputs[key] = output
        print(
            f"Saved {len(output):,} rows across {output['game_id'].nunique():,} games to {target}"
        )
    manifest = {
        "source_directory": str(source_dir),
        "selected_source_games": len(paths),
        "imported_games": sum(
            int(output["game_id"].nunique()) for output in outputs.values()
        ),
        "skipped_incomplete_games": skipped_incomplete,
        "outputs": {
            f"{season}_{season_type}": {
                "path": str(output_path(output_dir, season, season_type)),
                "rows": len(output),
                "games": int(output["game_id"].nunique()),
            }
            for (season, season_type), output in sorted(outputs.items())
        },
    }
    manifest_path = output_dir / "import_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if skipped_incomplete:
        print(
            f"Explicitly excluded {len(skipped_incomplete)} incomplete modern game(s); "
            f"see {manifest_path}"
        )
    return outputs


def parse_seasons(values: list[str]) -> set[int]:
    seasons: set[int] = set()
    for value in values:
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start, end = int(start_text), int(end_text)
            seasons.update(range(start, end + 1))
        else:
            seasons.add(int(value))
    return seasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize local Gabriel modern WNBA PBP with cached liveData boxscore starters"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2022-2026"],
        help="Season years or inclusive ranges, e.g. 2022-2025",
    )
    parser.add_argument(
        "--season-types",
        nargs="+",
        choices=["RS", "PS"],
        default=["RS", "PS"],
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--boxscore-cache", type=Path, default=DEFAULT_BOXSCORE_CACHE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--exclude-incomplete",
        action="store_true",
        help="Explicitly omit modern files without a terminal Game End and record them in the manifest",
    )
    parser.add_argument("--limit", type=int, help="Bounded smoke-test game count")
    args = parser.parse_args()

    import_games(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        cache_dir=args.boxscore_cache,
        seasons=parse_seasons(args.seasons),
        season_types=set(args.season_types),
        workers=args.workers,
        allow_network=not args.no_network,
        overwrite=args.overwrite,
        exclude_incomplete=args.exclude_incomplete,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
