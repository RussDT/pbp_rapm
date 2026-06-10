#!/usr/bin/env python3
"""Build a non-destructive source/artifact inventory for this repo."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ARTIFACT_SUFFIXES = {
    ".csv",
    ".csv.gz",
    ".gz",
    ".jsonl",
    ".joblib",
    ".log",
    ".npz",
    ".parquet",
    ".pkl",
    ".pt",
}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache"}

SCRIPT_DOC_HINTS = (
    "daily_rapm_update.sh",
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "PIPELINE_QUICKSTART.md",
)


def run_git(repo_root: Path, args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def iter_files(repo_root: Path):
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        root_path = Path(root)
        for file_name in files:
            path = root_path / file_name
            yield path.relative_to(repo_root)


def is_artifact(path: Path) -> bool:
    text = str(path)
    return any(text.endswith(suffix) for suffix in ARTIFACT_SUFFIXES)


def top_bucket(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 2 and parts[0] in {"nba_pipeline", "wnba_test", "randle_strong_weak", "csv-viewer"}:
        return f"{parts[0]}/{parts[1]}"
    if parts:
        return parts[0]
    return "."


def file_size(repo_root: Path, path: Path) -> int:
    try:
        return (repo_root / path).stat().st_size
    except OSError:
        return 0


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def grep_count(repo_root: Path, needle: str, haystacks: list[str]) -> int:
    count = 0
    for rel in haystacks:
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            if needle in path.read_text(errors="ignore"):
                count += 1
        except OSError:
            continue
    return count


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[2],
        type=Path,
        help="Repository root to inspect.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Directory for CSV/Markdown reports. Defaults to stdout only.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    all_files = list(iter_files(repo_root))
    tracked = set(Path(p) for p in run_git(repo_root, ["ls-files"]))
    ignored_lines = run_git(repo_root, ["status", "--ignored", "--short"])
    ignored_paths = {
        Path(line[3:])
        for line in ignored_lines
        if line.startswith("!! ") and line[3:]
    }

    artifact_files = [p for p in all_files if is_artifact(p)]
    artifact_bucket_count: Counter[str] = Counter()
    artifact_bucket_size: defaultdict[str, int] = defaultdict(int)
    for path in artifact_files:
        bucket = top_bucket(path)
        artifact_bucket_count[bucket] += 1
        artifact_bucket_size[bucket] += file_size(repo_root, path)

    ext_count: Counter[str] = Counter()
    for path in tracked:
        ext_count[path.suffix or "[no_ext]"] += 1

    doc_haystacks = [
        str(path)
        for path in tracked
        if path.suffix in {".md", ".sh", ".py", ".toml", ".txt"}
        or path.name in SCRIPT_DOC_HINTS
    ]
    script_rows: list[dict[str, object]] = []
    for path in sorted(p for p in tracked if p.suffix == ".py"):
        name = path.name
        references = grep_count(repo_root, name, doc_haystacks)
        if str(path).startswith("nba_pipeline/scripts/process_rapm_blocks/"):
            suggested = "active_core"
        elif references:
            suggested = "referenced"
        elif str(path).startswith("nba_pipeline/scripts/"):
            suggested = "needs_classification"
        elif str(path).startswith("wnba_test/"):
            suggested = "wnba_needs_classification"
        else:
            suggested = "archive_candidate_check"
        script_rows.append(
            {
                "path": str(path),
                "size_bytes": file_size(repo_root, path),
                "doc_or_code_reference_count": references,
                "suggested_bucket": suggested,
            }
        )

    artifact_rows = [
        {
            "bucket": bucket,
            "artifact_count": artifact_bucket_count[bucket],
            "size_bytes": artifact_bucket_size[bucket],
            "size_human": human_size(artifact_bucket_size[bucket]),
        }
        for bucket in sorted(
            artifact_bucket_count,
            key=lambda item: artifact_bucket_size[item],
            reverse=True,
        )
    ]

    summary_lines = [
        "# Repo Inventory",
        "",
        f"- Repo root: `{repo_root}`",
        f"- Total files outside skipped dirs: `{len(all_files)}`",
        f"- Tracked files: `{len(tracked)}`",
        f"- Tracked Python files: `{sum(1 for p in tracked if p.suffix == '.py')}`",
        f"- Artifact-like files: `{len(artifact_files)}`",
        f"- Ignored paths reported by git: `{len(ignored_paths)}`",
        "",
        "## Tracked Extensions",
        "",
    ]
    for ext, count in ext_count.most_common():
        summary_lines.append(f"- `{ext}`: `{count}`")

    summary_lines.extend(["", "## Largest Artifact Buckets", ""])
    for row in artifact_rows[:25]:
        summary_lines.append(
            f"- `{row['bucket']}`: `{row['artifact_count']}` files, `{row['size_human']}`"
        )

    summary_lines.extend(
        [
            "",
            "## Script Classification Notes",
            "",
            "The suggested buckets are a first-pass shortlist only. Check imports, docs, shell jobs, and git history before moving or deleting scripts.",
            "",
        ]
    )
    for row in script_rows[:40]:
        summary_lines.append(
            f"- `{row['path']}`: `{row['suggested_bucket']}`, references `{row['doc_or_code_reference_count']}`"
        )

    markdown = "\n".join(summary_lines) + "\n"

    if args.output:
        output_dir = args.output
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "repo_inventory.md").write_text(markdown)
        write_csv(
            output_dir / "artifact_buckets.csv",
            artifact_rows,
            ["bucket", "artifact_count", "size_bytes", "size_human"],
        )
        write_csv(
            output_dir / "script_inventory.csv",
            script_rows,
            ["path", "size_bytes", "doc_or_code_reference_count", "suggested_bucket"],
        )
        print(f"Wrote inventory reports to {output_dir}")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
