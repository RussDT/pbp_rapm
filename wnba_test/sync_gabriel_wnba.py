"""Safely clone or fast-forward Gabriel's public WNBA source checkout."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/gabriel1200/wnba.git"
DEFAULT_TARGET = Path("nba_pipeline/external/gabriel_wnba")


def run_git(args: list[str], cwd: Path | None = None) -> str:
    command = ["git", *args]
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def normalized_origin(origin: str) -> str:
    value = origin.strip().removesuffix(".git")
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.split(":", 1)[1]
    return value.rstrip("/")


def sync_checkout(repo_url: str, target: Path, branch: str) -> str:
    if (target / ".git").is_dir():
        origin = run_git(["config", "--get", "remote.origin.url"], cwd=target)
        if normalized_origin(origin) != normalized_origin(repo_url):
            raise RuntimeError(
                f"refusing to update {target}: origin is {origin!r}, expected {repo_url!r}"
            )
        dirty = run_git(["status", "--porcelain"], cwd=target)
        if dirty:
            raise RuntimeError(
                f"refusing to update dirty source checkout {target}; preserve or discard its changes first"
            )
        run_git(["pull", "--ff-only", "origin", branch], cwd=target)
    elif target.exists():
        if any(target.iterdir()):
            raise RuntimeError(
                f"refusing to clone into non-empty directory without a git checkout: {target}"
            )
        target.rmdir()
        run_git(["clone", "--depth", "1", "--branch", branch, repo_url, str(target)])
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", "--depth", "1", "--branch", branch, repo_url, str(target)])
    return run_git(["rev-parse", "HEAD"], cwd=target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone or fast-forward the public gabriel1200/wnba data checkout"
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()

    commit = sync_checkout(args.repo_url, args.target, args.branch)
    print(f"Gabriel WNBA checkout ready at {args.target}")
    print(f"Commit: {commit}")


if __name__ == "__main__":
    main()
