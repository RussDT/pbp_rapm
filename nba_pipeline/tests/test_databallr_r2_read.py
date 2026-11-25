"""R2 reader operations checks: wrangler failure reporting, redaction, mirror closure."""
from pathlib import Path
import importlib.util
import os
import socket
import subprocess

import pytest

P = Path(__file__).resolve().parents[1] / "scripts/databallr_r2_read.py"
spec = importlib.util.spec_from_file_location("databallr_r2_read_tests", P)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

FAKE_TOKEN = "Zq" * 25  # 50 opaque characters, never a real credential


def test_redact_masks_token_like_values():
    text = (f'{{"token":"{FAKE_TOKEN}"}} Authorization: Bearer {FAKE_TOKEN} '
            f'account {"ab" * 16} EPERM: operation not permitted')
    out = M.redact(text)
    assert FAKE_TOKEN not in out and "ab" * 16 not in out
    assert "EPERM: operation not permitted" in out


def test_wrangler_failure_surfaces_redacted_stderr(monkeypatch, tmp_path):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    wrangler = tmp_path / "a/node_modules/.bin/wrangler"
    wrangler.parent.mkdir(parents=True)
    wrangler.write_text("#!/bin/sh\n")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="",
            stderr=f"Error: EPERM: operation not permitted, open 'x.js' token={FAKE_TOKEN}")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as error:
        M.R2Reader(wrangler=wrangler)
    message = str(error.value)
    assert "exit 1" in message and "EPERM: operation not permitted" in message
    assert FAKE_TOKEN not in message


def _pnpm_package(root, entry, name, deps=()):
    modules = root / entry / "node_modules"
    (modules / name).mkdir(parents=True)
    (modules / name / "index.js").write_text("")
    for dep_entry, dep_name in deps:
        os.symlink(f"../../{dep_entry}/node_modules/{dep_name}", modules / dep_name)


def test_mirror_copies_transitive_closure_with_relative_links(tmp_path):
    product = tmp_path / "product"
    store = product / "node_modules/.pnpm"
    _pnpm_package(store, "dep-b@1", "dep-b")
    _pnpm_package(store, "dep-a@1", "dep-a", [("dep-b@1", "dep-b")])
    _pnpm_package(store, "wrangler@9", "wrangler", [("dep-a@1", "dep-a")])
    _pnpm_package(store, "unrelated@1", "unrelated")
    bin_dir = store / "wrangler@9/node_modules/wrangler/bin"
    bin_dir.mkdir()
    (bin_dir / "wrangler.js").write_text("")
    os.symlink(".pnpm/wrangler@9/node_modules/wrangler", product / "node_modules/wrangler")

    script = M.mirrored_wrangler_js(product=product, mirror_root=tmp_path / "cache")
    mirror = tmp_path / "cache/wrangler@9"
    assert script == mirror / ".pnpm/wrangler@9/node_modules/wrangler/bin/wrangler.js"
    assert sorted(p.name for p in (mirror / ".pnpm").iterdir()) == ["dep-a@1", "dep-b@1", "wrangler@9"]
    nested = mirror / ".pnpm/dep-a@1/node_modules/dep-b/index.js"
    assert nested.exists() and str(product) not in os.path.realpath(nested)
    assert script.parents[5] == mirror


def test_address_order_interleaves_and_prefers_last_working_family(monkeypatch):
    v6 = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (f"::{i}", 443, 0, 0)) for i in range(3)]
    v4 = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"10.0.0.{i}", 443)) for i in range(2)]
    monkeypatch.setattr(M.socket, "getaddrinfo", lambda *a, **k: v6 + v4)
    monkeypatch.setattr(M, "_CONNECTED_FAMILY", {})
    order = [info[0] for info in M._ordered_addresses("example.test", 443)]
    assert order[:2] == [socket.AF_INET6, socket.AF_INET] and len(order) == 5
    M._CONNECTED_FAMILY["example.test"] = socket.AF_INET
    assert M._ordered_addresses("example.test", 443)[0][0] == socket.AF_INET
