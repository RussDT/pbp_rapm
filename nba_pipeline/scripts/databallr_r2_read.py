"""Read-only, rate-limited R2 source access using the existing Wrangler login.

Credentials are captured in memory, never printed or written to source manifests.
Only GET is implemented. This avoids thousands of Worker/API log requests when
the same admitted archive can be read as larger source objects.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ACCOUNT = "c106bf9fdebefc994effa68697f1d8fe"
BUCKETS = {"databallr-datasets", "databallr-datasets-internal"}
PRODUCT = Path(__file__).resolve().parents[4] / "products/databallr"

_SECRET_PATTERNS = (
    # JSON/key=value fields whose value is a credential.
    re.compile(r'(?i)("?(?:[a-z_]*token|secret|password|api[_-]?key|authorization)"?\s*[:=]\s*"?)[^\s",}]+'),
    re.compile(r"(?i)(bearer\s+)\S+"),
    # Long opaque strings (tokens, keys, signatures) and 32-hex account ids.
    re.compile(r"()[A-Za-z0-9_\-\.]{40,}"),
    re.compile(r"()\b[0-9a-f]{32}\b"),
)


def redact(text, limit=2000):
    """Return subprocess output safe to log: credential-like values replaced."""
    text = (text or "").strip()
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) + "[REDACTED]", text)
    return text[-limit:]


CONNECT_TIMEOUT = 10.0
_CONNECTED_FAMILY = {}


def _ordered_addresses(host, port):
    """Resolved addresses with families interleaved, last working family first."""
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    by_family = {}
    for info in infos:
        by_family.setdefault(info[0], []).append(info)
    families = list(by_family)
    preferred = _CONNECTED_FAMILY.get(host)
    if preferred in by_family:
        families.remove(preferred)
        families.insert(0, preferred)
    ordered = []
    while any(by_family.values()):
        for family in families:
            if by_family[family]:
                ordered.append(by_family[family].pop(0))
    return ordered


def _create_connection(address, timeout=None, source_address=None):
    """socket.create_connection with a short per-address connect timeout.

    The standard version spends the full read timeout on each resolved address in
    turn. api.cloudflare.com resolves IPv6-first, so on a network without working
    IPv6 egress every R2 GET stalled for minutes before reaching IPv4. Here each
    address gets CONNECT_TIMEOUT to connect and the caller's timeout then applies
    to reads, as before.
    """
    host, port = address
    error = None
    for family, kind, proto, _, sockaddr in _ordered_addresses(host, port):
        sock = None
        try:
            sock = socket.socket(family, kind, proto)
            connect_timeout = CONNECT_TIMEOUT if timeout is None else min(CONNECT_TIMEOUT, timeout)
            sock.settimeout(connect_timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            sock.settimeout(timeout)
            _CONNECTED_FAMILY[host] = family
            return sock
        except OSError as exc:
            error = exc
            if sock is not None:
                sock.close()
    raise error if error is not None else OSError(f"getaddrinfo returned no addresses for {host}")


class _FallbackHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _create_connection


class _FallbackHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_FallbackHTTPSConnection, req,
                            context=self._context, check_hostname=self._check_hostname)


_OPENER = urllib.request.build_opener(_FallbackHTTPSHandler())


WRANGLER_MIRROR_ROOT =Path.home() / "Library/Caches/rapm-databallr-wrangler"


def _pnpm_closure(pnpm_root, first):
    """Names of the .pnpm store entries reachable from ``first`` via dependency symlinks."""
    seen, todo = set(), [first]
    while todo:
        entry = todo.pop()
        if entry in seen:
            continue
        seen.add(entry)
        modules = pnpm_root / entry / "node_modules"
        items = []
        for item in modules.iterdir():
            if item.name.startswith("@") and not item.is_symlink():
                items.extend(item.iterdir())
            else:
                items.append(item)
        for item in items:
            if item.is_symlink():
                target = Path(os.path.realpath(item))
                try:
                    todo.append(target.relative_to(pnpm_root).parts[0])
                except ValueError:
                    raise RuntimeError(f"wrangler dependency resolves outside the pnpm store: {item.name}")
    return seen


def mirrored_wrangler_js(product=PRODUCT, mirror_root=WRANGLER_MIRROR_ROOT):
    """Return wrangler's bin/wrangler.js inside a private mirror outside ~/Documents.

    Under launchd, macOS privacy protection refuses node (EPERM) when it opens any
    file below ~/Documents, while the Python interpreter running this job may read
    them. Python therefore copies wrangler's pnpm dependency closure, keeping pnpm's
    relative symlinks, into ~/Library/Caches and node runs from there. The mirror is
    keyed by the installed wrangler store entry and rebuilt when databallr upgrades.
    """
    pnpm_root = Path(product) / "node_modules/.pnpm"
    real = Path(os.path.realpath(Path(product) / "node_modules/wrangler"))
    first = real.relative_to(pnpm_root.resolve()).parts[0]
    pnpm_root = pnpm_root.resolve()
    mirror = Path(mirror_root) / first
    script = mirror / ".pnpm" / first / "node_modules/wrangler/bin/wrangler.js"
    if (mirror / ".complete").exists() and script.exists():
        return script
    import shutil
    Path(mirror_root).mkdir(parents=True, exist_ok=True)
    staging = Path(mirror_root) / f".staging-{first}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    for entry in sorted(_pnpm_closure(pnpm_root, first)):
        shutil.copytree(pnpm_root / entry, staging / ".pnpm" / entry, symlinks=True)
    (staging / ".complete").write_text(first + "\n")
    shutil.rmtree(mirror, ignore_errors=True)
    os.replace(staging, mirror)
    for stale in Path(mirror_root).iterdir():
        if stale.name != first and not stale.name.startswith(".staging-"):
            shutil.rmtree(stale, ignore_errors=True)
    return script


def _wrangler_command(wrangler):
    """Command and cwd for ``wrangler auth token --json``."""
    if wrangler is not None:
        wrangler = Path(wrangler)
        return [str(wrangler)], wrangler.parents[2]
    import shutil
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is not on PATH; wrangler auth token needs Node.js >= 22")
    script = mirrored_wrangler_js()
    return [node, str(script)], script.parents[5]


class R2Reader:
    def __init__(self, wrangler=None, requests_per_second=3.0):
        self.wrangler = Path(wrangler) if wrangler else PRODUCT / "node_modules/.bin/wrangler"
        token = os.environ.get("CLOUDFLARE_API_TOKEN")
        if not token:
            command, cwd = _wrangler_command(wrangler)
            result = subprocess.run(
                command + ["auth", "token", "--json"],
                cwd=cwd, capture_output=True, text=True,
                check=False, timeout=90,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"wrangler auth token failed (exit {result.returncode}); "
                    f"stderr: {redact(result.stderr)!r}; stdout: {redact(result.stdout)!r}"
                )
            token = json.loads(result.stdout)["token"]
        self._token = token
        self._lock = threading.Lock()
        self._next = 0.0
        if requests_per_second <= 0 or requests_per_second > 3:
            raise ValueError("R2 source reads must use 0 < requests_per_second <= 3")
        self.interval = 1.0 / requests_per_second

    def _get(self, bucket, suffix, query=None):
        if bucket not in BUCKETS:
            raise ValueError("Unexpected source bucket")
        url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/r2/buckets/{bucket}/objects"
        if suffix:
            url += "/" + urllib.parse.quote(suffix, safe="/")
        if query:
            url += "?" + urllib.parse.urlencode(query)
        for attempt in range(6):
            with self._lock:
                now = time.monotonic()
                delay = max(0.0, self._next - now)
                self._next = now + delay + self.interval
            if delay:
                time.sleep(delay)
            request = urllib.request.Request(url, headers={"Authorization": "Bearer " + self._token})
            try:
                with _OPENER.open(request, timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 5:
                    raise RuntimeError(f"R2 GET failed: HTTP {error.code}, bucket={bucket}, key={suffix}") from None
                delay = min(60.0, max(float(error.headers.get("Retry-After", 0)), 2.0 ** attempt))
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError):
                if attempt == 5:
                    raise RuntimeError(f"R2 GET timed out: bucket={bucket}, key={suffix}") from None
                time.sleep(min(30.0, 2.0 ** attempt))

    def objects(self, bucket, prefix, delimiter=None):
        query = {"prefix": prefix, "per_page": 1000}
        if delimiter:
            query["delimiter"] = delimiter
        rows, folders, cursors = [], set(), set()
        while True:
            payload = json.loads(self._get(bucket, "", query))
            if not payload.get("success"):
                raise RuntimeError("R2 object inventory was unsuccessful")
            rows.extend(payload["result"])
            info = payload.get("result_info", {})
            folders.update(info.get("delimited", []))
            if not info.get("is_truncated", False):
                break
            cursor = info.get("cursor")
            if not cursor or cursor in cursors:
                raise RuntimeError("Truncated R2 inventory without a fresh cursor")
            cursors.add(cursor)
            query["cursor"] = cursor
        if any(not row["key"].startswith(prefix) for row in rows):
            raise RuntimeError("R2 inventory escaped requested source prefix")
        return rows, sorted(folders)

    def catalog_table(self, name):
        if name not in {"player_game", "player_game_report"}:
            raise ValueError("Only the two historical player source tables are admitted")
        url = (f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/r2-catalog/"
               f"databallr-datasets-internal/namespaces/nba_pbp/tables/{name}")
        request = urllib.request.Request(url, headers={"Authorization": "Bearer " + self._token})
        with _OPENER.open(request, timeout=90) as response:
            result = json.load(response)
        if not result.get("success") or result["result"]["identifier"] != {"namespace": ["nba_pbp"], "name": name}:
            raise RuntimeError("Unexpected catalog source identity")
        return result["result"]

    def download(self, bucket, key, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = self._get(bucket, key)
        temporary = destination.with_name(destination.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_bytes(raw)
        temporary.replace(destination)
        return {"bucket": bucket, "key": key, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
