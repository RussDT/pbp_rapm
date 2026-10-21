"""Read-only, rate-limited R2 source access using the existing Wrangler login.

Credentials are captured in memory, never printed or written to source manifests.
Only GET is implemented. This avoids thousands of Worker/API log requests when
the same admitted archive can be read as larger source objects.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ACCOUNT = "c106bf9fdebefc994effa68697f1d8fe"
BUCKETS = {"databallr-datasets", "databallr-datasets-internal"}
PRODUCT = Path(__file__).resolve().parents[4] / "products/databallr"


class R2Reader:
    def __init__(self, wrangler=None, requests_per_second=3.0):
        self.wrangler = Path(wrangler or PRODUCT / "node_modules/.bin/wrangler")
        token = os.environ.get("CLOUDFLARE_API_TOKEN")
        if not token:
            result = subprocess.run(
                [str(self.wrangler), "auth", "token", "--json"],
                cwd=self.wrangler.parents[2], capture_output=True, text=True,
                check=True, timeout=90,
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
                with urllib.request.urlopen(request, timeout=120) as response:
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
        with urllib.request.urlopen(request, timeout=90) as response:
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
