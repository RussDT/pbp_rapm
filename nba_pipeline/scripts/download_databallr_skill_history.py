"""Pin and read the two historical Iceberg player tables, without SQL or writes.

Requires fastavro and pyarrow. Iceberg manifest season bounds are checked BEFORE
downloading any data file, so a future-expanded table fails closed. Current
snapshots are resolved through the named catalog table, never bucket recency.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import time
import urllib.parse

import fastavro
import pyarrow.parquet as pq

from databallr_r2_read import R2Reader

DEFAULT = Path(__file__).resolve().parents[1] / "results/estimated_skills/full_history_v2/source"
BUCKET = "databallr-datasets-internal"


def object_key(uri):
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != BUCKET:
        raise ValueError("Manifest refers to an unexpected source bucket")
    return parsed.path.lstrip("/")


def season_bounds(data_file, season_field_id):
    def lookup(name):
        bounds = data_file.get(name) or {}
        if isinstance(bounds, list):
            bounds = {x["key"]: x["value"] for x in bounds}
        return bounds.get(season_field_id, bounds.get(str(season_field_id)))
    lower = lookup("lower_bounds")
    upper = lookup("upper_bounds")
    if lower is None or upper is None:
        raise ValueError("No season bounds: refusing an unproven historical data file")
    lo = int.from_bytes(lower, "little", signed=True)
    hi = int.from_bytes(upper, "little", signed=True)
    if lo < 1997 or hi > 2026:
        raise ValueError(f"Data file crosses historical season fence: {lo}..{hi}")
    return lo, hi


def acquire(reader, name, out):
    out = Path(out) / name
    out.mkdir(parents=True, exist_ok=True)
    catalog_file = out / "catalog.json"
    if catalog_file.exists():
        catalog = json.loads(catalog_file.read_text())
    else:
        catalog = reader.catalog_table(name)
        catalog_file.write_text(json.dumps(catalog, indent=2) + "\n")
    metadata = catalog["metadata"]
    current = metadata["current-snapshot-id"]
    snapshot = next(x for x in metadata["snapshots"] if x["snapshot-id"] == current)
    schema = next(x for x in metadata["schemas"] if x["schema-id"] == metadata["current-schema-id"])
    season_id = next(x["id"] for x in schema["fields"] if x["name"] == "season")
    if int(snapshot["summary"].get("total-delete-files", 0)):
        raise ValueError("Delete files require an Iceberg engine; raw export is forbidden")
    receipt = {"name": name, "table_uuid": catalog["table_uuid"], "snapshot_id": current,
               "metadata_location": catalog["metadata_location"], "source_summary": snapshot["summary"], "objects": []}
    manifests = list(fastavro.reader(io.BytesIO(reader._get(BUCKET, object_key(snapshot["manifest-list"])))))
    data = {}
    for manifest in manifests:
        entries = list(fastavro.reader(io.BytesIO(reader._get(BUCKET, object_key(manifest["manifest_path"])))))
        for entry in entries:
            if entry["status"] == 2:
                continue
            f = entry["data_file"]
            if f.get("content", 0) != 0:
                raise ValueError("Unexpected non-data content in a historical manifest")
            bounds = season_bounds(f, season_id)
            data[f["file_path"]] = {"records": f["record_count"], "season_bounds": bounds, "bytes": f["file_size_in_bytes"]}
    expected = int(snapshot["summary"]["total-records"])
    if sum(x["records"] for x in data.values()) != expected:
        raise ValueError("Manifest record count differs from named snapshot")
    for index, (uri, description) in enumerate(sorted(data.items())):
        destination = out / f"part-{index:03d}.parquet"
        if destination.exists() and destination.stat().st_size == description["bytes"]:
            import hashlib
            record = {"bucket": BUCKET, "key": object_key(uri), "bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
        else:
            record = reader.download(BUCKET, object_key(uri), destination)
        if record["bytes"] != description["bytes"] or pq.read_metadata(destination).num_rows != description["records"]:
            raise ValueError("Downloaded source file differs from Iceberg manifest")
        record.update(description, local_file=str(destination.resolve()))
        receipt["objects"].append(record)
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"table": name, "snapshot": current, "rows": expected, "files": len(data), "bytes": sum(x["bytes"] for x in data.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    started = time.monotonic()
    reader = R2Reader()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda name: acquire(reader, name, args.out), ["player_game", "player_game_report"]):
            print(json.dumps(result), flush=True)
    print(json.dumps({"source_download_seconds": round(time.monotonic() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
