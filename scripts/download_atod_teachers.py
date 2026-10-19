"""Download the three pinned 4B teachers from HF mirror, resume and verify.

Uses only Python's standard library and never reads HF credentials or proxies.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "logs/setup_20261016/teacher_manifest.json"
DEST = Path("/mnt/data/home/zzr/models/ATOD_ckpt")
# One resolver request per shard minimizes pressure on the shared HF mirror.
CHUNK = 8 * 1024 * 1024 * 1024


def digest(path, etag):
    etag = etag.removeprefix("W/").strip('"')
    if len(etag) == 64:
        h = hashlib.sha256()
    elif len(etag) == 40:
        h = hashlib.sha1()
        h.update(f"blob {path.stat().st_size}\0".encode())
    else:
        raise ValueError(f"Unknown checksum type for {path.name}")
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest() == etag


def download(item, manifest):
    path = DEST / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path, item["etag"]):
            print(f"VERIFIED existing {item['path']}", flush=True)
            return
        raise RuntimeError(f"Existing final file checksum mismatch: {path}")
    temp = path.with_name(path.name + ".incomplete")
    url = f"{manifest['endpoint']}/{manifest['repo']}/resolve/{manifest['revision']}/{item['path']}"
    expected = item.get("size", 0)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_report = 0
    for attempt in range(4):
        try:
            while True:
                offset = temp.stat().st_size if temp.exists() else 0
                if expected and offset == expected:
                    break
                headers = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"}
                if expected:
                    headers["Range"] = f"bytes={offset}-{min(offset + CHUNK, expected) - 1}"
                elif offset:
                    headers["Range"] = f"bytes={offset}-"
                with opener.open(urllib.request.Request(url, headers=headers), timeout=90) as r:
                    if offset and r.status != 206:
                        raise RuntimeError("Server did not honor resume range")
                    if r.status == 206:
                        start = int(r.headers["Content-Range"].split()[1].split("-")[0])
                        if start != offset:
                            raise RuntimeError(f"Invalid range start {start}, expected {offset}")
                    with temp.open("ab") as out:
                        while block := r.read(4 * 1024 * 1024):
                            out.write(block)
                size = temp.stat().st_size
                if not expected:
                    break
                if size > expected or size == offset:
                    raise RuntimeError(f"Invalid download size {size} for {item['path']}")
                if time.monotonic() - last_report > 30:
                    print(f"PROGRESS {item['path']} {size}/{expected}", flush=True)
                    last_report = time.monotonic()
            if expected and temp.stat().st_size != expected:
                raise RuntimeError(f"Size mismatch: {item['path']}")
            if not digest(temp, item["etag"]):
                raise RuntimeError(f"Checksum mismatch: {item['path']}; partial file retained")
            temp.rename(path)
            print(f"VERIFIED {item['path']} {path.stat().st_size}", flush=True)
            return
        except Exception as e:
            print(f"ERROR attempt={attempt + 1} {item['path']} {type(e).__name__}: {str(e).split('?')[0]}", flush=True)
            if attempt == 3:
                raise
            delay = max(15, 15 * (attempt + 1))
            print(f"BACKOFF {delay}s {item['path']}", flush=True)
            time.sleep(delay)


if __name__ == "__main__":
    manifest = json.loads(MANIFEST.read_text())
    print(f"START pid={os.getpid()} repo={manifest['repo']} revision={manifest['revision']}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: download(item, manifest), manifest["files"]))
    print("ALL THREE TEACHERS VERIFIED", flush=True)
