"""Fetch WebShop's required spaCy English model via HF mirror, with SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

repo = "spacy/en_core_web_sm"
revision = "8fb6f8155286360bf5486fe399e2738d1df7ded2"
base = "https://hf-mirror.com"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def open_url(url):
    return opener.open(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60)

dest = Path("/mnt/data/home/zzr/cache/wheels/en_core_web_sm-3.7.1-py3-none-any.whl")
# SHA256 from the pinned official repository metadata, verified on first fetch.
expected_sha256 = "86cc141f63942d4b2c5fcee06630fd6f904788d2f0ab005cce45aadb8fb73889"
if dest.exists():
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == expected_sha256
    print(f"VERIFIED cached {dest}")
    raise SystemExit(0)
metadata = json.load(open_url(f"{base}/api/models/{repo}/revision/{revision}?blobs=true"))
item = next(x for x in metadata["siblings"] if x["rfilename"].endswith(".whl"))
dest = Path("/mnt/data/home/zzr/cache/wheels/en_core_web_sm-3.7.1-py3-none-any.whl")
dest.parent.mkdir(parents=True, exist_ok=True)
if not dest.exists():
    temp = dest.with_suffix(".incomplete")
    with open_url(f"{base}/{repo}/resolve/{revision}/{item['rfilename']}") as src, temp.open("wb") as out:
        while block := src.read(4 * 1024 * 1024):
            out.write(block)
    assert temp.stat().st_size == item["size"]
    assert hashlib.sha256(temp.read_bytes()).hexdigest() == item["lfs"]["sha256"]
    temp.rename(dest)
assert hashlib.sha256(dest.read_bytes()).hexdigest() == item["lfs"]["sha256"]
print(f"VERIFIED {dest}")
