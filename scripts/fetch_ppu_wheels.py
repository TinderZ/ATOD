"""Cache exact platform wheels from the official Alibaba PPU mirror."""
import argparse
import concurrent.futures
import email
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("environment", choices=["atod", "webshop"])
args = parser.parse_args()
dest = Path("/mnt/data/home/zzr/cache/wheels")
dest.mkdir(parents=True, exist_ok=True)
if args.environment == "atod":
    py, cuda, system, torch = "cp312", "129", "ubuntu2404", "2.8.0"
    packages = [("vllm", "0.11.0", "2.0.0", "oe", True), ("flash_attn", "2.7.4.post1", "2.0.0", "oe", True), ("triton", "3.4.0", "2.0.0", "oe", False), ("torchvision", "0.23.0", "1.7.0", "ce", True), ("torchaudio", "2.8.0", "1.7.0", "ce", True)]
else:
    py, cuda, system, torch = "cp310", "126", "ubuntu2204", "2.6.0"
    packages = [("vllm", "0.8.5", "1.7.0", "oe", True), ("flash_attn", "2.7.4.post1", "2.0.0", "oe", True), ("triton", "3.2.0", "1.7.0", "oe", False), ("torchvision", "0.21.0", "1.7.0", "ce", True), ("torchaudio", "2.6.0", "1.7.0", "ce", True), ("xformers", "0.0.29.post1", "1.7.0", "oe", True)]

packages.append(("acext", "1.0.0", "2.0.0", "oe", True))

def fetch(spec):
    name, version, sdk, edition, uses_torch = spec
    target = dest / f"{name}-{version}+ppu{sdk}-{py}-{py}-linux_x86_64.whl"
    framework = "torch" + torch if uses_torch else ""
    filename = f"{name}-{version}+cu{cuda}{framework}{system}{edition}-{py}-{py}-linux_x86_64.whl"
    url = f"https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/generic/{name}/{version}+ppu{sdk}/{filename}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if not target.exists():
        temporary = target.with_suffix(".incomplete")
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with opener.open(urllib.request.Request(url, headers=headers), timeout=90) as source:
            if offset and source.status != 206:
                raise RuntimeError("Server ignored Range; partial retained")
            with temporary.open("ab") as output:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
        with zipfile.ZipFile(temporary) as wheel:
            metadata = email.message_from_bytes(wheel.read(next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))))
        if metadata["Version"] != f"{version}+ppu{sdk}":
            raise RuntimeError(f"Unexpected package version: {metadata['Version']}")
        temporary.rename(target)
    sha = hashlib.sha256()
    with target.open("rb") as cached:
        for block in iter(lambda: cached.read(8 * 1024 * 1024), b""):
            sha.update(block)
    print(f"CACHED {target.name} {target.stat().st_size}", flush=True)
    return {"path": str(target), "source": url, "size": target.stat().st_size, "sha256": sha.hexdigest()}

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(fetch, packages))
out = Path(__file__).resolve().parents[1] / f"logs/setup_20261016/{args.environment}_ppu_wheels.json"
out.write_text(json.dumps(results, indent=2))
print("PPU WHEEL CACHE READY", flush=True)
