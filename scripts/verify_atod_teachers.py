"""Inspect the pinned teachers without a GPU.

Usage: python scripts/verify_atod_teachers.py --root MODEL_ROOT [--output REPORT]
MODEL_ROOT can also be supplied through ATOD_TEACHERS_ROOT. By default the
report is printed to stdout; --output creates a new report without overwriting.
"""
import argparse
import json
import os
from pathlib import Path
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=os.environ.get('ATOD_TEACHERS_ROOT'))
parser.add_argument('--output', type=Path)
args = parser.parse_args()
if args.root is None:
    parser.error('Pass --root or set ATOD_TEACHERS_ROOT')
root = args.root.resolve()
manifest = json.loads((Path(__file__).resolve().parents[1] / 'configs/setup/teachers.json').read_text())
results = {}
for task, relative in manifest['teachers'].items():
    folder = root / relative
    shards = sorted(folder.glob("*.safetensors"))
    if len(shards) != 2:
        results[task] = {"complete": False}
        continue
    config = json.loads((folder / "config.json").read_text())
    mapping = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    found = {}
    dtypes = set()
    for shard in shards:
        with shard.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            assert length < 32 * 1024 * 1024
            header = json.loads(f.read(length))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            assert mapping[key] == shard.name
            assert info["data_offsets"][1] <= shard.stat().st_size - 8 - length
            found[key] = info
            dtypes.add(info["dtype"])
    assert set(mapping) == set(found)
    results[task] = {"complete": True, "path": str(folder), "architecture": config["architectures"], "tensor_count": len(found), "weight_dtypes": sorted(dtypes), "weight_bytes": sum(x.stat().st_size for x in shards)}
if args.output:
    with args.output.open('x') as stream:
        json.dump(results, stream, indent=2)
print(json.dumps(results, indent=2))
raise SystemExit(0 if all(result['complete'] for result in results.values()) else 1)
