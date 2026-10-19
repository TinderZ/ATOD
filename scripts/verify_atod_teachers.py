"""Inspect downloaded teacher layouts and safetensors headers without a GPU."""
import json
from pathlib import Path
import struct

root = Path("/mnt/data/home/zzr/models/ATOD_ckpt")
results = {}
for task in ["alfworld", "webshop", "search"]:
    folder = root / f"{task}_grpo_qwen3_4b/actor_hf"
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
output = Path(__file__).resolve().parents[1] / "logs/setup_20261016/teacher_layout_validation.json"
output.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
