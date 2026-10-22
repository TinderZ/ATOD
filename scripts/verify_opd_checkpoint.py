"""Read and validate both FSDP checkpoint shards on CPU before resuming."""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import time
import zipfile
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Validation must be CPU-only"
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from torch.distributed.tensor import DTensor
    torch.set_num_threads(1)
    root = args.checkpoint.resolve()
    step = int(root.name.removeprefix("global_step_"))
    started = time.time()
    results = []
    config = AutoConfig.from_pretrained(root / "actor", local_files_only=True)
    with torch.device("meta"):
        expected = AutoModelForCausalLM.from_config(config).state_dict()

    def check_tensors(value, counters):
        if isinstance(value, DTensor):
            check_tensors(value.to_local(), counters)
        elif isinstance(value, torch.Tensor):
            assert value.device.type == "cpu"
            counters["tensors"] += 1
            counters["elements"] += value.numel()
            if value.is_floating_point():
                assert torch.isfinite(value).all().item(), "Non-finite checkpoint tensor"
        elif isinstance(value, dict):
            for item in value.values():
                check_tensors(item, counters)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_tensors(item, counters)

    for rank in range(2):
        for component in ("model", "optim", "extra_state"):
            path = root / "actor" / f"{component}_world_size_2_rank_{rank}.pt"
            tick = time.time()
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None, f"ZIP CRC failure: {path}"
            value = torch.load(path, map_location="cpu", weights_only=False)
            counters = dict(tensors=0, elements=0)
            check_tensors(value, counters)
            details = {}
            if component == "model":
                assert set(value) == set(expected), "Model state keys differ from model config"
                for key, tensor in value.items():
                    assert tensor.shape == expected[key].shape, f"Wrong shape: {key}"
                details["model_keys_and_global_shapes_match"] = True
            elif component == "optim":
                assert value["state"] and value["param_groups"]
                parameters = {p for group in value["param_groups"] for p in group["params"]}
                assert set(value["state"]) <= parameters
                steps = []
                for state in value["state"].values():
                    assert state["exp_avg"].shape == state["exp_avg_sq"].shape
                    steps.append(int(state["step"]))
                assert min(steps) >= step
                details.update(optimizer_parameters=len(parameters),
                               optimizer_step_range=[min(steps), max(steps)])
            else:
                assert value["lr_scheduler"]["last_epoch"] == step
                assert "cpu" in value["rng"] and "numpy" in value["rng"]
                details["scheduler_last_epoch"] = step
                details["rng_keys"] = sorted(value["rng"])
            result = dict(file=str(path), bytes=path.stat().st_size,
                          crc="passed", torch_load="passed", finite="passed",
                          seconds=time.time()-tick, **counters, **details)
            results.append(result)
            print(json.dumps(result), flush=True)
            del value
            gc.collect()
    data_path = root / "data.pt"
    with zipfile.ZipFile(data_path) as archive:
        assert archive.testzip() is None
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    assert isinstance(data, dict) and data["_num_yielded"] >= 0
    report = dict(passed=True, checkpoint=str(root), step=step, world_size=2,
                  verified_beijing=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  python=os.sys.executable, torch_version=torch.__version__,
                  elapsed_seconds=time.time()-started, files=results,
                  dataloader_state=data,
                  validation_scope="Full-file CRC, CPU deserialization, all floating tensors finite, model keys/shapes, optimizer and scheduler state. GPU application is checked during resume initialization.")
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("CHECKPOINT_CPU_VALIDATION_PASSED", args.report, flush=True)


if __name__ == "__main__":
    main()
