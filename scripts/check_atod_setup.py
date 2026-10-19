"""CPU-only installation smoke checks; never launches training or a retriever."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
p = argparse.ArgumentParser()
p.add_argument("mode", choices=["atod", "webshop", "retriever"])
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
modules = {
    "atod": ["torch", "torchvision", "transformers", "peft", "ray", "tensordict", "vllm", "vllm._C", "flash_attn", "datasets", "spacy", "alfworld", "skyrl_gym", "agent_system.environments.env_package.alfworld.envs"],
    "webshop": ["torch", "torchvision", "transformers", "ray", "tensordict", "vllm", "vllm._C", "flash_attn", "spacy", "faiss", "pyserini.search.lucene"],
    "retriever": ["torch", "transformers", "datasets", "fastapi", "uvicorn", "faiss"],
}[args.mode]
results = {}
for name in modules:
    try:
        m = importlib.import_module(name)
        results[name] = {"ok": True, "version": str(getattr(m, "__version__", ""))}
    except Exception as e:
        results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(name, results[name], flush=True)
try:
    import torch
    assert (torch.tensor([2., 3.]) ** 2).tolist() == [4., 9.]
    results["torch_cpu_operation"] = {"ok": True}
except Exception as e:
    results["torch_cpu_operation"] = {"ok": False, "error": str(e)}
if args.mode in ["webshop", "retriever"]:
    try:
        import faiss
        import numpy as np
        x = np.array([[0., 1.], [1., 0.]], dtype="float32")
        index = faiss.IndexFlatL2(2)
        index.add(x)
        assert index.search(x, 1)[1].tolist() == [[0], [1]]
        results["faiss_cpu_search"] = {"ok": True}
    except Exception as e:
        results["faiss_cpu_search"] = {"ok": False, "error": str(e)}
if args.mode == "webshop":
    try:
        sys.path.insert(0, str(root / "agent_system/environments/env_package/webshop/webshop"))
        importlib.import_module("web_agent_site.envs.web_agent_text_env")
        results["webshop_environment_import"] = {"ok": True}
    except Exception as e:
        results["webshop_environment_import"] = {"ok": False, "error": str(e)}
output = root / f"logs/setup_20261016/{args.mode}_validation.json"
output.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
raise SystemExit(0 if all(v["ok"] for v in results.values()) else 1)
