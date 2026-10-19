"""Read-only dependency resolution using cached roots and exact PPU wheels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("environment", choices=["atod", "webshop"])
p.add_argument("--offline", action="store_true")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
logs = root / "logs/setup_20261016"
subprocess.run([sys.executable, str(root / "scripts/prepare_cached_requirements.py"), a.environment], check=True)
packages = json.loads((logs / f"{a.environment}_ppu_wheels.json").read_text())
cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--prefer-binary", "--progress-bar", "off", "-c", str(logs / f"{a.environment}_cached_constraints.txt"), "-r", str(logs / f"{a.environment}_cached_requirements.txt")]
cmd += [x["path"] for x in packages]
if a.offline:
    cmd += ["--no-index", "--find-links", "/mnt/data/home/zzr/cache/wheels", "--no-build-isolation"]
cmd += ["--report", str(logs / f"{a.environment}_resolution_plan.json")]
raise SystemExit(subprocess.run(cmd, cwd=root).returncode)
