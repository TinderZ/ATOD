"""Prefer already-downloaded exact root versions without changing their specs."""
import argparse
from pathlib import Path
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

p = argparse.ArgumentParser()
p.add_argument("environment", choices=["atod", "webshop"])
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
cache = Path("/mnt/data/home/zzr/cache/wheels")
tags = set(sys_tags())
wheels = []
for f in cache.glob("*.whl"):
    try:
        name, version, _, wheel_tags = parse_wheel_filename(f.name)
        if tags.intersection(wheel_tags):
            wheels.append((canonicalize_name(name), version, f))
    except ValueError:
        pass
inputs = ["requirements-atod.txt"] if args.environment == "atod" else ["requirements-webshop-training.txt", "requirements-webshop.txt"]
output = []
for filename in inputs:
    for line in (root / "configs/setup" / filename).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        req = Requirement(line)
        exact = any(s.operator in ["==", "==="] and "*" not in s.version for s in req.specifier)
        matches = [(v, f) for name, v, f in wheels if name == canonicalize_name(req.name) and v in req.specifier and not v.local]
        if exact and matches and canonicalize_name(req.name) not in {"fastapi", "pydantic"}:
            _, f = max(matches, key=lambda x: x[0])
            extra = "[" + ",".join(sorted(req.extras)) + "]" if req.extras else ""
            marker = f"; {req.marker}" if req.marker else ""
            output.append(f"{req.name}{extra} @ {f.as_uri()}{marker}")
        else:
            output.append(line)
dest = root / f"logs/setup_20261016/{args.environment}_cached_requirements.txt"
dest.write_text("\n".join(output) + "\n")
print(dest)
constraints = []
for line in (root / f"configs/setup/constraints-{args.environment}.txt").read_text().splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    req = Requirement(line)
    matches = [(v, f) for name, v, f in wheels if name == canonicalize_name(req.name) and v in req.specifier]
    if matches and canonicalize_name(req.name) not in {"fastapi", "pydantic", "typer", "fastapi-cli", "starlette"}:
        _, f = max(matches, key=lambda x: x[0])
        constraints.append(f"{req.name} @ {f.as_uri()}")
    else:
        constraints.append(line)
(root / f"logs/setup_20261016/{args.environment}_cached_constraints.txt").write_text("\n".join(constraints) + "\n")
