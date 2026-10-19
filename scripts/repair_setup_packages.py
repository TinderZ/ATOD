"""Repair verified upstream packaging defects in these personal ATOD environments.

Decord's ctypes wheel advertises py3-none in its filename but cp36 in WHEEL.
TextWorld places a blank line before Tag, ending the wheel headers prematurely.
Fast Downward unlinks a loaded library on CPFS; tolerate deferred temp cleanup.
"""
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import sys

prefix = Path(sys.prefix)
assert prefix.parent == Path('/mnt/data/home/zzr/conda_envs')
assert prefix.name in ['atod', 'atod-webshop']
logs = Path(__file__).resolve().parents[1] / 'logs/setup_20261016'
report = logs / f'{prefix.name}_package_repairs.json'
changes = json.loads(report.read_text()) if report.exists() else []

def update(dist, path, new, reason):
    old = path.read_bytes()
    if old == new:
        return
    relative = path.relative_to(prefix / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages')
    backup = logs / f'{prefix.name}_{dist.metadata["Name"]}_{path.name}.original'
    if not backup.exists():
        backup.write_bytes(old)
    path.write_bytes(new)
    record = Path(dist._path) / 'RECORD'
    rows = list(csv.reader(io.StringIO(record.read_text())))
    row = next(row for row in rows if row[0] == str(relative))
    row[1] = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(new).digest()).decode().rstrip('=')
    row[2] = str(len(new))
    out = io.StringIO(newline='')
    csv.writer(out).writerows(rows)
    record.write_text(out.getvalue())
    changes.append({'package': dist.metadata['Name'], 'version': dist.version, 'file': str(path), 'backup': str(backup), 'reason': reason, 'before_sha256': hashlib.sha256(old).hexdigest(), 'after_sha256': hashlib.sha256(new).hexdigest()})
    print(reason, path, flush=True)

for name in ['decord', 'textworld', 'fast-downward-textworld']:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    if name == 'decord':
        assert dist.version == '0.6.0'
        path = Path(dist._path) / 'WHEEL'
        old = path.read_text()
        new = old.replace('Tag: cp36-cp36m-manylinux2010_x86_64', 'Tag: py3-none-manylinux2010_x86_64')
        update(dist, path, new.encode(), 'Match the Decord filename tag for its ctypes-only Python interface')
    elif name == 'textworld':
        assert dist.version == '1.7.0'
        path = Path(dist._path) / 'WHEEL'
        old = path.read_text()
        update(dist, path, old.replace('\n\nTag:', '\nTag:').encode(), 'Remove the premature end of TextWorld WHEEL headers')
    else:
        assert dist.version == '20.6.4'
        path = Path(dist.locate_file('fast_downward/interface.py'))
        old = path.read_text()
        update(dist, path, old.replace('with tempfile.TemporaryDirectory() as tmpdir:', 'with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:').encode(), 'Allow CPFS to defer cleanup of a mapped temporary planner library')
report.write_text(json.dumps(changes, indent=2))
