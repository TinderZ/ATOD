"""Preserve a failed attempt and prepare a fresh directory for the same plan."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import shlex
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('previous', type=Path)
parser.add_argument('--reason', required=True)
parser.add_argument('--output-root', type=Path, help='New attempt parent directory; defaults to the previous attempt parent')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
previous = args.previous.resolve()
plan = json.loads((previous / 'plan.json').read_text())
attempt = int(plan.get('attempt_number', 2)) + 1
name = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
output_root = (args.output_root or previous.parent).resolve()
output_root.mkdir(parents=True, exist_ok=True)
run = output_root / f'{name}_vanilla_opd_timing_attempt{attempt:02d}'
run.mkdir()
for sub in ['inputs', 'source_before']:
    if (previous / sub).is_dir():
        shutil.copytree(previous / sub, run / sub)
for sub in ['source_snapshot', 'checkpoints', 'rollouts', 'runtime']:
    (run / sub).mkdir()
shutil.copy2(previous / 'cpu_preflight.json', run / 'cpu_preflight.json')
plan.update(previous_attempt=str(previous), previous_attempt_result=args.reason, attempt_number=attempt,
            launch_command=shlex.join(['bash', str(root / 'scripts/run_vanilla_opd_timing.sh'), str(run)]))
(run / 'plan.json').write_text(json.dumps(plan, indent=2))
print(run)
