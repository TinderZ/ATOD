"""Preserve a failed attempt and prepare a fresh directory for the same plan."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('previous', type=Path)
parser.add_argument('--reason', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
previous = args.previous.resolve()
assert previous.is_relative_to(root / 'outputs')
plan = json.loads((previous / 'plan.json').read_text())
attempt = int(plan.get('attempt_number', 2)) + 1
name = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
run = root / 'outputs' / f'{name}_vanilla_opd_timing_t32b_s06b_2ppu_4096_10updates_attempt{attempt:02d}'
run.mkdir()
for sub in ['inputs', 'source_before']:
    shutil.copytree(previous / sub, run / sub)
for sub in ['source_snapshot', 'checkpoints', 'rollouts', 'runtime']:
    (run / sub).mkdir()
shutil.copy2(previous / 'cpu_preflight.json', run / 'cpu_preflight.json')
plan.update(previous_attempt=str(previous), previous_attempt_result=args.reason, attempt_number=attempt,
            launch_command='bash /mnt/data/home/zzr/ATOD/scripts/run_vanilla_opd_timing.sh ' + str(run))
(run / 'plan.json').write_text(json.dumps(plan, indent=2))
(root / 'configs/setup/active_opd_benchmark.txt').write_text(str(run) + '\n')
print(run)
