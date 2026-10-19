"""Summarize measured OPD updates, phases and native device samples (offline)."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), total=sum(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    records = rows(run / 'metrics.jsonl')
    updates = [r for r in records if 'timing_s/step' in r['metrics']]
    samples = rows(run / 'ppu_samples.jsonl')
    events = rows(run / 'phase_events.jsonl')
    phases = sorted({e['phase'] for e in events})
    phase_summary = {}
    for phase in phases:
        selected = [e for e in events if e['phase'] == phase]
        phase_summary[phase] = dict(
            seconds=stats([e['finished'] - e['started'] for e in selected]),
            peak_sampled_mib={str(d): max((device['memory_used_mib']
                for sample in samples if any(e['started'] <= sample['wall_time'] <= e['finished'] for e in selected)
                for device in sample.get('devices', []) if device['physical_device'] == d), default=None)
                for d in (0, 1)})
    flattened = [dict(step=r['step'], **r['metrics']) for r in updates]
    if flattened:
        with (run / 'update_metrics.csv').open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=['step'] + sorted(set().union(*(r.keys() for r in flattened)) - {'step'}))
            writer.writeheader()
            writer.writerows(flattened)
    steady = [r['metrics']['timing_s/step'] - r['metrics'].get('timing_s/save_checkpoint', 0)
              for r in updates if r['step'] > 1]
    measured_update = stats(steady)
    recent = [r['metrics']['timing_s/step'] - r['metrics'].get('timing_s/save_checkpoint', 0)
              for r in updates[-3:]]
    recent_update = stats(recent)
    loss_keys = sorted({k for r in updates for k in r['metrics'] if 'loss' in k})
    protocol = []
    for file in sorted((run / 'rollouts').glob('*.jsonl'), key=lambda p: int(p.stem)):
        generations = rows(file)
        outputs = [r['output'] for r in generations]
        strict = lambda s: bool(re.fullmatch(r'\s*<think>.*?</think>\s*<action>.*?</action>\s*', s, re.S)) and all(
            s.count(tag) == 1 for tag in ('<think>', '</think>', '<action>', '</action>'))
        protocol.append(dict(step=int(file.stem), turn_rows=len(outputs),
            strict_single_think_action=sum(strict(s) for s in outputs),
            contains_reasoning_tags=sum('<reasoning>' in s or '</reasoning>' in s for s in outputs),
            repeated_think_close=sum(s.count('</think>') > 1 for s in outputs),
            missing_required_tag=sum(any(tag not in s for tag in ('<think>', '</think>', '<action>', '</action>')) for s in outputs),
            contains_chinese=sum(bool(re.search(r'[\u4e00-\u9fff]', s)) for s in outputs)))
    summary = dict(
        run=str(run), completed_global_updates=len(updates),
        completed_steps=[r['step'] for r in updates],
        all_logged_losses_finite=(all(math.isfinite(float(r['metrics'][k]))
            for r in updates for k in loss_keys if k in r['metrics']) if updates and loss_keys else None),
        losses={k: stats([r['metrics'].get(k) for r in updates]) for k in loss_keys},
        protocol_by_update=protocol,
        turn_rows=sum(r['metrics'].get('benchmark/turn_rows', 0) for r in updates),
        response_tokens_approx=sum(round(r['metrics'].get('response_length/mean', 0) * r['metrics'].get('benchmark/turn_rows', 0)) for r in updates),
        phases=phase_summary,
        steady_step_seconds_excluding_checkpoint=measured_update,
        recent_3_step_seconds_excluding_checkpoint=recent_update,
        peak_sampled_mib={str(d): max((device['memory_used_mib'] for sample in samples
            for device in sample.get('devices', []) if device['physical_device'] == d), default=None) for d in (0, 1)},
        same_batch_2card_estimate_hours={str(n): measured_update['mean'] * n / 3600
            for n in (100, 500, 1000, 2000)} if measured_update else {},
        same_batch_2card_recent_estimate_hours={str(n): recent_update['mean'] * n / 3600
            for n in (100, 500, 1000, 2000)} if recent_update else {},
        estimate_caution='These are conditional projections at the measured output lengths, not upper bounds. Outputs grew throughout this run.',
        estimates_exclude_initialization_validation_and_checkpoint=True,
        device_sample_period_seconds=5,
    )
    for name in ('initialization', 'entry_status'):
        if (run / (name + '.json')).exists():
            summary[name] = json.loads((run / (name + '.json')).read_text())
    (run / 'timing_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.plot and updates:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = [r['step'] for r in updates]
        metrics = [r['metrics'] for r in updates]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        bottom = [0.] * len(x)
        for key, label, color in [('gen', 'Student rollout + environment', '#4c78a8'),
                                  ('teacher_forward', 'Teacher scoring', '#f2a541'),
                                  ('old_log_prob', 'Student log-prob', '#72b7b2'),
                                  ('update_actor', 'Student update', '#54a24b'),
                                  ('save_checkpoint', 'Checkpoint', '#b279a2')]:
            values = [m.get('timing_s/' + key, 0) / 60 for m in metrics]
            axes[0].bar(x, values, bottom=bottom, color=color, label=label)
            bottom = [a + b for a, b in zip(bottom, values)]
        axes[0].set_ylabel('Minutes / update')
        axes[0].legend(fontsize=8, ncol=2, frameon=False, loc='upper left')
        axes[0].set_ylim(0, max(m['timing_s/step'] / 60 for m in metrics) * 1.3)
        axes[1].plot(x, [m['actor/pg_loss'] for m in metrics], 'o-', color='#4c78a8')
        axes[1].set_ylabel('Logged policy loss')
        axes[2].plot(x, [m['response_length/mean'] for m in metrics], 'o-', color='#4c78a8', label='Mean response tokens')
        axes[2].set_ylabel('Tokens / turn', color='#4c78a8')
        cap = axes[2].twinx()
        cap.plot(x, [100 * m['response_length/clip_ratio'] for m in metrics], 's--', color='#e45756', label='Turns reaching 4096')
        cap.set_ylabel('Turns reaching token cap (%)', color='#e45756')
        cap.set_ylim(0, 30)
        axes[2].set_xlabel('Global OPD update (4 trajectories per update; at most 20 turns each)')
        axes[2].set_xticks(x)
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle('Qwen3-32B teacher to Qwen3-0.6B student | 2 PPU-ZW810E\nVanilla OPD, thinking=True, full tags, 4096 tokens/turn', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(run / 'timing_curves.png', dpi=160)
        fig.savefig(run / 'timing_curves.svg')
        plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
