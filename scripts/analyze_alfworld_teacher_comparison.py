"""Read-only trajectory analysis; writes diagnostics into a new experiment dir."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'outputs/20261017_100746_408769649_alfworld_teacher_qwen3_4b_greedy_seed0_thinking_true_tokens1024'
STRICT = re.compile(r'\s*<think>.*?</think>\s*<action>[^<>]*</action>\s*', re.DOTALL)
TAGS = ['<think>', '</think>', '<action>', '</action>']


def analyze(path):
    config = json.loads((path / 'config.json').read_text())
    trajectory_path = path / 'trajectory.jsonl'
    rows = [json.loads(line) for line in trajectory_path.read_text().splitlines()] if trajectory_path.exists() else []
    result_path = path / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    turns = []
    for row in rows:
        response = row['response']
        counts = {tag: response.count(tag) for tag in TAGS}
        turns.append({'step': row['step'], 'generated_tokens': row['generated_tokens'],
                      'action': row['action'], 'parser_valid': row['format_valid'],
                      'admissible': row['admissible'],
                      'strict_protocol': bool(STRICT.fullmatch(response)) and all(n == 1 for n in counts.values()) and
                                         set(re.findall(r'</?([A-Za-z][A-Za-z0-9_-]*)\s*>', response)) <= {'think', 'action'},
                      'hit_token_budget': row['generated_tokens'] >= config['max_new_tokens'],
                      'ended_with_eos': row.get('ended_with_eos'),
                      'tag_counts': counts,
                      'reasoning_tag': '<reasoning>' in response or '</reasoning>' in response,
                      'won': row['won']})
    summary = {'directory': str(path), 'model': config['model'], 'model_repo': config.get('model_repo'),
               'policy': config['policy'], 'revision': config['teacher_revision'],
               'pipeline_completed': bool(result and result['pipeline_completed']),
               'exit_code': int((path / 'exit_code.txt').read_text()),
               'task_success': result['task_success'] if result else None,
               'steps': len(rows), 'generated_tokens': sum(t['generated_tokens'] for t in turns),
               'parser_valid_turns': sum(t['parser_valid'] for t in turns),
               'strict_protocol_turns': sum(t['strict_protocol'] for t in turns),
               'admissible_turns': sum(t['admissible'] for t in turns),
               'budget_hit_turns': [t['step'] for t in turns if t['hit_token_budget']],
               'reasoning_tag_turns': [t['step'] for t in turns if t['reasoning_tag']],
               'repeated_think_closing_turns': [t['step'] for t in turns if t['tag_counts']['</think>'] > 1],
               'elapsed_seconds': result['elapsed_seconds'] if result else None,
               'peak_gpu_reserved_gib': result['peak_gpu_reserved_gib'] if result else None,
               'turns': turns}
    if result:
        assert summary['generated_tokens'] == result['generated_tokens']
    return config, rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    configs, rows, summaries = [], [], []
    for path in [BASELINE, args.run_dir]:
        c, r, s = analyze(path)
        configs.append(c)
        rows.append(r)
        summaries.append(s)
    keys = ['branch', 'commit', 'environment', 'python', 'packages', 'game', 'game_sha256',
            'split', 'seed', 'max_steps', 'max_new_tokens', 'max_input_tokens', 'history_length',
            'do_sample', 'dtype', 'attention', 'reference_actions_used', 'CUDA_VISIBLE_DEVICES', 'enable_thinking']
    checks = {k: configs[0][k] == configs[1][k] for k in keys}
    assert all(checks.values())
    prompt_equal = (BASELINE / 'initial_prompt.txt').read_bytes() == (args.run_dir / 'initial_prompt.txt').read_bytes()
    assert prompt_equal
    comparison = {'controlled_config_checks': checks, 'initial_rendered_prompt_identical': prompt_equal,
                  'runs': summaries,
                  'strict_protocol_definition': 'Exactly one think block followed by one action block, no other text except whitespace. Diagnostic only; original parser and execution unchanged.',
                  'limitations': 'Official Qwen3-32B versus ALFWorld-GRPO Qwen3-4B: model scale and training differ; one train task with greedy decoding cannot establish general teacher quality. Official Qwen recommends sampling in thinking mode.'}
    with (args.run_dir / 'teacher_comparison.json').open('x') as f:
        json.dump(comparison, f, indent=2)
    text = []
    for config, trajectory in zip(configs, rows):
        text.extend(['=' * 80, config['policy'], config['model']])
        for row in trajectory:
            text.extend(['', f"--- TURN {row['step']}, generated_tokens={row['generated_tokens']}, parser_valid={row['format_valid']}, admissible={row['admissible']} ---", row['response']])
    with (args.run_dir / 'raw_teacher_outputs.txt').open('x') as f:
        f.write('\n'.join(text) + '\n')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
