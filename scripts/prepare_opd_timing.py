"""Prepare an OPD run in ATOD_BENCHMARK_DIR using local model directories.

Set ATOD_STUDENT_MODEL_PATH and ATOD_TEACHER_MODEL_PATH before running.
Task observations come from ALFWorld at rollout, not the offline control rows.
"""
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
os.environ['ATOD_PROJECT_ROOT'] = str(ROOT)
RUN = Path(os.environ['ATOD_BENCHMARK_DIR']).resolve()
RUN.mkdir(parents=True, exist_ok=True)
(RUN / 'inputs').mkdir(exist_ok=True)
# CPU preparation does not create a Ray session. Resolve its placeholder path
# locally; the launcher replaces it with a fresh IPC directory before training.
os.environ.setdefault('ATOD_RAY_IPC_DIR', str(RUN / 'runtime' / 'ray_ipc_placeholder'))
for split in ['train', 'val']:
    rows = [{'data_source': 'text', 'prompt': [{'role': 'user', 'content': ''}], 'ability': 'agent',
             'extra_info': {'split': split, 'index': i}} for i in range(4)]
    destination = RUN / 'inputs' / f'{split}.parquet'
    assert not destination.exists()
    pq.write_table(pa.Table.from_pylist(rows), destination)
with initialize_config_dir(config_dir=str(ROOT / 'configs/experiments'), version_base=None):
    config = compose(config_name='vanilla_opd_timing')
    OmegaConf.resolve(config)
OmegaConf.save(config, RUN / 'resolved_config.yaml')
assert config.trainer.n_gpus_per_node == config.ray_init.num_gpus
tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path, local_files_only=True)
teacher = AutoTokenizer.from_pretrained(config.actor_rollout_ref.ref.model.path, local_files_only=True)
assert tokenizer.get_vocab() == teacher.get_vocab()
prompt = 'Protocol check: <think>reasoning</think><action>action</action>'
a = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=True)
b = teacher.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
assert a == b and a.endswith('<|im_start|>assistant\n')
assert config.algorithm.sod.mode == 'uniform' and config.algorithm.sod.opd_only
assert config.actor_rollout_ref.actor.entropy_coeff == 0
assert config.data.max_response_length == 4096
assert config.data.apply_chat_template_kwargs.enable_thinking
result = {'offline_control_rows_per_split': 4, 'task_source': 'ALFWorld environment, not parquet text',
          'student_teacher_vocab_identical': True, 'rendered_template_identical': True,
          'full_tags_must_be_generated_by_model': True, 'mode': 'uniform OPD only',
          'max_generated_tokens_per_turn': 4096}
(RUN / 'cpu_preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
