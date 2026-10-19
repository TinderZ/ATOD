"""Reset and step one public ALFWorld game on CPU, without changing the dataset."""
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from textworld import EnvInfos
from textworld.envs.pddl.pddl import PddlEnv

root = Path(__file__).resolve().parents[1]
data = Path(os.environ['ALFWORLD_DATA']) / 'json_2.1.1/train'
game = next(data.glob('*/*/game.tw-pddl')).resolve()
assert str(game).startswith(str(data.resolve()) + '/')
env = PddlEnv(EnvInfos(admissible_commands=True, won=True))
try:
    env.load(str(game))
    state = env.reset()
    assert isinstance(state.feedback, str) and state.feedback
    assert state.admissible_commands
    count = len(state.admissible_commands)
    action = 'look' if 'look' in state.admissible_commands else state.admissible_commands[0]
    observation, reward, done = env.step(action)
    assert isinstance(observation.feedback, str) and observation.feedback
    result = {'ok': True, 'game': str(game), 'game_sha256': hashlib.sha256(game.read_bytes()).hexdigest(), 'admissible_commands': count, 'action': action, 'reward': reward, 'done': done}
finally:
    env.close()
(root / 'logs/setup_20261016/alfworld_cpu_game.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
