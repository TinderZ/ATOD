"""Replay one reference episode to check ALFWorld observations and rewards on CPU."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textworld import EnvInfos
from textworld.envs.pddl.pddl import PddlEnv
from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
from agent_system.environments.env_package.alfworld.envs import compute_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, default=Path(os.environ["ALFWORLD_DATA"]) / "json_2.1.1/train/pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-417/trial_T20190908_185320_708158/game.tw-pddl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = ROOT / "outputs" / f"{stamp}_alfworld_cpu_walkthrough_seed{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output: {args.output_dir.resolve()}", flush=True)
    random.seed(args.seed)
    game = args.game.resolve()
    assert game.is_relative_to(Path(os.environ["ALFWORLD_DATA"]).resolve())
    data = json.loads(game.read_text())
    actions = data["walkthrough"]
    assert data["solvable"] and actions
    started = time.monotonic()
    metadata = {
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "python": sys.version, "environment": sys.prefix,
        "packages": {name: importlib.metadata.version(name) for name in ["textworld", "alfworld", "torch"]},
        "game": str(game), "game_sha256": hashlib.sha256(game.read_bytes()).hexdigest(),
        "split": "train", "policy": "dataset_reference_walkthrough", "model": None,
        "seed": args.seed, "gpu_count": 0, "output_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    env = AlfredInfos(AlfredDemangler(PddlEnv(EnvInfos(admissible_commands=True, won=True)), shuffle=False))
    history = []
    try:
        env.load(str(game))
        state = env.reset()
        print("INITIAL OBSERVATION\n" + state.feedback, flush=True)
        history.append({"step": 0, "observation": state.feedback, "admissible_commands": state.admissible_commands})
        for step, action in enumerate(actions, 1):
            assert action in state.admissible_commands, (step, action, state.admissible_commands)
            state, raw_reward, done = env.step(action)
            reward = compute_reward({"won": state.won}, multi_modal=False)
            history.append({"step": step, "action": action, "observation": state.feedback,
                            "raw_reward": raw_reward, "atod_reward": reward, "done": done, "won": state.won})
            print(f"\nSTEP {step}: {action}\n{state.feedback}\nreward={reward}, done={done}, won={state.won}", flush=True)
            if done:
                break
        result = {"passed": bool(state.won and done and reward == 10.0), "won": bool(state.won),
                  "done": bool(done), "steps": len(history) - 1, "raw_reward": raw_reward,
                  "atod_reward": reward, "elapsed_seconds": time.monotonic() - started,
                  "policy": "dataset_reference_walkthrough", "model": None, "gpu_count": 0}
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
        print("\nRESULT " + json.dumps(result), flush=True)
        assert result["passed"], "The reference episode did not reach success."
    finally:
        (args.output_dir / "trajectory.json").write_text(json.dumps(history, indent=2))
        env.close()


if __name__ == "__main__":
    main()
