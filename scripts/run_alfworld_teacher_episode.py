"""One offline teacher-policy episode; --prepare-only never loads model weights."""
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
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LocalAlfworld:
    """Single-environment adapter for the project's existing environment manager."""
    def __init__(self, game):
        from textworld import EnvInfos
        from textworld.envs.pddl.pddl import PddlEnv
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
        self.env = AlfredInfos(AlfredDemangler(PddlEnv(EnvInfos(admissible_commands=True, won=True)), shuffle=False))
        self.game = game
        self.state = None

    @property
    def get_admissible_commands(self):
        return [list(self.state.admissible_commands)]

    def info(self):
        return {"won": bool(self.state.won), "admissible_commands": list(self.state.admissible_commands), "extra.gamefile": str(self.game)}

    def reset(self):
        self.env.load(str(self.game))
        self.state = self.env.reset()
        return [self.state.feedback], None, [self.info()]

    def step(self, actions):
        from agent_system.environments.env_package.alfworld.envs import compute_reward
        self.state, _, done = self.env.step(actions[0])
        info = self.info()
        return [self.state.feedback], None, [compute_reward(info)], [bool(done)], [info]

    def close(self):
        self.env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--model", type=Path, default=Path("/mnt/data/home/zzr/models/ATOD_ckpt/alfworld_grpo_qwen3_4b/actor_hf"))
    p.add_argument("--model-repo")
    p.add_argument("--model-revision")
    p.add_argument("--policy-label")
    p.add_argument("--game", type=Path, default=Path(os.environ["ALFWORLD_DATA"]) / "json_2.1.1/train/pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-417/trial_T20190908_185320_708158/game.tw-pddl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--enable-thinking", choices=["true", "false"], default="false")
    args = p.parse_args()
    assert 0 < args.max_steps <= 12 and 0 < args.max_new_tokens <= 1024
    assert 0 < args.max_input_tokens <= 2048
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert not (args.output_dir / "config.json").exists(), "Use a new output directory."
    if args.prepare_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection
    random.seed(args.seed)
    np.random.seed(args.seed)
    game = args.game.resolve()
    assert game.is_relative_to(Path(os.environ["ALFWORLD_DATA"]).resolve())
    local = LocalAlfworld(game)
    manager = AlfWorldEnvironmentManager(local, alfworld_projection, OmegaConf.create({"env": {"history_length": 2}}))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    is_atod_teacher = args.model.resolve() == Path("/mnt/data/home/zzr/models/ATOD_ckpt/alfworld_grpo_qwen3_4b/actor_hf").resolve()
    started = time.monotonic()
    trajectory = []
    config = {
        "created_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "environment": sys.prefix, "python": sys.version,
        "packages": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "textworld", "alfworld"]},
        "game": str(game), "game_sha256": hashlib.sha256(game.read_bytes()).hexdigest(),
        "split": "train", "policy": args.policy_label or ("qwen3_4b_grpo_teacher" if is_atod_teacher else "external_teacher"),
        "model": str(args.model.resolve()),
        "model_repo": args.model_repo or ("TanQT24/ATOD_ckpt" if is_atod_teacher else None),
        "teacher_revision": args.model_revision or ("fd6a12024f8eed189757dc9d770994b051f2fe27" if is_atod_teacher else None),
        "seed": args.seed, "max_steps": args.max_steps, "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens, "history_length": 2,
        "do_sample": False, "dtype": "bfloat16", "attention": "eager",
        "enable_thinking": args.enable_thinking == "true", "reference_actions_used": False,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "prepare_only": args.prepare_only,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    try:
        observations, _ = manager.reset({})

        def encode(prompt):
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=args.enable_thinking == "true")
            batch = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            length = batch["input_ids"].shape[1]
            assert length <= args.max_input_tokens, f"Prompt exceeds budget: {length}"
            return rendered, batch

        rendered, batch = encode(observations["text"][0])
        (args.output_dir / "initial_prompt.txt").write_text(rendered)
        print(f"Initial prompt tokens: {batch['input_ids'].shape[1]}", flush=True)
        if args.prepare_only:
            # Validate the same adapter/manager path with an arbitrary admissible action,
            # without reading the reference walkthrough or loading a language model.
            action = next(a for a in local.get_admissible_commands[0] if a != "help")
            next_obs, _, _, infos = manager.step([f"<think>CPU adapter check.</think><action>{action}</action>"])
            _, next_batch = encode(next_obs["text"][0])
            result = {"prepared": True, "model_loaded": False, "gpu_allocations": 0,
                      "initial_prompt_tokens": batch["input_ids"].shape[1],
                      "history_prompt_tokens": next_batch["input_ids"].shape[1],
                      "adapter_action_format_valid": bool(infos[0]["is_action_valid"])}
            assert result["adapter_action_format_valid"]
            (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            return

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        assert visible.isdecimal(), "Expose exactly one approved physical GPU."
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Approved GPU unavailable."
        probe = (torch.tensor([2., 3.], device="cuda") ** 2).cpu().tolist()
        assert probe == [4., 9.]
        bf16 = torch.ones((2, 2), dtype=torch.bfloat16, device="cuda")
        assert (bf16 @ bf16).float().cpu().tolist() == [[2., 2.], [2., 2.]]
        print(f"GPU arithmetic passed; loading teacher on physical device {visible}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager")
        model = model.to("cuda:0").eval()
        print(f"Teacher loaded in {time.monotonic() - started:.1f}s", flush=True)
        total_tokens, valid_actions, admissible_actions, format_valid_actions = 0, 0, 0, 0
        done, won, reward = False, False, 0.0
        for step in range(1, args.max_steps + 1):
            prompt = observations["text"][0]
            _, batch = encode(prompt)
            inputs = {k: v.to("cuda:0") for k, v in batch.items()}
            begin = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                        use_cache=True)
            generated = output[0, inputs["input_ids"].shape[1]:].detach().cpu()
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            parsed, flags = alfworld_projection([raw], local.get_admissible_commands)
            admissible = parsed[0] in local.get_admissible_commands[0]
            observations, rewards, dones, infos = manager.step([raw])
            reward, done, won = float(rewards[0]), bool(dones[0]), bool(infos[0]["won"])
            total_tokens += len(generated)
            valid_actions += int(bool(flags[0]) and admissible)
            admissible_actions += int(admissible)
            format_valid_actions += int(bool(flags[0]))
            record = {"step": step, "prompt": prompt, "response": raw, "action": parsed[0],
                      "format_valid": bool(flags[0]), "admissible": admissible,
                      "observation": observations["anchor"][0], "reward": reward,
                      "done": done, "won": won, "generated_tokens": len(generated),
                      "hit_token_budget": len(generated) >= args.max_new_tokens,
                      "ended_with_eos": bool(len(generated) and generated[-1].item() == tokenizer.eos_token_id),
                      "input_tokens": inputs["input_ids"].shape[1],
                      "turn_seconds": time.monotonic() - begin}
            trajectory.append(record)
            with (args.output_dir / "trajectory.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"TURN {step}: action={parsed[0]!r} format_valid={bool(flags[0])} admissible={admissible} reward={reward} won={won}", flush=True)
            if done:
                break
        result = {"pipeline_completed": True, "task_success": won, "done": done,
                  "steps": len(trajectory), "atod_reward": reward,
                  "valid_action_rate": valid_actions / len(trajectory), "generated_tokens": total_tokens,
                  "admissible_action_rate": admissible_actions / len(trajectory),
                  "response_format_valid_rate": format_valid_actions / len(trajectory),
                  "elapsed_seconds": time.monotonic() - started,
                  "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
        print("RESULT " + json.dumps(result), flush=True)
    except Exception:
        (args.output_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        local.close()


if __name__ == "__main__":
    main()
