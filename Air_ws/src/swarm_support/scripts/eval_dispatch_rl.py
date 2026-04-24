#!/usr/bin/env python3
"""Offline evaluation helper for RL dispatch policies."""

from __future__ import annotations

import argparse
import json

from dispatch_dataset import tensorize_observation
from dispatch_env import SyntheticDispatchGenerator
from dispatch_expert import VRPTWExpertPlanner
from dispatch_model import load_policy_checkpoint
from dispatch_runtime import evaluate_order_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RL dispatch model on synthetic hold-out tasks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--ugv-num", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--v-max", type=float, default=3.0)
    parser.add_argument("--a-max", type=float, default=1.0)
    parser.add_argument("--output-json", default="", help="Optional path to save evaluation metrics as JSON.")
    args = parser.parse_args()

    model = load_policy_checkpoint(args.checkpoint, device=args.device)
    generator = SyntheticDispatchGenerator(
        ugv_num=args.ugv_num,
        seed=args.seed,
        v_max=args.v_max,
        a_max=args.a_max,
    )
    expert = VRPTWExpertPlanner(
        ugv_num=args.ugv_num,
        uav_num=1,
        v_max=args.v_max,
        a_max=args.a_max,
    )

    rl_success = 0
    expert_success = 0
    for _ in range(args.episodes):
        scenario = generator.sample()
        observation = scenario.observation
        batch = tensorize_observation(observation, device=args.device)
        output = model.greedy_order(batch)
        active_count = int(observation["active_count"])
        rl_order = output["actions"][0, :active_count].detach().cpu().tolist()
        rl_metrics = evaluate_order_metrics(rl_order, observation)
        expert_solution = expert.solve(scenario.tasks, scenario.self_pos, scenario.self_vel, now_sec=0.0)
        expert_metrics = evaluate_order_metrics(expert_solution.slot_order, observation)
        rl_success += rl_metrics.success_count
        expert_success += expert_metrics.success_count

    rl_rate = rl_success / max(args.episodes, 1)
    expert_rate = expert_success / max(args.episodes, 1)
    ratio = rl_rate / expert_rate if expert_rate > 1e-6 else 0.0
    metrics = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "ugv_num": args.ugv_num,
        "seed": args.seed,
        "device": args.device,
        "rl_success_total": rl_success,
        "expert_success_total": expert_success,
        "rl_success_per_episode": rl_rate,
        "expert_success_per_episode": expert_rate,
        "rl_vs_expert_ratio": ratio,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=True)
            fh.write("\n")
    print(f"rl_success_per_episode={rl_rate:.4f}")
    print(f"expert_success_per_episode={expert_rate:.4f}")
    print(f"rl_vs_expert_ratio={ratio:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
