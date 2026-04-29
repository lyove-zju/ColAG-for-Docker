#!/usr/bin/env python3
"""Offline evaluation helper for RL dispatch policies."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from dispatch_dataset import tensorize_observation
from dispatch_env import EventDrivenDispatchEnv, SyntheticDispatchGenerator
from dispatch_expert import VRPTWExpertPlanner
from dispatch_model import load_policy_checkpoint
from dispatch_runtime import DEFAULT_FLY_HEIGHT, DispatchTask, evaluate_order_metrics


def evaluate_bandit(args: argparse.Namespace) -> Dict[str, object]:
    model = load_policy_checkpoint(args.checkpoint, device=args.device)
    generator = SyntheticDispatchGenerator(
        ugv_num=args.ugv_num,
        seed=args.seed,
        map_size_x=args.map_size_x,
        map_size_y=args.map_size_y,
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
    return {
        "checkpoint": args.checkpoint,
        "eval_mode": "bandit",
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


def _tasks_from_observation(observation: Dict[str, object]) -> List[DispatchTask]:
    tasks: List[DispatchTask] = []
    active_mask = observation["active_mask"]
    node_features = observation["node_features"]
    for slot, enabled in enumerate(active_mask):
        if not enabled:
            continue
        features = node_features[slot]
        tasks.append(
            DispatchTask(
                ugv_id=int(features[1]),
                collision_x=float(features[2]),
                collision_y=float(features[3]),
                collision_z=DEFAULT_FLY_HEIGHT,
                deadline_sec=float(features[7]),
            )
        )
    return tasks


def _event_policy_episode(args: argparse.Namespace, model, expert: VRPTWExpertPlanner, seed: int, policy: str):
    env = EventDrivenDispatchEnv(
        ugv_num=args.ugv_num,
        seed=seed,
        map_size_x=args.map_size_x,
        map_size_y=args.map_size_y,
        v_max=args.v_max,
        a_max=args.a_max,
        horizon_sec=args.horizon_sec,
        replan_period=args.replan_period,
        arrival_radius=args.arrival_radius,
        min_event_interval=args.min_event_interval,
        max_event_interval=args.max_event_interval,
        min_slack_factor=args.min_slack_factor,
        max_slack_factor=args.max_slack_factor,
        max_extra_deadline=args.max_extra_deadline,
    )
    observation = env.reset()
    done = False
    info = {}
    decisions = 0
    while not done and decisions < args.max_decisions_per_episode:
        if policy == "rl":
            batch = tensorize_observation(observation, device=args.device)
            output = model.greedy_order(batch)
            active_count = int(observation["active_count"])
            order = output["actions"][0, :active_count].detach().cpu().tolist()
        else:
            solution = expert.solve(
                _tasks_from_observation(observation),
                observation["self_pos"],
                observation["self_vel"],
                now_sec=0.0,
            )
            order = solution.slot_order

        observation, _, done, info = env.step(order)
        decisions += 1
        if observation is None:
            done = True

    info["decisions"] = float(decisions)
    return info


def evaluate_event(args: argparse.Namespace) -> Dict[str, object]:
    model = load_policy_checkpoint(args.checkpoint, device=args.device)
    expert = VRPTWExpertPlanner(
        ugv_num=args.ugv_num,
        uav_num=1,
        v_max=args.v_max,
        a_max=args.a_max,
    )

    rl_totals = {
        "success_count": 0.0,
        "missed_count": 0.0,
        "tardiness": 0.0,
        "response_time": 0.0,
        "response_count": 0.0,
        "flight_distance": 0.0,
        "flight_time": 0.0,
        "route_churn": 0.0,
        "decisions": 0.0,
    }
    expert_totals = dict(rl_totals)

    for episode_idx in range(args.episodes):
        episode_seed = args.seed + episode_idx
        for totals, policy in ((rl_totals, "rl"), (expert_totals, "expert")):
            info = _event_policy_episode(args, model, expert, episode_seed, policy)
            for key in totals:
                totals[key] += float(info.get(key, 0.0))

    def prefixed(prefix: str, totals: Dict[str, float]) -> Dict[str, float]:
        total_tasks = totals["success_count"] + totals["missed_count"]
        hit_rate = totals["success_count"] / max(total_tasks, 1.0)
        return {
            f"{prefix}_success_total": totals["success_count"],
            f"{prefix}_missed_total": totals["missed_count"],
            f"{prefix}_deadline_hit_rate": hit_rate,
            f"{prefix}_success_per_episode": totals["success_count"] / max(args.episodes, 1),
            f"{prefix}_missed_per_episode": totals["missed_count"] / max(args.episodes, 1),
            f"{prefix}_avg_tardiness_seconds": totals["tardiness"] / max(totals["missed_count"], 1.0),
            f"{prefix}_avg_response_time_seconds": totals["response_time"] / max(totals["response_count"], 1.0),
            f"{prefix}_flight_distance_per_episode": totals["flight_distance"] / max(args.episodes, 1),
            f"{prefix}_flight_time_per_episode": totals["flight_time"] / max(args.episodes, 1),
            f"{prefix}_route_churn_per_episode": totals["route_churn"] / max(args.episodes, 1),
            f"{prefix}_decisions_per_episode": totals["decisions"] / max(args.episodes, 1),
        }

    rl_success_rate = rl_totals["success_count"] / max(args.episodes, 1)
    expert_success_rate = expert_totals["success_count"] / max(args.episodes, 1)
    metrics: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "eval_mode": "event",
        "episodes": args.episodes,
        "ugv_num": args.ugv_num,
        "seed": args.seed,
        "device": args.device,
        "rl_vs_expert_ratio": rl_success_rate / expert_success_rate if expert_success_rate > 1e-6 else 0.0,
    }
    metrics.update(prefixed("rl", rl_totals))
    metrics.update(prefixed("expert", expert_totals))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RL dispatch model on synthetic hold-out tasks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--ugv-num", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-mode", choices=["bandit", "event"], default="bandit")
    parser.add_argument("--map-size-x", type=float, default=35.0)
    parser.add_argument("--map-size-y", type=float, default=35.0)
    parser.add_argument("--v-max", type=float, default=3.0)
    parser.add_argument("--a-max", type=float, default=1.0)
    parser.add_argument("--horizon-sec", type=float, default=150.0)
    parser.add_argument("--replan-period", type=float, default=10.0)
    parser.add_argument("--arrival-radius", type=float, default=0.8)
    parser.add_argument("--min-event-interval", type=float, default=1.0)
    parser.add_argument("--max-event-interval", type=float, default=4.0)
    parser.add_argument("--min-slack-factor", type=float, default=1.1)
    parser.add_argument("--max-slack-factor", type=float, default=2.6)
    parser.add_argument("--max-extra-deadline", type=float, default=3.0)
    parser.add_argument("--max-decisions-per-episode", type=int, default=80)
    parser.add_argument("--output-json", default="", help="Optional path to save evaluation metrics as JSON.")
    args = parser.parse_args()

    if args.eval_mode == "event":
        metrics = evaluate_event(args)
    else:
        metrics = evaluate_bandit(args)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=True)
            fh.write("\n")
    print(f"eval_mode={metrics['eval_mode']}")
    print(f"rl_success_per_episode={float(metrics['rl_success_per_episode']):.4f}")
    print(f"expert_success_per_episode={float(metrics['expert_success_per_episode']):.4f}")
    print(f"rl_vs_expert_ratio={float(metrics['rl_vs_expert_ratio']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
