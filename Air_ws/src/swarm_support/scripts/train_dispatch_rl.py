#!/usr/bin/env python3
"""Behavior cloning + PPO trainer for UAV dispatch."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict

from dispatch_dataset import (
    DispatchDataset,
    collate_dispatch_batch,
    permute_observation_slots,
    random_active_slot_permutation,
    require_torch,
    unpermute_order_slots,
)
from dispatch_env import DispatchBanditEnv, EventDrivenDispatchEnv, SyntheticDispatchGenerator
from dispatch_model import DispatchPointerPolicy, build_policy_from_config, save_checkpoint
from dispatch_runtime import evaluate_order_metrics


def _move_batch(batch: Dict[str, "torch.Tensor"], device: str) -> Dict[str, "torch.Tensor"]:
    return {key: value.to(device) for key, value in batch.items()}


def _observation_to_batch(observation: Dict[str, object], device: str) -> Dict[str, "torch.Tensor"]:
    require_torch()
    import torch

    return {
        "global_features": torch.tensor([observation["global_features"]], dtype=torch.float32, device=device),
        "node_features": torch.tensor([observation["node_features"]], dtype=torch.float32, device=device),
        "pairwise_time_matrix": torch.tensor([observation["pairwise_time_matrix"]], dtype=torch.float32, device=device),
        "active_mask": torch.tensor([observation["active_mask"]], dtype=torch.bool, device=device),
    }


def _augment_observation_slots(observation: Dict[str, object], rng: random.Random):
    permutation = random_active_slot_permutation(observation["active_mask"], rng)
    return permute_observation_slots(observation, permutation), permutation


def _reset_metrics_file(path: str) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


def _append_metrics(path: str, payload: Dict[str, object]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=True))
        fh.write("\n")


def _make_model(args: argparse.Namespace):
    return DispatchPointerPolicy(ugv_num=args.ugv_num, hidden_dim=args.hidden_dim)


def _load_or_make_model(args: argparse.Namespace, torch_module):
    if not args.init_checkpoint:
        return _make_model(args)
    payload = torch_module.load(args.init_checkpoint, map_location=args.device)
    config = payload.get("config", {})
    legacy_checkpoint = "use_ugv_id_feature" not in config
    model = build_policy_from_config(config, legacy_checkpoint=legacy_checkpoint)
    model.load_state_dict(payload["state_dict"], strict=not legacy_checkpoint)
    return model


def train_bc(args: argparse.Namespace) -> int:
    require_torch()
    import torch
    from torch.utils.data import DataLoader

    dataset = DispatchDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dispatch_batch)
    model = _make_model(args)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _reset_metrics_file(args.metrics_jsonl)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_steps = 0.0
        for batch in loader:
            batch = _move_batch(batch, args.device)
            output = model(batch, actions=batch["expert_order"])
            log_probs = output["log_probs"]
            step_mask = batch["step_mask"]
            loss = -((log_probs * step_mask).sum() / step_mask.sum().clamp_min(1.0))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * float(step_mask.sum().item())
            total_steps += float(step_mask.sum().item())
        avg_loss = total_loss / max(total_steps, 1.0)
        print(f"[BC] epoch={epoch} avg_nll={avg_loss:.6f}")
        _append_metrics(
            args.metrics_jsonl,
            {
                "phase": "bc",
                "epoch": epoch,
                "avg_nll": avg_loss,
                "total_steps": total_steps,
                "lr": args.lr,
            },
        )

    save_checkpoint(args.output, model)
    print(f"[BC] saved checkpoint to {args.output}")
    return 0


def train_ppo(args: argparse.Namespace) -> int:
    require_torch()
    import torch

    model = _load_or_make_model(args, torch)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _reset_metrics_file(args.metrics_jsonl)

    env = DispatchBanditEnv(
        SyntheticDispatchGenerator(
            ugv_num=args.ugv_num,
            seed=args.seed,
            map_size_x=args.map_size_x,
            map_size_y=args.map_size_y,
            v_max=args.v_max,
            a_max=args.a_max,
        )
    )
    slot_rng = random.Random(args.seed + 1000003)

    for update_idx in range(args.updates):
        rollout = []
        for _ in range(args.batch_size):
            observation = env.reset()
            policy_observation, permutation = _augment_observation_slots(observation, slot_rng)
            batch = _observation_to_batch(policy_observation, args.device)
            output = model(batch, greedy=False)
            active_count = int(policy_observation["active_count"])
            actions = output["actions"][0, :active_count]
            log_prob = output["log_probs"][0, :active_count].sum()
            value = output["values"][0]
            env_actions = unpermute_order_slots(actions.detach().cpu().tolist(), permutation)
            metrics = evaluate_order_metrics(env_actions, observation)
            reward = torch.tensor(metrics.reward, dtype=torch.float32, device=args.device)
            rollout.append(
                {
                    "batch": batch,
                    "actions": actions.detach(),
                    "old_log_prob": log_prob.detach(),
                    "reward": reward,
                    "value": value.detach(),
                    "success_count": metrics.success_count,
                    "missed_count": metrics.missed_count,
                    "tardiness": metrics.tardiness,
                    "route_time": metrics.route_time,
                }
            )

        loss_values = []
        policy_loss_values = []
        value_loss_values = []
        entropy_values = []
        for _ in range(args.ppo_epochs):
            policy_loss_total = torch.tensor(0.0, device=args.device)
            value_loss_total = torch.tensor(0.0, device=args.device)
            entropy_total = torch.tensor(0.0, device=args.device)
            for item in rollout:
                current = model(item["batch"], actions=item["actions"].unsqueeze(0))
                new_log_prob = current["log_probs"][0, : item["actions"].numel()].sum()
                value = current["values"][0]
                advantage = item["reward"] - item["value"]
                ratio = torch.exp(new_log_prob - item["old_log_prob"])
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * advantage
                policy_loss_total = policy_loss_total - torch.min(unclipped, clipped)
                value_loss_total = value_loss_total + 0.5 * torch.square(value - item["reward"])
                entropy_total = entropy_total + current["entropy"][0, : item["actions"].numel()].mean()

            policy_component = policy_loss_total / len(rollout)
            value_component = value_loss_total / len(rollout)
            entropy_component = entropy_total / len(rollout)
            loss = policy_component + args.value_coef * value_component - args.entropy_coef * entropy_component
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_values.append(float(loss.detach().cpu().item()))
            policy_loss_values.append(float(policy_component.detach().cpu().item()))
            value_loss_values.append(float(value_component.detach().cpu().item()))
            entropy_values.append(float(entropy_component.detach().cpu().item()))

        avg_reward = sum(item["reward"].item() for item in rollout) / max(len(rollout), 1)
        print(f"[PPO] update={update_idx} avg_reward={avg_reward:.6f}")
        _append_metrics(
            args.metrics_jsonl,
            {
                "phase": "ppo",
                "update": update_idx,
                "avg_reward": avg_reward,
                "avg_success_count": sum(float(item["success_count"]) for item in rollout) / max(len(rollout), 1),
                "avg_missed_count": sum(float(item["missed_count"]) for item in rollout) / max(len(rollout), 1),
                "avg_tardiness": sum(float(item["tardiness"]) for item in rollout) / max(len(rollout), 1),
                "avg_route_time": sum(float(item["route_time"]) for item in rollout) / max(len(rollout), 1),
                "loss": sum(loss_values) / max(len(loss_values), 1),
                "policy_loss": sum(policy_loss_values) / max(len(policy_loss_values), 1),
                "value_loss": sum(value_loss_values) / max(len(value_loss_values), 1),
                "entropy": sum(entropy_values) / max(len(entropy_values), 1),
                "lr": args.lr,
            },
        )

    save_checkpoint(args.output, model)
    print(f"[PPO] saved checkpoint to {args.output}")
    return 0


def train_ppo_event(args: argparse.Namespace) -> int:
    require_torch()
    import torch

    model = _load_or_make_model(args, torch)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _reset_metrics_file(args.metrics_jsonl)

    env = EventDrivenDispatchEnv(
        ugv_num=args.ugv_num,
        seed=args.seed,
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
        success_reward=args.success_reward,
        miss_penalty=args.miss_penalty,
        tardiness_weight=args.tardiness_weight,
        ongoing_tardiness_weight=args.ongoing_tardiness_weight,
        response_time_weight=args.response_time_weight,
        flight_time_weight=args.flight_time_weight,
        flight_distance_weight=args.flight_distance_weight,
        route_churn_weight=args.route_churn_weight,
    )
    slot_rng = random.Random(args.seed + 2000003)

    for update_idx in range(args.updates):
        rollout = []
        episode_rewards = []
        episode_hits = []
        episode_misses = []
        for _ in range(args.batch_size):
            observation = env.reset()
            episode_items = []
            total_reward = 0.0
            done = False
            final_info = {}
            while not done and len(episode_items) < args.max_decisions_per_episode:
                policy_observation, permutation = _augment_observation_slots(observation, slot_rng)
                batch = _observation_to_batch(policy_observation, args.device)
                output = model(batch, greedy=False)
                active_count = int(policy_observation["active_count"])
                actions = output["actions"][0, :active_count]
                log_prob = output["log_probs"][0, :active_count].sum()
                value = output["values"][0]
                env_actions = unpermute_order_slots(actions.detach().cpu().tolist(), permutation)
                next_observation, reward_value, done, final_info = env.step(env_actions)
                reward = torch.tensor(float(reward_value), dtype=torch.float32, device=args.device)
                episode_items.append(
                    {
                        "batch": batch,
                        "actions": actions.detach(),
                        "old_log_prob": log_prob.detach(),
                        "reward": reward,
                        "value": value.detach(),
                    }
                )
                total_reward += float(reward_value)
                if next_observation is None:
                    done = True
                else:
                    observation = next_observation

            discounted_return = torch.tensor(0.0, dtype=torch.float32, device=args.device)
            for item in reversed(episode_items):
                discounted_return = item["reward"] + args.gamma * discounted_return
                item["return"] = discounted_return.detach()
                rollout.append(item)

            episode_rewards.append(total_reward)
            episode_hits.append(float(final_info.get("success_count", 0.0)))
            episode_misses.append(float(final_info.get("missed_count", 0.0)))

        if not rollout:
            print(f"[PPO_EVENT] update={update_idx} skipped_empty_rollout")
            continue

        advantages = torch.stack([item["return"] - item["value"] for item in rollout])
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
        for item, advantage in zip(rollout, advantages):
            item["advantage"] = ((advantage - adv_mean) / adv_std).detach()

        loss_values = []
        policy_loss_values = []
        value_loss_values = []
        entropy_values = []
        for _ in range(args.ppo_epochs):
            policy_loss_total = torch.tensor(0.0, device=args.device)
            value_loss_total = torch.tensor(0.0, device=args.device)
            entropy_total = torch.tensor(0.0, device=args.device)
            for item in rollout:
                current = model(item["batch"], actions=item["actions"].unsqueeze(0))
                new_log_prob = current["log_probs"][0, : item["actions"].numel()].sum()
                value = current["values"][0]
                ratio = torch.exp(new_log_prob - item["old_log_prob"])
                unclipped = ratio * item["advantage"]
                clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * item["advantage"]
                policy_loss_total = policy_loss_total - torch.min(unclipped, clipped)
                value_loss_total = value_loss_total + 0.5 * torch.square(value - item["return"])
                entropy_total = entropy_total + current["entropy"][0, : item["actions"].numel()].mean()

            policy_component = policy_loss_total / len(rollout)
            value_component = value_loss_total / len(rollout)
            entropy_component = entropy_total / len(rollout)
            loss = policy_component + args.value_coef * value_component - args.entropy_coef * entropy_component
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_values.append(float(loss.detach().cpu().item()))
            policy_loss_values.append(float(policy_component.detach().cpu().item()))
            value_loss_values.append(float(value_component.detach().cpu().item()))
            entropy_values.append(float(entropy_component.detach().cpu().item()))

        avg_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
        avg_hits = sum(episode_hits) / max(len(episode_hits), 1)
        avg_misses = sum(episode_misses) / max(len(episode_misses), 1)
        print(
            f"[PPO_EVENT] update={update_idx} "
            f"avg_reward={avg_reward:.6f} avg_hits={avg_hits:.3f} avg_misses={avg_misses:.3f} "
            f"transitions={len(rollout)}"
        )
        _append_metrics(
            args.metrics_jsonl,
            {
                "phase": "ppo_event",
                "update": update_idx,
                "avg_reward": avg_reward,
                "avg_hits": avg_hits,
                "avg_misses": avg_misses,
                "transitions": len(rollout),
                "loss": sum(loss_values) / max(len(loss_values), 1),
                "policy_loss": sum(policy_loss_values) / max(len(policy_loss_values), 1),
                "value_loss": sum(value_loss_values) / max(len(value_loss_values), 1),
                "entropy": sum(entropy_values) / max(len(entropy_values), 1),
                "lr": args.lr,
                "success_reward": args.success_reward,
                "miss_penalty": args.miss_penalty,
                "tardiness_weight": args.tardiness_weight,
                "response_time_weight": args.response_time_weight,
                "flight_time_weight": args.flight_time_weight,
                "flight_distance_weight": args.flight_distance_weight,
            },
        )

    save_checkpoint(args.output, model)
    print(f"[PPO_EVENT] saved checkpoint to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train UAV RL dispatch policy.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bc = subparsers.add_parser("bc", help="Train behavior cloning policy from expert data.")
    bc.add_argument("--dataset", required=True)
    bc.add_argument("--output", required=True)
    bc.add_argument("--ugv-num", type=int, default=7)
    bc.add_argument("--hidden-dim", type=int, default=128)
    bc.add_argument("--epochs", type=int, default=20)
    bc.add_argument("--batch-size", type=int, default=32)
    bc.add_argument("--lr", type=float, default=1e-3)
    bc.add_argument("--device", default="cpu")
    bc.add_argument("--metrics-jsonl", default="", help="Optional JSONL path for BC training metrics.")
    bc.set_defaults(func=train_bc)

    ppo = subparsers.add_parser("ppo", help="Fine-tune dispatch policy with PPO.")
    ppo.add_argument("--output", required=True)
    ppo.add_argument("--init-checkpoint", default="")
    ppo.add_argument("--ugv-num", type=int, default=7)
    ppo.add_argument("--hidden-dim", type=int, default=128)
    ppo.add_argument("--updates", type=int, default=50)
    ppo.add_argument("--ppo-epochs", type=int, default=4)
    ppo.add_argument("--batch-size", type=int, default=64)
    ppo.add_argument("--lr", type=float, default=3e-4)
    ppo.add_argument("--clip-ratio", type=float, default=0.2)
    ppo.add_argument("--value-coef", type=float, default=0.5)
    ppo.add_argument("--entropy-coef", type=float, default=0.01)
    ppo.add_argument("--map-size-x", type=float, default=35.0)
    ppo.add_argument("--map-size-y", type=float, default=35.0)
    ppo.add_argument("--v-max", type=float, default=3.0)
    ppo.add_argument("--a-max", type=float, default=1.0)
    ppo.add_argument("--seed", type=int, default=0)
    ppo.add_argument("--device", default="cpu")
    ppo.add_argument("--metrics-jsonl", default="", help="Optional JSONL path for PPO training metrics.")
    ppo.set_defaults(func=train_ppo)

    ppo_event = subparsers.add_parser("ppo_event", help="Fine-tune policy in an event-driven online dispatch approximation.")
    ppo_event.add_argument("--output", required=True)
    ppo_event.add_argument("--init-checkpoint", default="")
    ppo_event.add_argument("--ugv-num", type=int, default=7)
    ppo_event.add_argument("--hidden-dim", type=int, default=128)
    ppo_event.add_argument("--updates", type=int, default=200)
    ppo_event.add_argument("--ppo-epochs", type=int, default=4)
    ppo_event.add_argument("--batch-size", type=int, default=16, help="Event-driven episodes per update.")
    ppo_event.add_argument("--lr", type=float, default=3e-4)
    ppo_event.add_argument("--clip-ratio", type=float, default=0.2)
    ppo_event.add_argument("--value-coef", type=float, default=0.5)
    ppo_event.add_argument("--entropy-coef", type=float, default=0.01)
    ppo_event.add_argument("--gamma", type=float, default=0.98)
    ppo_event.add_argument("--map-size-x", type=float, default=35.0)
    ppo_event.add_argument("--map-size-y", type=float, default=35.0)
    ppo_event.add_argument("--v-max", type=float, default=3.0)
    ppo_event.add_argument("--a-max", type=float, default=1.0)
    ppo_event.add_argument("--horizon-sec", type=float, default=150.0)
    ppo_event.add_argument("--replan-period", type=float, default=10.0)
    ppo_event.add_argument("--arrival-radius", type=float, default=0.8)
    ppo_event.add_argument("--min-event-interval", type=float, default=1.0)
    ppo_event.add_argument("--max-event-interval", type=float, default=4.0)
    ppo_event.add_argument("--min-slack-factor", type=float, default=1.1)
    ppo_event.add_argument("--max-slack-factor", type=float, default=2.6)
    ppo_event.add_argument("--max-extra-deadline", type=float, default=3.0)
    ppo_event.add_argument("--success-reward", type=float, default=4.0)
    ppo_event.add_argument("--miss-penalty", type=float, default=6.0)
    ppo_event.add_argument("--tardiness-weight", type=float, default=0.25)
    ppo_event.add_argument("--ongoing-tardiness-weight", type=float, default=0.04)
    ppo_event.add_argument("--response-time-weight", type=float, default=0.05)
    ppo_event.add_argument("--flight-time-weight", type=float, default=0.01)
    ppo_event.add_argument("--flight-distance-weight", type=float, default=0.005)
    ppo_event.add_argument("--route-churn-weight", type=float, default=0.02)
    ppo_event.add_argument("--max-decisions-per-episode", type=int, default=80)
    ppo_event.add_argument("--seed", type=int, default=0)
    ppo_event.add_argument("--device", default="cpu")
    ppo_event.add_argument("--metrics-jsonl", default="", help="Optional JSONL path for event-driven PPO metrics.")
    ppo_event.set_defaults(func=train_ppo_event)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
