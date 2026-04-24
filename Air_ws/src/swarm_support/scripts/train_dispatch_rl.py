#!/usr/bin/env python3
"""Behavior cloning + PPO trainer for UAV dispatch."""

from __future__ import annotations

import argparse
import math
from typing import Dict

from dispatch_dataset import DispatchDataset, collate_dispatch_batch, require_torch
from dispatch_env import DispatchBanditEnv, SyntheticDispatchGenerator
from dispatch_model import DispatchPointerPolicy, save_checkpoint
from dispatch_runtime import evaluate_order_metrics


def _move_batch(batch: Dict[str, "torch.Tensor"], device: str) -> Dict[str, "torch.Tensor"]:
    return {key: value.to(device) for key, value in batch.items()}


def train_bc(args: argparse.Namespace) -> int:
    require_torch()
    import torch
    from torch.utils.data import DataLoader

    dataset = DispatchDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dispatch_batch)
    model = DispatchPointerPolicy(ugv_num=args.ugv_num, hidden_dim=args.hidden_dim)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    save_checkpoint(args.output, model)
    print(f"[BC] saved checkpoint to {args.output}")
    return 0


def train_ppo(args: argparse.Namespace) -> int:
    require_torch()
    import torch

    model = DispatchPointerPolicy(ugv_num=args.ugv_num, hidden_dim=args.hidden_dim)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=args.device)
        model.load_state_dict(payload["state_dict"])
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    for update_idx in range(args.updates):
        rollout = []
        for _ in range(args.batch_size):
            observation = env.reset()
            batch = {
                "global_features": torch.tensor([observation["global_features"]], dtype=torch.float32, device=args.device),
                "node_features": torch.tensor([observation["node_features"]], dtype=torch.float32, device=args.device),
                "pairwise_time_matrix": torch.tensor([observation["pairwise_time_matrix"]], dtype=torch.float32, device=args.device),
                "active_mask": torch.tensor([observation["active_mask"]], dtype=torch.bool, device=args.device),
            }
            output = model(batch, greedy=False)
            active_count = int(observation["active_count"])
            actions = output["actions"][0, :active_count]
            log_prob = output["log_probs"][0, :active_count].sum()
            value = output["values"][0]
            metrics = evaluate_order_metrics(actions.detach().cpu().tolist(), observation)
            reward = torch.tensor(metrics.reward, dtype=torch.float32, device=args.device)
            rollout.append(
                {
                    "batch": batch,
                    "actions": actions.detach(),
                    "old_log_prob": log_prob.detach(),
                    "reward": reward,
                    "value": value.detach(),
                }
            )

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

            loss = (
                policy_loss_total / len(rollout)
                + args.value_coef * value_loss_total / len(rollout)
                - args.entropy_coef * entropy_total / len(rollout)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_reward = sum(item["reward"].item() for item in rollout) / max(len(rollout), 1)
        print(f"[PPO] update={update_idx} avg_reward={avg_reward:.6f}")

    save_checkpoint(args.output, model)
    print(f"[PPO] saved checkpoint to {args.output}")
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
    ppo.set_defaults(func=train_ppo)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
