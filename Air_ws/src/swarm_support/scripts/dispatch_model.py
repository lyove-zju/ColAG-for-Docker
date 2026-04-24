#!/usr/bin/env python3
"""Transformer + autoregressive pointer policy for UAV dispatch."""

from __future__ import annotations

import math
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

from dispatch_runtime import GLOBAL_FEATURE_DIM, MAX_UGV_NUM, NODE_FEATURE_DIM


_BaseModule = nn.Module if nn is not None else object


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for RL dispatch training/inference. Install it before running this script.")


class DispatchPointerPolicy(_BaseModule):
    def __init__(
        self,
        ugv_num: int = MAX_UGV_NUM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        require_torch()
        super().__init__()
        self.ugv_num = ugv_num
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.global_embed = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_embed = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        self.pairwise_row_embed = nn.Linear(ugv_num, hidden_dim)
        self.slot_embed = nn.Embedding(ugv_num + 1, hidden_dim)
        self.depot_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.query_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_output = nn.Linear(hidden_dim, 1)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, batch: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        global_features = batch["global_features"]
        node_features = batch["node_features"]
        pairwise_time = batch["pairwise_time_matrix"]
        active_mask = batch["active_mask"]

        batch_size = global_features.size(0)
        slot_ids = torch.arange(self.ugv_num, device=global_features.device).unsqueeze(0).expand(batch_size, -1)
        node_tokens = (
            self.node_embed(node_features)
            + self.pairwise_row_embed(pairwise_time)
            + self.slot_embed(slot_ids + 1)
        )
        depot = self.global_embed(global_features).unsqueeze(1) + self.depot_token
        tokens = torch.cat([depot, node_tokens], dim=1)
        src_padding_mask = torch.cat(
            [torch.zeros(batch_size, 1, dtype=torch.bool, device=active_mask.device), ~active_mask],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=src_padding_mask)
        return {
            "global_token": encoded[:, 0],
            "node_tokens": encoded[:, 1:],
            "active_mask": active_mask,
        }

    def _decode(
        self,
        encoded: Dict[str, "torch.Tensor"],
        actions: Optional["torch.Tensor"] = None,
        greedy: bool = False,
    ) -> Dict[str, "torch.Tensor"]:
        node_tokens = encoded["node_tokens"]
        active_mask = encoded["active_mask"]
        global_token = encoded["global_token"]
        batch_size = node_tokens.size(0)
        max_steps = actions.size(1) if actions is not None else int(active_mask.sum(dim=1).max().item())
        hidden = global_token
        last_token = global_token
        remaining = active_mask.clone()

        selected_actions = []
        log_probs = []
        entropies = []

        for step in range(max_steps):
            hidden = self.query_gru(last_token, hidden)
            query = self.pointer_query(hidden).unsqueeze(1).expand_as(node_tokens)
            score_input = torch.tanh(self.pointer_key(node_tokens) + query)
            logits = self.pointer_output(score_input).squeeze(-1) / math.sqrt(self.hidden_dim)

            valid_rows = remaining.any(dim=1)
            safe_logits = logits.masked_fill(~remaining, float("-inf"))
            if (~valid_rows).any():
                safe_logits = safe_logits.clone()
                safe_logits[~valid_rows, 0] = 0.0

            dist = torch.distributions.Categorical(logits=safe_logits)
            if actions is not None:
                step_actions = actions[:, step].clone()
                step_actions[step_actions < 0] = 0
            elif greedy:
                step_actions = torch.argmax(safe_logits, dim=-1)
            else:
                step_actions = dist.sample()

            step_log_prob = dist.log_prob(step_actions)
            step_entropy = dist.entropy()
            if (~valid_rows).any():
                step_log_prob = step_log_prob * valid_rows.float()
                step_entropy = step_entropy * valid_rows.float()

            selected_actions.append(step_actions)
            log_probs.append(step_log_prob)
            entropies.append(step_entropy)

            gather_index = step_actions.view(batch_size, 1, 1).expand(-1, 1, node_tokens.size(-1))
            last_token = torch.gather(node_tokens, 1, gather_index).squeeze(1)
            remaining = remaining & ~F.one_hot(step_actions, num_classes=self.ugv_num).bool()

        return {
            "actions": torch.stack(selected_actions, dim=1) if selected_actions else torch.zeros(batch_size, 0, dtype=torch.long, device=node_tokens.device),
            "log_probs": torch.stack(log_probs, dim=1) if log_probs else torch.zeros(batch_size, 0, device=node_tokens.device),
            "entropy": torch.stack(entropies, dim=1) if entropies else torch.zeros(batch_size, 0, device=node_tokens.device),
        }

    def forward(
        self,
        batch: Dict[str, "torch.Tensor"],
        actions: Optional["torch.Tensor"] = None,
        greedy: bool = False,
    ) -> Dict[str, "torch.Tensor"]:
        encoded = self.encode(batch)
        decoded = self._decode(encoded, actions=actions, greedy=greedy)
        pooled = encoded["global_token"]
        values = self.critic(pooled).squeeze(-1)
        decoded["values"] = values
        return decoded

    def greedy_order(self, batch: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        require_torch()
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self.forward(batch, greedy=True)
        if was_training:
            self.train()
        return result


def checkpoint_payload(model: DispatchPointerPolicy) -> Dict[str, object]:
    return {
        "state_dict": model.state_dict(),
        "config": {
            "ugv_num": model.ugv_num,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "dropout": model.dropout,
        },
    }


def save_checkpoint(path: str, model: DispatchPointerPolicy) -> None:
    require_torch()
    torch.save(checkpoint_payload(model), path)


def load_policy_checkpoint(path: str, device: str = "cpu") -> DispatchPointerPolicy:
    require_torch()
    payload = torch.load(path, map_location=device)
    config = payload.get("config", {})
    model = DispatchPointerPolicy(
        ugv_num=config.get("ugv_num", MAX_UGV_NUM),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
