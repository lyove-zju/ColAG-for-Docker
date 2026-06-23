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
        use_ugv_id_feature: bool = False,
        slot_embedding_scale: float = 0.0,
        pairwise_row_embedding_scale: float = 0.0,
        use_edge_bias: bool = True,
        strong_critic: bool = True,
    ):
        require_torch()
        super().__init__()
        self.ugv_num = ugv_num
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_ugv_id_feature = bool(use_ugv_id_feature)
        self.slot_embedding_scale = float(slot_embedding_scale)
        self.pairwise_row_embedding_scale = float(pairwise_row_embedding_scale)
        self.use_edge_bias = bool(use_edge_bias)
        self.strong_critic = bool(strong_critic)

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
        self.edge_bias = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        critic_input_dim = hidden_dim if not self.strong_critic else hidden_dim * 3 + 4
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _policy_node_features(self, node_features: "torch.Tensor") -> "torch.Tensor":
        if self.use_ugv_id_feature:
            return node_features
        sanitized = node_features.clone()
        sanitized[:, :, 1] = 0.0
        return sanitized

    def encode(self, batch: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        global_features = batch["global_features"]
        raw_node_features = batch["node_features"]
        node_features = self._policy_node_features(raw_node_features)
        pairwise_time = batch["pairwise_time_matrix"]
        active_mask = batch["active_mask"]

        batch_size = global_features.size(0)
        slot_ids = torch.arange(self.ugv_num, device=global_features.device).unsqueeze(0).expand(batch_size, -1)
        node_tokens = self.node_embed(node_features)
        if self.pairwise_row_embedding_scale != 0.0:
            node_tokens = node_tokens + self.pairwise_row_embedding_scale * self.pairwise_row_embed(pairwise_time)
        if self.slot_embedding_scale != 0.0:
            node_tokens = node_tokens + self.slot_embedding_scale * self.slot_embed(slot_ids + 1)
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
            "node_features": raw_node_features,
            "pairwise_time": pairwise_time,
        }

    def _edge_logits(
        self,
        encoded: Dict[str, "torch.Tensor"],
        step: int,
        last_actions: Optional["torch.Tensor"],
        elapsed_time: "torch.Tensor",
    ) -> "torch.Tensor":
        node_features = encoded["node_features"]
        pairwise_time = encoded["pairwise_time"]
        batch_size = node_features.size(0)

        if step == 0 or last_actions is None:
            edge_time = node_features[:, :, 8]
        else:
            gather_index = last_actions.view(batch_size, 1, 1).expand(-1, 1, self.ugv_num)
            edge_time = torch.gather(pairwise_time, 1, gather_index).squeeze(1)

        deadline_remaining = node_features[:, :, 7]
        slack_after_edge = deadline_remaining - elapsed_time.unsqueeze(1) - edge_time
        travel_from_now = node_features[:, :, 8]
        edge_features = torch.stack(
            [edge_time, deadline_remaining, slack_after_edge, travel_from_now],
            dim=-1,
        )
        return self.edge_bias(edge_features).squeeze(-1)

    def _critic_features(self, encoded: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        if not self.strong_critic:
            return encoded["global_token"]

        node_tokens = encoded["node_tokens"]
        active_mask = encoded["active_mask"]
        node_features = encoded["node_features"]
        mask_f = active_mask.float()
        active_count = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        mean_pool = (node_tokens * mask_f.unsqueeze(-1)).sum(dim=1) / active_count
        masked_tokens = node_tokens.masked_fill(~active_mask.unsqueeze(-1), -1e9)
        max_pool = masked_tokens.max(dim=1).values
        has_active = active_mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_active, max_pool, torch.zeros_like(max_pool))

        deadline = node_features[:, :, 7]
        slack = node_features[:, :, 9]
        inf = torch.full_like(deadline, 1e6)
        min_deadline = torch.where(active_mask, deadline, inf).min(dim=1, keepdim=True).values
        min_slack = torch.where(active_mask, slack, inf).min(dim=1, keepdim=True).values
        mean_slack = (slack * mask_f).sum(dim=1, keepdim=True) / active_count
        active_ratio = mask_f.sum(dim=1, keepdim=True) / float(self.ugv_num)
        stats = torch.cat(
            [
                active_ratio,
                torch.where(has_active, min_deadline, torch.zeros_like(min_deadline)),
                torch.where(has_active, min_slack, torch.zeros_like(min_slack)),
                mean_slack,
            ],
            dim=1,
        )

        return torch.cat([encoded["global_token"], mean_pool, max_pool, stats], dim=1)

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
        elapsed_time = torch.zeros(batch_size, device=node_tokens.device)
        last_actions = None

        selected_actions = []
        log_probs = []
        entropies = []

        for step in range(max_steps):
            hidden = self.query_gru(last_token, hidden)
            query = self.pointer_query(hidden).unsqueeze(1).expand_as(node_tokens)
            score_input = torch.tanh(self.pointer_key(node_tokens) + query)
            logits = self.pointer_output(score_input).squeeze(-1) / math.sqrt(self.hidden_dim)
            if self.use_edge_bias:
                logits = logits + self._edge_logits(encoded, step, last_actions, elapsed_time)

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
            if self.use_edge_bias:
                if step == 0 or last_actions is None:
                    current_edge_time = encoded["node_features"][:, :, 8]
                else:
                    edge_index = last_actions.view(batch_size, 1, 1).expand(-1, 1, self.ugv_num)
                    current_edge_time = torch.gather(encoded["pairwise_time"], 1, edge_index).squeeze(1)
                selected_edge_time = current_edge_time.gather(1, step_actions.view(batch_size, 1)).squeeze(1)
                elapsed_time = elapsed_time + selected_edge_time * valid_rows.float()
            last_actions = step_actions
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
        pooled = self._critic_features(encoded)
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
            "use_ugv_id_feature": model.use_ugv_id_feature,
            "slot_embedding_scale": model.slot_embedding_scale,
            "pairwise_row_embedding_scale": model.pairwise_row_embedding_scale,
            "use_edge_bias": model.use_edge_bias,
            "strong_critic": model.strong_critic,
        },
    }


def save_checkpoint(path: str, model: DispatchPointerPolicy) -> None:
    require_torch()
    torch.save(checkpoint_payload(model), path)


def build_policy_from_config(config: Dict[str, object], legacy_checkpoint: bool = False) -> DispatchPointerPolicy:
    return DispatchPointerPolicy(
        ugv_num=config.get("ugv_num", MAX_UGV_NUM),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
        use_ugv_id_feature=config.get("use_ugv_id_feature", True if legacy_checkpoint else False),
        slot_embedding_scale=config.get("slot_embedding_scale", 1.0 if legacy_checkpoint else 0.0),
        pairwise_row_embedding_scale=config.get("pairwise_row_embedding_scale", 1.0 if legacy_checkpoint else 0.0),
        use_edge_bias=config.get("use_edge_bias", False if legacy_checkpoint else True),
        strong_critic=config.get("strong_critic", False if legacy_checkpoint else True),
    )


def load_policy_checkpoint(path: str, device: str = "cpu") -> DispatchPointerPolicy:
    require_torch()
    payload = torch.load(path, map_location=device)
    config = payload.get("config", {})
    legacy_checkpoint = "use_ugv_id_feature" not in config
    model = build_policy_from_config(config, legacy_checkpoint=legacy_checkpoint)
    model.load_state_dict(payload["state_dict"], strict=not legacy_checkpoint)
    model.to(device)
    model.eval()
    return model
