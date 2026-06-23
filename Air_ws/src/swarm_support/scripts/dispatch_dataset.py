#!/usr/bin/env python3
"""Dataset and tensor helpers for UAV RL dispatch."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

from dispatch_runtime import MAX_UGV_NUM


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for RL dispatch training/inference. Install it before running this script.")


def padded_order(order_slots: Sequence[int], pad_value: int = -1, max_len: int = MAX_UGV_NUM) -> List[int]:
    padded = [int(slot) for slot in order_slots[:max_len]]
    while len(padded) < max_len:
        padded.append(pad_value)
    return padded


def random_active_slot_permutation(active_mask: Sequence[bool], rng=None) -> List[int]:
    rng = rng or random
    active_slots = [idx for idx, enabled in enumerate(active_mask) if enabled]
    inactive_slots = [idx for idx, enabled in enumerate(active_mask) if not enabled]
    rng.shuffle(active_slots)
    return active_slots + inactive_slots


def inverse_permutation(permutation: Sequence[int]) -> List[int]:
    inverse = [0 for _ in permutation]
    for new_slot, old_slot in enumerate(permutation):
        inverse[int(old_slot)] = int(new_slot)
    return inverse


def remap_order_slots(order_slots: Sequence[int], inverse: Sequence[int]) -> List[int]:
    return [int(inverse[int(slot)]) for slot in order_slots if int(slot) >= 0]


def unpermute_order_slots(order_slots: Sequence[int], permutation: Sequence[int]) -> List[int]:
    return [int(permutation[int(slot)]) for slot in order_slots if int(slot) >= 0]


def permute_observation_slots(observation: Dict[str, object], permutation: Sequence[int]) -> Dict[str, object]:
    perm = [int(slot) for slot in permutation]
    inverse = inverse_permutation(perm)
    result = dict(observation)

    for key in ("node_features", "active_mask", "slot_task_ids", "slot_waypoints"):
        if key in result:
            result[key] = [result[key][old_slot] for old_slot in perm]

    if "pairwise_time_matrix" in result:
        matrix = result["pairwise_time_matrix"]
        result["pairwise_time_matrix"] = [
            [matrix[old_i][old_j] for old_j in perm]
            for old_i in perm
        ]

    if "previous_order_slots" in result:
        result["previous_order_slots"] = remap_order_slots(result["previous_order_slots"], inverse)

    if "slot_task_ids" in result and "active_mask" in result:
        result["active_ugv_ids"] = [
            result["slot_task_ids"][idx]
            for idx, enabled in enumerate(result["active_mask"])
            if enabled
        ]

    return result


def save_jsonl(records: Iterable[Dict[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True))
            fh.write("\n")


def load_jsonl(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def tensorize_observation(observation: Dict[str, object], device: str = "cpu") -> Dict[str, "torch.Tensor"]:
    require_torch()
    return {
        "global_features": torch.tensor([observation["global_features"]], dtype=torch.float32, device=device),
        "node_features": torch.tensor([observation["node_features"]], dtype=torch.float32, device=device),
        "pairwise_time_matrix": torch.tensor([observation["pairwise_time_matrix"]], dtype=torch.float32, device=device),
        "active_mask": torch.tensor([observation["active_mask"]], dtype=torch.bool, device=device),
    }


class DispatchDataset(Dataset):
    def __init__(self, path: str, permute_slots: bool = True):
        require_torch()
        self.path = Path(path)
        self.records = load_jsonl(str(self.path))
        self.permute_slots = permute_slots

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        observation = {
            "global_features": record["global_features"],
            "node_features": record["node_features"],
            "pairwise_time_matrix": record["pairwise_time_matrix"],
            "active_mask": record["active_mask"],
        }
        expert_order_slots = list(record["expert_order_slots"])
        if self.permute_slots:
            permutation = random_active_slot_permutation(record["active_mask"])
            inverse = inverse_permutation(permutation)
            observation = permute_observation_slots(observation, permutation)
            expert_order_slots = remap_order_slots(expert_order_slots, inverse)
        expert_order = padded_order(expert_order_slots)
        return {
            "global_features": observation["global_features"],
            "node_features": observation["node_features"],
            "pairwise_time_matrix": observation["pairwise_time_matrix"],
            "active_mask": observation["active_mask"],
            "expert_order": expert_order,
            "step_mask": [1 if slot >= 0 else 0 for slot in expert_order],
            "meta": record.get("meta", {}),
        }


def collate_dispatch_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, "torch.Tensor"]:
    require_torch()
    return {
        "global_features": torch.tensor([item["global_features"] for item in batch], dtype=torch.float32),
        "node_features": torch.tensor([item["node_features"] for item in batch], dtype=torch.float32),
        "pairwise_time_matrix": torch.tensor([item["pairwise_time_matrix"] for item in batch], dtype=torch.float32),
        "active_mask": torch.tensor([item["active_mask"] for item in batch], dtype=torch.bool),
        "expert_order": torch.tensor([item["expert_order"] for item in batch], dtype=torch.long),
        "step_mask": torch.tensor([item["step_mask"] for item in batch], dtype=torch.float32),
    }
