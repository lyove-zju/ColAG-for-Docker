#!/usr/bin/env python3
"""Dataset and tensor helpers for UAV RL dispatch."""

from __future__ import annotations

import json
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
    def __init__(self, path: str):
        require_torch()
        self.path = Path(path)
        self.records = load_jsonl(str(self.path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        expert_order = padded_order(record["expert_order_slots"])
        return {
            "global_features": record["global_features"],
            "node_features": record["node_features"],
            "pairwise_time_matrix": record["pairwise_time_matrix"],
            "active_mask": record["active_mask"],
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
