#!/usr/bin/env python3
"""Lightweight dispatch environment for offline RL training and evaluation."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from dispatch_runtime import (
    DEFAULT_FLY_HEIGHT,
    DispatchObservationBuilder,
    DispatchTask,
    evaluate_order_metrics,
    random_permutation,
)


@dataclass
class SyntheticDispatchScenario:
    tasks: List[DispatchTask]
    self_pos: List[float]
    self_vel: List[float]
    previous_order_slots: List[int]
    observation: Dict[str, object]


class SyntheticDispatchGenerator:
    def __init__(
        self,
        ugv_num: int,
        map_size_x: float = 35.0,
        map_size_y: float = 35.0,
        max_active_tasks: Optional[int] = None,
        seed: int = 0,
        v_max: float = 3.0,
        a_max: float = 1.0,
        time_resolution: int = 100,
    ):
        self.ugv_num = ugv_num
        self.map_size_x = map_size_x
        self.map_size_y = map_size_y
        self.max_active_tasks = max_active_tasks or max(1, ugv_num)
        self.rng = random.Random(seed)
        self.builder = DispatchObservationBuilder(
            ugv_num=ugv_num,
            v_max=v_max,
            a_max=a_max,
            time_resolution=time_resolution,
        )

    def _uniform_xy(self) -> List[float]:
        return [
            self.rng.uniform(-self.map_size_x / 2.0, self.map_size_x / 2.0),
            self.rng.uniform(-self.map_size_y / 2.0, self.map_size_y / 2.0),
        ]

    def sample(self) -> SyntheticDispatchScenario:
        active_count = self.rng.randint(1, min(self.ugv_num, self.max_active_tasks))
        ugv_ids = sorted(self.rng.sample(range(self.ugv_num), active_count))
        self_pos = self._uniform_xy() + [DEFAULT_FLY_HEIGHT]
        self_vel = [self.rng.uniform(-1.0, 1.0), self.rng.uniform(-1.0, 1.0), 0.0]
        now_sec = 0.0

        provisional_tasks: List[DispatchTask] = []
        for ugv_id in ugv_ids:
            x, y = self._uniform_xy()
            provisional_tasks.append(
                DispatchTask(
                    ugv_id=ugv_id,
                    collision_x=x,
                    collision_y=y,
                    collision_z=DEFAULT_FLY_HEIGHT,
                    deadline_sec=1.0,
                )
            )

        base_observation = self.builder.build_observation(provisional_tasks, self_pos, self_vel, now_sec)
        tasks: List[DispatchTask] = []
        for slot, task in enumerate(provisional_tasks):
            travel_time = base_observation["time_matrix"][0][slot + 1] / float(self.builder.time_resolution)
            slack_factor = self.rng.uniform(1.2, 2.8)
            extra_time = self.rng.uniform(0.0, 4.0)
            deadline_sec = max(0.5, travel_time * slack_factor + extra_time)
            tasks.append(
                DispatchTask(
                    ugv_id=task.ugv_id,
                    collision_x=task.collision_x,
                    collision_y=task.collision_y,
                    collision_z=task.collision_z,
                    deadline_sec=deadline_sec,
                )
            )

        observation = self.builder.build_observation(tasks, self_pos, self_vel, now_sec)
        previous_order = []
        if active_count > 1 and self.rng.random() < 0.5:
            previous_order = random_permutation(observation["active_mask"], self.rng)
            observation["previous_order_slots"] = previous_order

        return SyntheticDispatchScenario(
            tasks=tasks,
            self_pos=self_pos,
            self_vel=self_vel,
            previous_order_slots=previous_order,
            observation=observation,
        )


class DispatchBanditEnv:
    def __init__(self, generator: SyntheticDispatchGenerator):
        self.generator = generator
        self.current: Optional[SyntheticDispatchScenario] = None

    def reset(self) -> Dict[str, object]:
        self.current = self.generator.sample()
        return self.current.observation

    def step(self, order_slots: Sequence[int]):
        if self.current is None:
            raise RuntimeError("Call reset() before step().")
        metrics = evaluate_order_metrics(order_slots, self.current.observation)
        info = {
            "success_count": metrics.success_count,
            "missed_count": metrics.missed_count,
            "tardiness": metrics.tardiness,
            "route_time": metrics.route_time,
            "route_churn": metrics.route_churn,
        }
        return None, metrics.reward, True, info


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-sample lightweight dispatch scenarios.")
    parser.add_argument("--ugv-num", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    generator = SyntheticDispatchGenerator(ugv_num=args.ugv_num, seed=args.seed)
    for idx in range(args.num_samples):
        scenario = generator.sample()
        print(
            "sample",
            idx,
            "active_ids=",
            scenario.observation["active_ugv_ids"],
            "time_windows=",
            scenario.observation["time_windows"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
