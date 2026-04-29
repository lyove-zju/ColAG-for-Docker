#!/usr/bin/env python3
"""Lightweight dispatch environment for offline RL training and evaluation."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from dispatch_runtime import (
    DEFAULT_FLY_HEIGHT,
    DispatchObservationBuilder,
    DispatchTask,
    compute_route_churn,
    evaluate_order_metrics,
    random_permutation,
    sanitize_slot_order,
    slot_order_to_ugv_ids,
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


@dataclass
class OnlineDispatchTask:
    task: DispatchTask
    start_time: float
    missed: bool = False


class EventDrivenDispatchEnv:
    """Event-driven approximation of the online UAV dispatch loop.

    The environment intentionally keeps the same dispatch boundary as the ROS
    node: active blind-info-like tasks plus UAV odometry go in, and an ordered
    list of UGV slots comes out. It does not model obstacle avoidance; the
    downstream planner remains responsible for that in real simulations.
    """

    def __init__(
        self,
        ugv_num: int,
        map_size_x: float = 35.0,
        map_size_y: float = 35.0,
        seed: int = 0,
        v_max: float = 3.0,
        a_max: float = 1.0,
        time_resolution: int = 100,
        horizon_sec: float = 150.0,
        replan_period: float = 10.0,
        arrival_radius: float = 0.8,
        min_event_interval: float = 1.0,
        max_event_interval: float = 4.0,
        min_slack_factor: float = 1.1,
        max_slack_factor: float = 2.6,
        max_extra_deadline: float = 3.0,
        dt: float = 0.2,
        success_reward: float = 4.0,
        miss_penalty: float = 6.0,
        tardiness_weight: float = 0.25,
        ongoing_tardiness_weight: float = 0.04,
        response_time_weight: float = 0.05,
        flight_time_weight: float = 0.01,
        flight_distance_weight: float = 0.005,
        route_churn_weight: float = 0.02,
    ):
        self.ugv_num = ugv_num
        self.map_size_x = map_size_x
        self.map_size_y = map_size_y
        self.v_max = v_max
        self.a_max = a_max
        self.horizon_sec = horizon_sec
        self.replan_period = replan_period
        self.arrival_radius = arrival_radius
        self.min_event_interval = min_event_interval
        self.max_event_interval = max_event_interval
        self.min_slack_factor = min_slack_factor
        self.max_slack_factor = max_slack_factor
        self.max_extra_deadline = max_extra_deadline
        self.dt = dt
        self.success_reward = success_reward
        self.miss_penalty = miss_penalty
        self.tardiness_weight = tardiness_weight
        self.ongoing_tardiness_weight = ongoing_tardiness_weight
        self.response_time_weight = response_time_weight
        self.flight_time_weight = flight_time_weight
        self.flight_distance_weight = flight_distance_weight
        self.route_churn_weight = route_churn_weight
        self.rng = random.Random(seed)
        self.builder = DispatchObservationBuilder(
            ugv_num=ugv_num,
            v_max=v_max,
            a_max=a_max,
            time_resolution=time_resolution,
        )
        self.now_sec = 0.0
        self.self_pos = [0.0, 0.0, DEFAULT_FLY_HEIGHT]
        self.self_vel = [0.0, 0.0, 0.0]
        self.events: List[Tuple[float, int]] = []
        self.event_index = 0
        self.active: Dict[int, OnlineDispatchTask] = {}
        self.previous_order_slots: List[int] = []
        self.last_observation: Optional[Dict[str, object]] = None
        self.stats: Dict[str, float] = {}

    def _uniform_xy(self) -> List[float]:
        return [
            self.rng.uniform(-self.map_size_x / 2.0, self.map_size_x / 2.0),
            self.rng.uniform(-self.map_size_y / 2.0, self.map_size_y / 2.0),
        ]

    def _distance_xy(self, x: float, y: float) -> float:
        return math.hypot(float(x) - self.self_pos[0], float(y) - self.self_pos[1])

    def _next_event_time(self) -> float:
        if self.event_index >= len(self.events):
            return self.horizon_sec
        return min(self.events[self.event_index][0], self.horizon_sec)

    def _generate_events(self) -> List[Tuple[float, int]]:
        events: List[Tuple[float, int]] = [(0.0, self.rng.randrange(self.ugv_num))]
        current = 0.0
        while current < self.horizon_sec:
            current += self.rng.uniform(self.min_event_interval, self.max_event_interval)
            if current >= self.horizon_sec:
                break
            events.append((current, self.rng.randrange(self.ugv_num)))
            if self.rng.random() < 0.25:
                burst_time = min(self.horizon_sec - 1e-3, current + self.rng.uniform(0.05, 0.4))
                events.append((burst_time, self.rng.randrange(self.ugv_num)))
        events.sort(key=lambda item: item[0])
        return events

    def _make_task(self, ugv_id: int) -> OnlineDispatchTask:
        x, y = self._uniform_xy()
        nominal_travel_time = math.hypot(x, y) / max(self.v_max, 1e-6)
        deadline_duration = max(
            0.5,
            nominal_travel_time * self.rng.uniform(self.min_slack_factor, self.max_slack_factor)
            + self.rng.uniform(0.0, self.max_extra_deadline),
        )
        return OnlineDispatchTask(
            task=DispatchTask(
                ugv_id=ugv_id,
                collision_x=x,
                collision_y=y,
                collision_z=DEFAULT_FLY_HEIGHT,
                deadline_sec=self.now_sec + deadline_duration,
            ),
            start_time=self.now_sec,
        )

    def _apply_events_until_now(self) -> None:
        while self.event_index < len(self.events) and self.events[self.event_index][0] <= self.now_sec + 1e-9:
            _, ugv_id = self.events[self.event_index]
            old_task = self.active.get(ugv_id)
            if old_task is not None and not old_task.missed:
                self.stats["missed_count"] += 1.0
                old_task.missed = True
            self.active[ugv_id] = self._make_task(ugv_id)
            self.event_index += 1

    def _tasks(self) -> List[DispatchTask]:
        return [item.task for _, item in sorted(self.active.items())]

    def _build_observation(self) -> Dict[str, object]:
        observation = self.builder.build_observation(
            tasks=self._tasks(),
            pos=self.self_pos,
            vel=self.self_vel,
            now_sec=self.now_sec,
            previous_order_slots=self.previous_order_slots,
        )
        self.last_observation = observation
        return observation

    def reset(self) -> Dict[str, object]:
        self.now_sec = 0.0
        self.self_pos = self._uniform_xy() + [DEFAULT_FLY_HEIGHT]
        self.self_vel = [0.0, 0.0, 0.0]
        self.events = self._generate_events()
        self.event_index = 0
        self.active = {}
        self.previous_order_slots = []
        self.last_observation = None
        self.stats = {
            "success_count": 0.0,
            "missed_count": 0.0,
            "late_served_count": 0.0,
            "tardiness": 0.0,
            "response_time": 0.0,
            "response_count": 0.0,
            "flight_distance": 0.0,
            "flight_time": 0.0,
            "route_churn": 0.0,
        }
        self._apply_events_until_now()
        if not self.active and self.event_index < len(self.events):
            self.now_sec = self.events[self.event_index][0]
            self._apply_events_until_now()
        return self._build_observation()

    def _mark_new_deadline_misses(self, reward_info: Dict[str, float]) -> None:
        for item in self.active.values():
            if item.missed:
                continue
            if self.now_sec > item.task.deadline_sec:
                item.missed = True
                reward_info["new_misses"] += 1.0
                self.stats["missed_count"] += 1.0

    def _serve_task(self, ugv_id: int, reward_info: Dict[str, float]) -> None:
        item = self.active.pop(ugv_id, None)
        if item is None:
            return
        response_time = max(0.0, self.now_sec - item.start_time)
        reward_info["response_time"] += response_time
        reward_info["response_count"] += 1.0
        self.stats["response_time"] += response_time
        self.stats["response_count"] += 1.0

        if self.now_sec <= item.task.deadline_sec:
            reward_info["hits"] += 1.0
            self.stats["success_count"] += 1.0
        else:
            tardiness = self.now_sec - item.task.deadline_sec
            reward_info["late_served"] += 1.0
            reward_info["tardiness"] += tardiness
            self.stats["late_served_count"] += 1.0
            self.stats["tardiness"] += tardiness
            if not item.missed:
                reward_info["new_misses"] += 1.0
                self.stats["missed_count"] += 1.0

    def _current_target_id(self, ordered_ugv_ids: Sequence[int]) -> Optional[int]:
        for ugv_id in ordered_ugv_ids:
            if ugv_id in self.active:
                return ugv_id
        if not self.active:
            return None
        return sorted(self.active)[0]

    def _advance(self, until_time: float, ordered_ugv_ids: Sequence[int], reward_info: Dict[str, float]) -> None:
        while self.now_sec < until_time - 1e-9:
            step = min(self.dt, until_time - self.now_sec)
            target_id = self._current_target_id(ordered_ugv_ids)
            moved = 0.0
            if target_id is not None:
                target = self.active[target_id].task
                dx = target.collision_x - self.self_pos[0]
                dy = target.collision_y - self.self_pos[1]
                distance = math.hypot(dx, dy)
                if distance <= self.arrival_radius:
                    self._serve_task(target_id, reward_info)
                    continue
                travel = min(distance, self.v_max * step)
                if distance > 1e-9:
                    ux = dx / distance
                    uy = dy / distance
                    self.self_pos[0] += ux * travel
                    self.self_pos[1] += uy * travel
                    self.self_vel[0] = ux * self.v_max
                    self.self_vel[1] = uy * self.v_max
                    moved = travel
            else:
                self.self_vel[0] = 0.0
                self.self_vel[1] = 0.0

            self.now_sec += step
            reward_info["flight_distance"] += moved
            reward_info["flight_time"] += step
            self.stats["flight_distance"] += moved
            self.stats["flight_time"] += step

            self._mark_new_deadline_misses(reward_info)
            for item in self.active.values():
                if item.missed and self.now_sec > item.task.deadline_sec:
                    reward_info["ongoing_tardiness"] += step

    def _reward(self, reward_info: Dict[str, float]) -> float:
        response_penalty = reward_info["response_time"] / max(1.0, reward_info["response_count"])
        return (
            self.success_reward * reward_info["hits"]
            - self.miss_penalty * reward_info["new_misses"]
            - self.tardiness_weight * reward_info["tardiness"]
            - self.ongoing_tardiness_weight * reward_info["ongoing_tardiness"]
            - self.response_time_weight * response_penalty
            - self.flight_time_weight * reward_info["flight_time"]
            - self.flight_distance_weight * reward_info["flight_distance"]
            - self.route_churn_weight * reward_info["route_churn"]
        )

    def step(self, order_slots: Sequence[int]):
        if self.last_observation is None:
            raise RuntimeError("Call reset() before step().")

        active_count = int(self.last_observation["active_count"])
        if active_count <= 0:
            return self.reset(), 0.0, False, dict(self.stats)

        sanitized = sanitize_slot_order(order_slots, self.last_observation["active_mask"])
        ordered_ugv_ids = slot_order_to_ugv_ids(sanitized, self.last_observation["slot_task_ids"])
        route_churn = compute_route_churn(sanitized, self.previous_order_slots, active_count)
        self.previous_order_slots = sanitized
        self.stats["route_churn"] += route_churn

        reward_info = {
            "hits": 0.0,
            "new_misses": 0.0,
            "late_served": 0.0,
            "tardiness": 0.0,
            "ongoing_tardiness": 0.0,
            "response_time": 0.0,
            "response_count": 0.0,
            "flight_distance": 0.0,
            "flight_time": 0.0,
            "route_churn": route_churn,
        }

        decision_until = min(self.horizon_sec, self.now_sec + self.replan_period, self._next_event_time())
        self._advance(decision_until, ordered_ugv_ids, reward_info)
        self._apply_events_until_now()

        done = self.now_sec >= self.horizon_sec - 1e-9
        if done:
            for item in list(self.active.values()):
                if not item.missed:
                    reward_info["new_misses"] += 1.0
                    self.stats["missed_count"] += 1.0
                if self.horizon_sec > item.task.deadline_sec:
                    tardiness = self.horizon_sec - item.task.deadline_sec
                    reward_info["tardiness"] += tardiness
                    self.stats["tardiness"] += tardiness
            self.active.clear()

        reward = self._reward(reward_info)
        if done:
            return None, reward, True, dict(self.stats)

        if not self.active and self.event_index < len(self.events):
            self.now_sec = max(self.now_sec, self.events[self.event_index][0])
            self._apply_events_until_now()

        if not self.active:
            return None, reward, True, dict(self.stats)

        return self._build_observation(), reward, False, dict(self.stats)


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
