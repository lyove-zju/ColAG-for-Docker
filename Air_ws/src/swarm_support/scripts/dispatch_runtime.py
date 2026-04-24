#!/usr/bin/env python3
"""Shared runtime utilities for VRPTW-compatible UAV dispatch."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import Path
    from visualization_msgs.msg import Marker
except ImportError:  # pragma: no cover - training utilities may run without ROS installed
    Point = None
    PoseStamped = None
    Path = None
    Marker = None


DEFAULT_FLY_HEIGHT = 1.0
DEFAULT_TIME_RESOLUTION = 100
MAX_UGV_NUM = 10
GLOBAL_FEATURE_DIM = 6
NODE_FEATURE_DIM = 10


@dataclass
class DispatchTask:
    ugv_id: int
    collision_x: float
    collision_y: float
    collision_z: float
    deadline_sec: float

    def waypoint(self, fly_height: float = DEFAULT_FLY_HEIGHT) -> Tuple[float, float, float]:
        return (self.collision_x, self.collision_y, fly_height)


@dataclass
class DispatchMetrics:
    success_count: int
    missed_count: int
    tardiness: float
    route_time: float
    route_churn: float
    reward: float


def _float_list(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def _safe_distance(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


def compute_time_cost(
    c1: Sequence[float],
    c2: Sequence[float],
    v_max: float,
    time_resolution: int,
) -> int:
    distance = _safe_distance(c1[0] - c2[0], c1[1] - c2[1])
    if v_max <= 1e-6:
        return 0
    return int(distance / v_max * time_resolution)


def compute_time_cost_from_now(
    target: Sequence[float],
    pos: Sequence[float],
    vel: Sequence[float],
    v_max: float,
    a_max: float,
    time_resolution: int,
) -> int:
    """Mirror guide_plan.py exactly, including its 1D tangent approximation."""
    distance = _safe_distance(target[0] - pos[0], target[1] - pos[1])
    if distance <= 1e-9:
        return 0
    if v_max <= 1e-6 or a_max <= 1e-6:
        return int(distance * time_resolution)

    direct_x = (target[0] - pos[0]) / distance
    v_tangent = vel[0] * direct_x
    time1 = (v_max - v_tangent) / a_max
    distance1 = (v_max * v_max - v_tangent * v_tangent) / 2.0 * a_max

    if distance1 > distance:
        time_cost = (math.sqrt(v_tangent * v_tangent + 2.0 * a_max * distance) - v_tangent) / a_max
    else:
        time2 = (distance - distance1) / v_max
        time_cost = time1 + time2
    return int(time_cost * time_resolution)


class BlindTaskStore:
    def __init__(self, ugv_num: int, fly_height: float = DEFAULT_FLY_HEIGHT):
        self.ugv_num = ugv_num
        self.fly_height = fly_height
        self.deadlines = [0.0 for _ in range(ugv_num)]
        self.positions = [(0.0, 0.0, fly_height) for _ in range(ugv_num)]
        self.active = [False for _ in range(ugv_num)]

    def update_from_blind_info(self, msg) -> None:
        ugv_id = int(msg.id)
        if ugv_id < 0 or ugv_id >= self.ugv_num:
            return

        if float(msg.collision_time) == -1.0:
            self.deadlines[ugv_id] = 0.0
            self.positions[ugv_id] = (0.0, 0.0, self.fly_height)
            self.active[ugv_id] = False
            return

        self.deadlines[ugv_id] = float(msg.collision_time) + float(msg.header.stamp.to_sec())
        self.positions[ugv_id] = (float(msg.collision_x), float(msg.collision_y), self.fly_height)
        self.active[ugv_id] = True

    def active_tasks(self) -> List[DispatchTask]:
        tasks: List[DispatchTask] = []
        for ugv_id, enabled in enumerate(self.active):
            if not enabled:
                continue
            x, y, z = self.positions[ugv_id]
            tasks.append(
                DispatchTask(
                    ugv_id=ugv_id,
                    collision_x=x,
                    collision_y=y,
                    collision_z=z,
                    deadline_sec=self.deadlines[ugv_id],
                )
            )
        return tasks


class DispatchObservationBuilder:
    def __init__(
        self,
        ugv_num: int,
        v_max: float,
        a_max: float,
        time_resolution: int = DEFAULT_TIME_RESOLUTION,
        fly_height: float = DEFAULT_FLY_HEIGHT,
    ):
        if ugv_num > MAX_UGV_NUM:
            raise ValueError("dispatch v1 supports ugv_num <= 10.")
        self.ugv_num = ugv_num
        self.v_max = v_max
        self.a_max = a_max
        self.time_resolution = int(time_resolution)
        self.fly_height = fly_height

    def build_time_matrix(
        self,
        tasks: Sequence[DispatchTask],
        pos: Sequence[float],
        vel: Sequence[float],
    ) -> List[List[int]]:
        num = len(tasks)
        matrix = [[0 for _ in range(num + 1)] for _ in range(num + 1)]
        for i in range(num + 1):
            for j in range(i, num + 1):
                if i == j:
                    matrix[i][j] = 0
                elif i == 0:
                    task = tasks[j - 1]
                    matrix[i][j] = compute_time_cost_from_now(
                        (task.collision_x, task.collision_y, task.collision_z),
                        pos,
                        vel,
                        self.v_max,
                        self.a_max,
                        self.time_resolution,
                    )
                else:
                    t1 = tasks[i - 1]
                    t2 = tasks[j - 1]
                    time_cost = compute_time_cost(
                        (t1.collision_x, t1.collision_y, t1.collision_z),
                        (t2.collision_x, t2.collision_y, t2.collision_z),
                        self.v_max,
                        self.time_resolution,
                    )
                    matrix[i][j] = time_cost
                    matrix[j][i] = time_cost
        return matrix

    def build_time_windows(self, tasks: Sequence[DispatchTask], now_sec: float) -> List[List[int]]:
        windows = [[0, 0] for _ in range(len(tasks) + 1)]
        for idx, task in enumerate(tasks, start=1):
            windows[idx][1] = int((task.deadline_sec - now_sec) * self.time_resolution)
        return windows

    def apply_original_time_bias(
        self,
        tasks: Sequence[DispatchTask],
        time_windows: List[List[int]],
        pos: Sequence[float],
        vel: Sequence[float],
    ) -> Dict[str, int]:
        time_bias = 0
        last_id = -1
        for idx in range(len(tasks)):
            if time_windows[idx + 1][1] <= time_bias:
                time_bias = time_windows[idx + 1][1]
                last_id = idx

        applied_bias = 0
        if time_bias <= 0 and last_id >= 0:
            task = tasks[last_id]
            applied_bias = compute_time_cost_from_now(
                (task.collision_x, task.collision_y, task.collision_z),
                pos,
                vel,
                self.v_max,
                self.a_max,
                self.time_resolution,
            ) - time_bias
            for idx in range(len(tasks)):
                time_windows[idx + 1][1] = time_windows[idx + 1][1] + applied_bias + 1

        return {"time_bias": applied_bias, "last_id": last_id}

    def build_observation(
        self,
        tasks: Sequence[DispatchTask],
        pos: Sequence[float],
        vel: Sequence[float],
        now_sec: float,
        previous_order_slots: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        sorted_tasks = sorted(tasks, key=lambda item: item.ugv_id)
        active_count = len(sorted_tasks)
        if active_count > self.ugv_num:
            raise ValueError("active task count exceeds configured ugv_num.")

        time_matrix = self.build_time_matrix(sorted_tasks, pos, vel)
        time_windows = self.build_time_windows(sorted_tasks, now_sec)
        bias_info = self.apply_original_time_bias(sorted_tasks, time_windows, pos, vel)

        global_features = [
            float(pos[0]),
            float(pos[1]),
            float(vel[0]),
            float(vel[1]),
            math.sqrt(float(vel[0]) * float(vel[0]) + float(vel[1]) * float(vel[1])),
            float(active_count),
        ]

        node_features: List[List[float]] = []
        pairwise_time_matrix: List[List[float]] = []
        active_mask: List[int] = []
        slot_task_ids: List[int] = []
        slot_waypoints: List[Optional[List[float]]] = []

        for slot in range(self.ugv_num):
            if slot < active_count:
                task = sorted_tasks[slot]
                dx = task.collision_x - float(pos[0])
                dy = task.collision_y - float(pos[1])
                distance = _safe_distance(dx, dy)
                deadline_remaining = time_windows[slot + 1][1] / float(self.time_resolution)
                travel_time_from_now = time_matrix[0][slot + 1] / float(self.time_resolution)
                slack = deadline_remaining - travel_time_from_now
                node_features.append(
                    [
                        1.0,
                        float(task.ugv_id),
                        float(task.collision_x),
                        float(task.collision_y),
                        float(dx),
                        float(dy),
                        float(distance),
                        float(deadline_remaining),
                        float(travel_time_from_now),
                        float(slack),
                    ]
                )
                pairwise_row = [
                    time_matrix[slot + 1][other + 1] / float(self.time_resolution)
                    if other < active_count
                    else 0.0
                    for other in range(self.ugv_num)
                ]
                pairwise_time_matrix.append(pairwise_row)
                active_mask.append(1)
                slot_task_ids.append(task.ugv_id)
                slot_waypoints.append(_float_list(task.waypoint(self.fly_height)))
            else:
                node_features.append([0.0 for _ in range(NODE_FEATURE_DIM)])
                pairwise_time_matrix.append([0.0 for _ in range(self.ugv_num)])
                active_mask.append(0)
                slot_task_ids.append(-1)
                slot_waypoints.append(None)

        return {
            "global_features": global_features,
            "node_features": node_features,
            "pairwise_time_matrix": pairwise_time_matrix,
            "active_mask": active_mask,
            "active_count": active_count,
            "active_ugv_ids": [task.ugv_id for task in sorted_tasks],
            "slot_task_ids": slot_task_ids,
            "slot_waypoints": slot_waypoints,
            "time_matrix": time_matrix,
            "time_windows": time_windows,
            "time_bias": bias_info["time_bias"],
            "previous_order_slots": list(previous_order_slots or []),
            "self_pos": _float_list(pos),
            "self_vel": _float_list(vel),
        }


def sanitize_slot_order(order_slots: Iterable[int], active_mask: Sequence[int]) -> List[int]:
    valid_slots = [idx for idx, enabled in enumerate(active_mask) if enabled]
    seen = set()
    sanitized: List[int] = []
    for raw_slot in order_slots:
        slot = int(raw_slot)
        if slot in seen:
            continue
        if slot < 0 or slot >= len(active_mask):
            continue
        if not active_mask[slot]:
            continue
        sanitized.append(slot)
        seen.add(slot)

    for slot in valid_slots:
        if slot not in seen:
            sanitized.append(slot)
            seen.add(slot)

    return sanitized


def slot_order_to_ugv_ids(order_slots: Sequence[int], slot_task_ids: Sequence[int]) -> List[int]:
    return [slot_task_ids[slot] for slot in order_slots if 0 <= slot < len(slot_task_ids) and slot_task_ids[slot] >= 0]


def slot_order_to_waypoints(
    order_slots: Sequence[int],
    slot_waypoints: Sequence[Optional[Sequence[float]]],
) -> List[Tuple[float, float, float]]:
    result: List[Tuple[float, float, float]] = []
    for slot in order_slots:
        waypoint = slot_waypoints[slot]
        if waypoint is None:
            continue
        result.append((float(waypoint[0]), float(waypoint[1]), float(waypoint[2])))
    return result


def build_path_msg(waypoints: Sequence[Tuple[float, float, float]]):
    if Path is None or PoseStamped is None:
        raise RuntimeError("ROS Path message types are unavailable in this environment.")
    path_msg = Path()
    path_msg.header.frame_id = "world"
    path_msg.header.stamp = _ros_now()
    for idx, waypoint in enumerate(waypoints):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = _ros_time_from_sec(_ros_now_sec() + idx)
        pose.pose.position.x = waypoint[0]
        pose.pose.position.y = waypoint[1]
        pose.pose.position.z = waypoint[2]
        path_msg.poses.append(pose)
    return path_msg


def build_waypoint_marker(waypoints: Sequence[Tuple[float, float, float]]):
    if Marker is None or Point is None:
        raise RuntimeError("ROS Marker message types are unavailable in this environment.")
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = _ros_now()
    marker.action = Marker.ADD
    marker.type = Marker.SPHERE_LIST
    marker.id = 0
    marker.color.r = 0.1
    marker.color.g = 0.3
    marker.color.b = 0.9
    marker.color.a = 1.0
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.pose.orientation.w = 1.0
    for waypoint in waypoints:
        pt = Point()
        pt.x = waypoint[0]
        pt.y = waypoint[1]
        pt.z = waypoint[2]
        marker.points.append(pt)
    return marker


def evaluate_order_metrics(
    order_slots: Sequence[int],
    observation: Dict[str, object],
    reward_config: Optional[Dict[str, float]] = None,
) -> DispatchMetrics:
    reward_cfg = {
        "success_reward": 1.0,
        "miss_penalty": -1.5,
        "tardiness_weight": -0.05,
        "route_time_weight": -0.01,
        "route_churn_weight": -0.01,
    }
    if reward_config:
        reward_cfg.update(reward_config)

    active_mask = observation["active_mask"]
    time_matrix = observation["time_matrix"]
    time_windows = observation["time_windows"]
    previous_order = observation.get("previous_order_slots") or []
    active_count = int(observation["active_count"])
    order = sanitize_slot_order(order_slots, active_mask)

    accum_time = 0
    prev_node = 0
    success_count = 0
    missed_count = 0
    tardiness = 0.0
    for slot in order:
        node = slot + 1
        accum_time += time_matrix[prev_node][node]
        deadline = time_windows[node][1]
        if accum_time <= deadline:
            success_count += 1
        else:
            missed_count += 1
            tardiness += (accum_time - deadline) / float(DEFAULT_TIME_RESOLUTION)
        prev_node = node

    deadline_budget = sum(max(window[1], 0) for window in time_windows[1:]) / float(DEFAULT_TIME_RESOLUTION)
    route_time = accum_time / float(DEFAULT_TIME_RESOLUTION)
    normalized_tardiness = tardiness / max(1.0, deadline_budget)
    normalized_route_time = route_time / max(1.0, deadline_budget)
    route_churn = compute_route_churn(order, previous_order, active_count)

    reward = (
        reward_cfg["success_reward"] * success_count
        + reward_cfg["miss_penalty"] * missed_count
        + reward_cfg["tardiness_weight"] * normalized_tardiness
        + reward_cfg["route_time_weight"] * normalized_route_time
        + reward_cfg["route_churn_weight"] * route_churn
    )
    return DispatchMetrics(
        success_count=success_count,
        missed_count=missed_count,
        tardiness=tardiness,
        route_time=route_time,
        route_churn=route_churn,
        reward=reward,
    )


def compute_route_churn(
    current_order: Sequence[int],
    previous_order: Sequence[int],
    active_count: int,
) -> float:
    if active_count <= 1 or not previous_order:
        return 0.0
    current = sanitize_slot_order(current_order, [1] * max(active_count, len(current_order)))
    previous = sanitize_slot_order(previous_order, [1] * max(active_count, len(previous_order)))
    length = min(len(current), len(previous))
    if length == 0:
        return 0.0
    mismatch = 0
    for idx in range(length):
        if current[idx] != previous[idx]:
            mismatch += 1
    mismatch += abs(len(current) - len(previous))
    return float(mismatch) / float(max(len(current), len(previous), 1))


def random_permutation(active_mask: Sequence[int], rng: random.Random) -> List[int]:
    slots = [idx for idx, enabled in enumerate(active_mask) if enabled]
    rng.shuffle(slots)
    return slots


def observation_to_jsonable(observation: Dict[str, object]) -> Dict[str, object]:
    serializable = {}
    for key, value in observation.items():
        if isinstance(value, dict):
            serializable[key] = value
        elif isinstance(value, list):
            serializable[key] = value
        else:
            serializable[key] = value
    return serializable


def dumps_jsonl(records: Sequence[Dict[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True))
            fh.write("\n")


def _ros_now():
    import rospy

    return rospy.Time.now()


def _ros_now_sec() -> float:
    import rospy

    return float(rospy.Time.now().to_sec())


def _ros_time_from_sec(value: float):
    import rospy

    return rospy.Time.from_sec(value)
