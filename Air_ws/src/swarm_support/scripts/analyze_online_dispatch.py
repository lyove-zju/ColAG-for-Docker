#!/usr/bin/env python3
"""Analyze online UAV dispatch performance from rosbag recordings.

This tool is intended for comparing the online VRPTW baseline and the RL
replacement in the real MARSIM/ROS closed loop. It operates on rosbag files and
uses the same runtime request source as the dispatch nodes: `blind_info`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def require_rosbag():
    try:
        import rosbag
    except ImportError as exc:  # pragma: no cover - only available in ROS envs
        raise RuntimeError(
            "rosbag Python API is unavailable. Run this script inside the ROS Noetic environment."
        ) from exc
    return rosbag


def maybe_header_stamp(msg, fallback_time: float) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback_time
    try:
        value = float(stamp.to_sec())
    except Exception:
        return fallback_time
    if value <= 0.0:
        return fallback_time
    return value


def path_length(points: Sequence[Tuple[float, float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for p1, p2 in zip(points[:-1], points[1:]):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


@dataclass
class SupportRequest:
    request_index: int
    ugv_id: int
    start_time: float
    deadline_time: float
    target_x: float
    target_y: float
    target_z: float
    end_time: Optional[float] = None
    end_reason: str = ""
    arrival_time: Optional[float] = None
    arrival_distance: Optional[float] = None
    closest_distance: float = float("inf")
    closest_time: Optional[float] = None

    def update_closest(self, distance_xy: float, timestamp: float) -> None:
        if distance_xy < self.closest_distance:
            self.closest_distance = distance_xy
            self.closest_time = timestamp

    def maybe_mark_arrival(self, distance_xy: float, timestamp: float, radius: float) -> None:
        self.update_closest(distance_xy, timestamp)
        if self.arrival_time is None and distance_xy <= radius:
            self.arrival_time = timestamp
            self.arrival_distance = distance_xy

    def close(self, end_time: float, reason: str) -> None:
        if self.end_time is None:
            self.end_time = end_time
            self.end_reason = reason

    def response_time(self) -> Optional[float]:
        if self.arrival_time is None:
            return None
        return self.arrival_time - self.start_time

    def is_deadline_hit(self) -> bool:
        return self.arrival_time is not None and self.arrival_time <= self.deadline_time

    def is_served_late(self) -> bool:
        return self.arrival_time is not None and self.arrival_time > self.deadline_time

    def tardiness(self) -> float:
        terminal_time = self.arrival_time if self.arrival_time is not None else self.end_time
        if terminal_time is None:
            return 0.0
        return max(0.0, terminal_time - self.deadline_time)

    def to_summary(self) -> Dict[str, object]:
        return {
            "request_index": self.request_index,
            "ugv_id": self.ugv_id,
            "start_time": self.start_time,
            "deadline_time": self.deadline_time,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_z": self.target_z,
            "end_time": self.end_time,
            "end_reason": self.end_reason,
            "arrival_time": self.arrival_time,
            "arrival_distance": self.arrival_distance,
            "closest_distance": None if math.isinf(self.closest_distance) else self.closest_distance,
            "closest_time": self.closest_time,
            "response_time": self.response_time(),
            "deadline_hit": self.is_deadline_hit(),
            "served_late": self.is_served_late(),
            "tardiness": self.tardiness(),
        }


@dataclass
class BagAnalysis:
    bag_path: str
    blind_topic: str
    odom_topic: str
    arrival_radius: float
    fly_height: float
    total_requests: int
    served_count: int
    deadline_hit_count: int
    missed_count: int
    late_served_count: int
    unserved_count: int
    cleared_unserved_count: int
    replaced_unserved_count: int
    bag_end_unserved_count: int
    deadline_hit_rate: float
    avg_tardiness_seconds: float
    avg_response_time_seconds: float
    avg_hit_response_time_seconds: float
    uav_flight_distance_m: float
    uav_flight_time_seconds: float
    odom_sample_count: int
    blind_message_count: int
    first_request_time: Optional[float]
    last_request_time: Optional[float]
    analysis_start_time: Optional[float]
    analysis_end_time: Optional[float]
    requests: List[Dict[str, object]]


def summarize_requests(
    requests: Sequence[SupportRequest],
    bag_path: str,
    blind_topic: str,
    odom_topic: str,
    arrival_radius: float,
    fly_height: float,
    flight_distance: float,
    flight_time: float,
    odom_sample_count: int,
    blind_message_count: int,
    analysis_start_time: Optional[float],
    analysis_end_time: Optional[float],
) -> BagAnalysis:
    total_requests = len(requests)
    served = [req for req in requests if req.arrival_time is not None]
    hits = [req for req in requests if req.is_deadline_hit()]
    late_served = [req for req in requests if req.is_served_late()]
    unserved = [req for req in requests if req.arrival_time is None]
    positive_tardiness = [req.tardiness() for req in requests if req.tardiness() > 1e-9]
    response_times = [req.response_time() for req in served if req.response_time() is not None]
    hit_response_times = [req.response_time() for req in hits if req.response_time() is not None]

    cleared_unserved = [req for req in unserved if req.end_reason == "cleared"]
    replaced_unserved = [req for req in unserved if req.end_reason == "replaced"]
    bag_end_unserved = [req for req in unserved if req.end_reason == "bag_end"]

    return BagAnalysis(
        bag_path=bag_path,
        blind_topic=blind_topic,
        odom_topic=odom_topic,
        arrival_radius=arrival_radius,
        fly_height=fly_height,
        total_requests=total_requests,
        served_count=len(served),
        deadline_hit_count=len(hits),
        missed_count=total_requests - len(hits),
        late_served_count=len(late_served),
        unserved_count=len(unserved),
        cleared_unserved_count=len(cleared_unserved),
        replaced_unserved_count=len(replaced_unserved),
        bag_end_unserved_count=len(bag_end_unserved),
        deadline_hit_rate=float(len(hits)) / float(total_requests) if total_requests else 0.0,
        avg_tardiness_seconds=mean(positive_tardiness) if positive_tardiness else 0.0,
        avg_response_time_seconds=mean(response_times) if response_times else 0.0,
        avg_hit_response_time_seconds=mean(hit_response_times) if hit_response_times else 0.0,
        uav_flight_distance_m=flight_distance,
        uav_flight_time_seconds=flight_time,
        odom_sample_count=odom_sample_count,
        blind_message_count=blind_message_count,
        first_request_time=requests[0].start_time if requests else None,
        last_request_time=requests[-1].start_time if requests else None,
        analysis_start_time=analysis_start_time,
        analysis_end_time=analysis_end_time,
        requests=[req.to_summary() for req in requests],
    )


def analyze_bag(
    bag_path: Path,
    blind_topic: str,
    odom_topic: str,
    arrival_radius: float,
    fly_height: float,
) -> BagAnalysis:
    rosbag = require_rosbag()

    active_requests: Dict[int, SupportRequest] = {}
    closed_requests: List[SupportRequest] = []
    request_index = 0
    last_time: Optional[float] = None
    odom_points: List[Tuple[float, float, float]] = []
    odom_times: List[float] = []
    blind_message_count = 0

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[blind_topic, odom_topic]):
            bag_time = float(t.to_sec())
            timestamp = maybe_header_stamp(msg, bag_time)
            if last_time is None or timestamp > last_time:
                last_time = timestamp

            if topic == blind_topic:
                blind_message_count += 1
                ugv_id = int(msg.id)
                if ugv_id < 0:
                    continue

                old_request = active_requests.pop(ugv_id, None)
                if old_request is not None:
                    old_request.close(timestamp, "replaced" if float(msg.collision_time) != -1.0 else "cleared")
                    closed_requests.append(old_request)

                if float(msg.collision_time) == -1.0:
                    continue

                request_index += 1
                active_requests[ugv_id] = SupportRequest(
                    request_index=request_index,
                    ugv_id=ugv_id,
                    start_time=timestamp,
                    deadline_time=timestamp + float(msg.collision_time),
                    target_x=float(msg.collision_x),
                    target_y=float(msg.collision_y),
                    target_z=fly_height,
                )
                continue

            if topic == odom_topic:
                pos = (
                    float(msg.pose.pose.position.x),
                    float(msg.pose.pose.position.y),
                    float(msg.pose.pose.position.z),
                )
                odom_points.append(pos)
                odom_times.append(timestamp)
                for request in active_requests.values():
                    distance_xy = math.hypot(pos[0] - request.target_x, pos[1] - request.target_y)
                    request.maybe_mark_arrival(distance_xy, timestamp, arrival_radius)

    if last_time is None:
        last_time = 0.0

    for request in active_requests.values():
        request.close(last_time, "bag_end")
        closed_requests.append(request)

    closed_requests.sort(key=lambda item: (item.start_time, item.request_index))
    analysis = summarize_requests(
        closed_requests,
        bag_path=str(bag_path),
        blind_topic=blind_topic,
        odom_topic=odom_topic,
        arrival_radius=arrival_radius,
        fly_height=fly_height,
        flight_distance=path_length(odom_points),
        flight_time=(odom_times[-1] - odom_times[0]) if len(odom_times) >= 2 else 0.0,
        odom_sample_count=len(odom_points),
        blind_message_count=blind_message_count,
        analysis_start_time=min((odom_times[0], closed_requests[0].start_time)) if odom_times and closed_requests else (odom_times[0] if odom_times else (closed_requests[0].start_time if closed_requests else None)),
        analysis_end_time=last_time,
    )
    return analysis


def print_analysis(label: str, analysis: BagAnalysis) -> None:
    print(f"[{label}] bag={analysis.bag_path}")
    print(f"  requests={analysis.total_requests} served={analysis.served_count} hits={analysis.deadline_hit_count} misses={analysis.missed_count}")
    print(f"  deadline_hit_rate={analysis.deadline_hit_rate:.4f}")
    print(f"  late_served={analysis.late_served_count} unserved={analysis.unserved_count}")
    print(
        "  unserved_breakdown="
        f"cleared:{analysis.cleared_unserved_count}, "
        f"replaced:{analysis.replaced_unserved_count}, "
        f"bag_end:{analysis.bag_end_unserved_count}"
    )
    print(f"  avg_tardiness_seconds={analysis.avg_tardiness_seconds:.4f}")
    print(f"  avg_response_time_seconds={analysis.avg_response_time_seconds:.4f}")
    print(f"  avg_hit_response_time_seconds={analysis.avg_hit_response_time_seconds:.4f}")
    print(f"  uav_flight_distance_m={analysis.uav_flight_distance_m:.4f}")
    print(f"  uav_flight_time_seconds={analysis.uav_flight_time_seconds:.4f}")


def compute_compare_delta(baseline: BagAnalysis, candidate: BagAnalysis) -> Dict[str, float]:
    return {
        "deadline_hit_rate_delta": candidate.deadline_hit_rate - baseline.deadline_hit_rate,
        "deadline_hit_count_delta": candidate.deadline_hit_count - baseline.deadline_hit_count,
        "missed_count_delta": candidate.missed_count - baseline.missed_count,
        "avg_tardiness_seconds_delta": candidate.avg_tardiness_seconds - baseline.avg_tardiness_seconds,
        "avg_response_time_seconds_delta": candidate.avg_response_time_seconds - baseline.avg_response_time_seconds,
        "uav_flight_distance_m_delta": candidate.uav_flight_distance_m - baseline.uav_flight_distance_m,
        "uav_flight_time_seconds_delta": candidate.uav_flight_time_seconds - baseline.uav_flight_time_seconds,
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    blind_topic = args.blind_topic or f"/drone_{args.drone_id}/broadcast/blind_info"
    odom_topic = args.odom_topic or f"/drone_{args.drone_id}/lidar_slam/odom"
    analysis = analyze_bag(
        bag_path=Path(args.bag).expanduser().resolve(),
        blind_topic=blind_topic,
        odom_topic=odom_topic,
        arrival_radius=args.arrival_radius,
        fly_height=args.fly_height,
    )
    print_analysis(args.label, analysis)
    if args.output_json:
        payload = asdict(analysis)
        Path(args.output_json).expanduser().resolve().write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    blind_topic = args.blind_topic or f"/drone_{args.drone_id}/broadcast/blind_info"
    odom_topic = args.odom_topic or f"/drone_{args.drone_id}/lidar_slam/odom"
    baseline = analyze_bag(
        bag_path=Path(args.baseline_bag).expanduser().resolve(),
        blind_topic=blind_topic,
        odom_topic=odom_topic,
        arrival_radius=args.arrival_radius,
        fly_height=args.fly_height,
    )
    candidate = analyze_bag(
        bag_path=Path(args.candidate_bag).expanduser().resolve(),
        blind_topic=blind_topic,
        odom_topic=odom_topic,
        arrival_radius=args.arrival_radius,
        fly_height=args.fly_height,
    )

    print_analysis(args.baseline_label, baseline)
    print_analysis(args.candidate_label, candidate)
    delta = compute_compare_delta(baseline, candidate)
    print("[delta] candidate - baseline")
    for key, value in delta.items():
        print(f"  {key}={value:.4f}")

    if args.output_json:
        payload = {
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "baseline": asdict(baseline),
            "candidate": asdict(candidate),
            "delta": delta,
        }
        Path(args.output_json).expanduser().resolve().write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze online UAV dispatch metrics from rosbag recordings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one rosbag and print online dispatch metrics.")
    analyze.add_argument("--bag", required=True, help="Path to a rosbag file recorded from one run.")
    analyze.add_argument("--label", default="run", help="Label shown in console output. Default: run")
    analyze.add_argument("--drone-id", type=int, default=0)
    analyze.add_argument("--blind-topic", default="", help="Override blind_info topic.")
    analyze.add_argument("--odom-topic", default="", help="Override UAV odom topic.")
    analyze.add_argument("--arrival-radius", type=float, default=0.8, help="XY arrival radius in meters. Default: 0.8")
    analyze.add_argument("--fly-height", type=float, default=1.0, help="Support waypoint height used by dispatch. Default: 1.0")
    analyze.add_argument("--output-json", default="", help="Optional JSON output path.")
    analyze.set_defaults(func=cmd_analyze)

    compare = subparsers.add_parser("compare", help="Analyze and compare two rosbag runs.")
    compare.add_argument("--baseline-bag", required=True, help="Reference rosbag, e.g. VRPTW run.")
    compare.add_argument("--candidate-bag", required=True, help="Candidate rosbag, e.g. RL run.")
    compare.add_argument("--baseline-label", default="baseline")
    compare.add_argument("--candidate-label", default="candidate")
    compare.add_argument("--drone-id", type=int, default=0)
    compare.add_argument("--blind-topic", default="", help="Override blind_info topic.")
    compare.add_argument("--odom-topic", default="", help="Override UAV odom topic.")
    compare.add_argument("--arrival-radius", type=float, default=0.8, help="XY arrival radius in meters. Default: 0.8")
    compare.add_argument("--fly-height", type=float, default=1.0, help="Support waypoint height used by dispatch. Default: 1.0")
    compare.add_argument("--output-json", default="", help="Optional JSON output path.")
    compare.set_defaults(func=cmd_compare)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
