#!/usr/bin/env python3
"""Wait until topo virtual obstacle cells reached every UGV map, then trigger."""

import argparse
import math
import sys
import time
from typing import Dict, Iterable, Set, Tuple

import rospy
import sensor_msgs.point_cloud2 as pc2
from custom_msgs.msg import map_info
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray


Point3 = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for UAV topo virtual obstacle cells to appear in every UGV broadcast "
            "grid_map.occu_address, then optionally publish /traj_start_trigger."
        )
    )
    parser.add_argument("--ugv-num", type=int, required=True)
    parser.add_argument("--drone-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--ready-frames", type=int, default=2)
    parser.add_argument("--min-closure-cells", type=int, default=100)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--stable-seconds", type=float, default=8.0)
    parser.add_argument("--required-ratio", type=float, default=0.98)
    parser.add_argument("--no-trigger", action="store_true")
    parser.add_argument("--trigger-repeat", type=int, default=5)
    parser.add_argument("--detour-enable", action="store_true")
    parser.add_argument("--detour-settle-seconds", type=float, default=2.0)
    return parser.parse_args()


def read_cloud_points(msg: PointCloud2) -> Tuple[Point3, ...]:
    return tuple(
        (float(x), float(y), float(z))
        for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    )


def wait_for_closure_points(
    topic: str,
    min_cells: int,
    stable_frames: int,
    stable_seconds: float,
    timeout: float,
) -> Tuple[Point3, ...]:
    deadline = time.monotonic() + timeout
    last_count = -1
    stable_count = 0
    last_change_time = time.monotonic()
    last_points: Tuple[Point3, ...] = tuple()
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        try:
            msg = rospy.wait_for_message(topic, PointCloud2, timeout=1.0)
        except rospy.ROSException:
            rospy.loginfo_throttle(5.0, "[TopoReady] waiting for %s", topic)
            continue

        points = read_cloud_points(msg)
        if len(points) >= min_cells and len(points) == last_count:
            stable_count += 1
        else:
            stable_count = 1
            last_count = len(points)
            last_change_time = time.monotonic()

        last_points = points
        stable_age = time.monotonic() - last_change_time
        if (len(points) >= min_cells and
            stable_count >= stable_frames and
            stable_age >= stable_seconds):
            rospy.loginfo(
                "[TopoReady] received stable topo obstacle cloud: %d points from %s, stable_age=%.1fs",
                len(points),
                topic,
                stable_age,
            )
            return points
        rospy.loginfo_throttle(
            5.0,
            "[TopoReady] topo obstacle cloud has %d points, waiting for >=%d, %d stable frames, %.1fs stable time",
            len(points),
            min_cells,
            stable_frames,
            stable_seconds,
        )

    raise TimeoutError(
        f"timed out waiting for stable topo obstacle cloud on {topic}; last_count={len(last_points)}"
    )


def address_for_point(point: Point3, msg: map_info) -> int:
    x, y, z = point
    ix = int(math.floor((x - msg.origin_x) / msg.resolution))
    iy = int(math.floor((y - msg.origin_y) / msg.resolution))
    iz = int(math.floor((z - msg.origin_z) / msg.resolution))
    if ix < 0 or iy < 0 or iz < 0:
        return -1
    if ix >= msg.size_x or iy >= msg.size_y or iz >= msg.size_z:
        return -1
    return ix * msg.size_y * msg.size_z + iy * msg.size_z + iz


def closure_addresses_for_msg(points: Iterable[Point3], msg: map_info) -> Set[int]:
    addresses = {address_for_point(point, msg) for point in points}
    addresses.discard(-1)
    return addresses


def publish_trigger(repeat: int) -> None:
    pub = rospy.Publisher("/traj_start_trigger", PoseStamped, queue_size=1, latch=True)
    rospy.sleep(0.5)

    trigger = PoseStamped()
    trigger.header.frame_id = ""
    trigger.pose.position.x = 0.0
    trigger.pose.position.y = 0.0
    trigger.pose.position.z = 0.0
    trigger.pose.orientation.x = 0.0
    trigger.pose.orientation.y = 0.0
    trigger.pose.orientation.z = 0.0
    trigger.pose.orientation.w = 0.0

    rate = rospy.Rate(5.0)
    for _ in range(max(1, repeat)):
        trigger.header.stamp = rospy.Time.now()
        pub.publish(trigger)
        rate.sleep()


def wait_for_detour_acks(
    ugv_num: int,
    drone_id: int,
    timeout: float,
    settle_seconds: float,
) -> None:
    latest_required = {"pairs": None}
    latest_ack: Dict[int, int] = {ugv_id: 0 for ugv_id in range(ugv_num)}

    def required_cb(msg: Int32MultiArray) -> None:
        pairs = []
        data = list(msg.data)
        for i in range(0, len(data) - 1, 2):
            ugv_id = int(data[i])
            seq = int(data[i + 1])
            if 0 <= ugv_id < ugv_num and seq > 0:
                pairs.append((ugv_id, seq))
        latest_required["pairs"] = pairs

    def make_ack_cb(ugv_id: int):
        def ack_cb(msg: Int32) -> None:
            latest_ack[ugv_id] = max(latest_ack[ugv_id], int(msg.data))
        return ack_cb

    subs = [
        rospy.Subscriber(f"/drone_{drone_id}/topo_detour_required", Int32MultiArray, required_cb, queue_size=1)
    ]
    for ugv_id in range(ugv_num):
        subs.append(
            rospy.Subscriber(
                f"/ugv_{ugv_id}/ego_planner_node/topo_route_ack",
                Int32,
                make_ack_cb(ugv_id),
                queue_size=5,
            )
        )

    start = time.monotonic()
    deadline = start + timeout
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        pairs = latest_required["pairs"]
        if pairs is None:
            rospy.loginfo_throttle(
                2.0,
                "[TopoReady] waiting for /drone_%d/topo_detour_required",
                drone_id,
            )
            rate.sleep()
            continue

        missing = [(ugv_id, seq, latest_ack.get(ugv_id, 0)) for ugv_id, seq in pairs
                   if latest_ack.get(ugv_id, 0) < seq]
        if not missing and time.monotonic() - start >= settle_seconds:
            if pairs:
                rospy.loginfo("[TopoReady] topo detour route acks ready: %s", pairs)
            else:
                rospy.loginfo("[TopoReady] no topo detour routes required")
            return

        rospy.loginfo_throttle(
            2.0,
            "[TopoReady] waiting detour acks, required=%s missing=%s",
            pairs,
            missing,
        )
        rate.sleep()

    raise TimeoutError("timed out waiting for topo detour route acknowledgements")


def main() -> int:
    args = parse_args()
    if args.ugv_num <= 0:
        raise ValueError("--ugv-num must be positive")

    rospy.init_node("topo_ready_trigger", anonymous=True)
    cloud_topic = f"/drone_{args.drone_id}/ego_planner_node/grid_map/topo_virtual_obstacle"
    closure_points = wait_for_closure_points(
        cloud_topic,
        min_cells=args.min_closure_cells,
        stable_frames=args.stable_frames,
        stable_seconds=args.stable_seconds,
        timeout=args.timeout,
    )

    ready_frames: Dict[int, int] = {ugv_id: 0 for ugv_id in range(args.ugv_num)}
    deadline = time.monotonic() + args.timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        all_ready = True
        for ugv_id in range(args.ugv_num):
            topic = f"/ugv_{ugv_id}/broadcast/grid_map"
            try:
                msg = rospy.wait_for_message(topic, map_info, timeout=1.0)
            except rospy.ROSException:
                rospy.loginfo_throttle(5.0, "[TopoReady] waiting for %s", topic)
                ready_frames[ugv_id] = 0
                all_ready = False
                continue

            closure_addresses = closure_addresses_for_msg(closure_points, msg)
            if not closure_addresses:
                ready_frames[ugv_id] = 0
                all_ready = False
                rospy.logwarn_throttle(
                    5.0,
                    "[TopoReady] could not convert closure cloud to map addresses for ugv_%d",
                    ugv_id,
                )
                continue

            occ_addresses = set(msg.occu_address)
            hit_count = len(closure_addresses & occ_addresses)
            ratio = hit_count / float(len(closure_addresses))
            if ratio >= args.required_ratio:
                ready_frames[ugv_id] += 1
            else:
                ready_frames[ugv_id] = 0

            if ready_frames[ugv_id] < args.ready_frames:
                all_ready = False

            rospy.loginfo_throttle(
                2.0,
                "[TopoReady] ugv_%d closure in occu_address: %d/%d %.1f%%, ready_frames=%d/%d",
                ugv_id,
                hit_count,
                len(closure_addresses),
                100.0 * ratio,
                ready_frames[ugv_id],
                args.ready_frames,
            )

        if all_ready:
            rospy.loginfo("[TopoReady] all %d UGV maps contain topo closure cells", args.ugv_num)
            if args.detour_enable:
                wait_for_detour_acks(args.ugv_num, args.drone_id, args.timeout, args.detour_settle_seconds)
            if not args.no_trigger:
                publish_trigger(args.trigger_repeat)
                rospy.loginfo("[TopoReady] published /traj_start_trigger")
            return 0

    raise TimeoutError("timed out waiting for topo closure cells in all UGV maps")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        rospy.logerr("[TopoReady] %s", exc)
        sys.exit(1)
