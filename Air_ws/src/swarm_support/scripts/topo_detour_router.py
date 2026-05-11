#!/usr/bin/env python3
"""Generate UGV detour waypoints around confirmed topo closure walls."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray, Int32, Int32MultiArray
from visualization_msgs.msg import Marker


Vec2 = Tuple[float, float]


@dataclass
class Closure:
    closure_id: int
    center: Vec2
    wall_dir: Vec2
    branch_dir: Vec2
    left_width: float
    right_width: float
    thickness: float
    z_min: float
    z_max: float
    cell_count: int
    branch_back_depth: float
    branch_front_depth: float


def add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def normalize(a: Vec2, fallback: Vec2) -> Vec2:
    length = norm(a)
    if length < 1e-6:
        return fallback
    return (a[0] / length, a[1] / length)


def dist(a: Vec2, b: Vec2) -> float:
    return norm(sub(a, b))


def point3(xy: Vec2, z: float) -> Point:
    p = Point()
    p.x = xy[0]
    p.y = xy[1]
    p.z = z
    return p


class TopoDetourRouter:
    ROW_SIZE = 15

    def __init__(self) -> None:
        self.drone_id = rospy.get_param("~drone_id", 0)
        self.ugv_num = rospy.get_param("~ugv_num", 1)
        self.period = rospy.get_param("~period", 0.5)
        self.route_z = rospy.get_param("~route_z", 0.3)
        self.side_clearance = rospy.get_param("~side_clearance", 1.6)
        self.front_clearance = rospy.get_param("~front_clearance", 2.0)
        self.rear_clearance = rospy.get_param("~rear_clearance", 1.2)
        self.intersection_margin = rospy.get_param("~intersection_margin", 0.45)
        self.min_route_change = rospy.get_param("~min_route_change", 0.25)
        self.map_min_x = rospy.get_param("~map_min_x", -20.0)
        self.map_max_x = rospy.get_param("~map_max_x", 20.0)
        self.map_min_y = rospy.get_param("~map_min_y", -20.0)
        self.map_max_y = rospy.get_param("~map_max_y", 20.0)

        self.closures: List[Closure] = []
        self.odom: Dict[int, Vec2] = {}
        self.goals: Dict[int, Vec2] = {}
        self.goal_z: Dict[int, float] = {}
        self.route_seq: Dict[int, int] = {ugv_id: 0 for ugv_id in range(self.ugv_num)}
        self.last_routes: Dict[int, List[Vec2]] = {}
        self.last_required: Dict[int, int] = {}
        self.acks: Dict[int, int] = {}

        self.path_pubs = [
            rospy.Publisher(
                f"/ugv_{ugv_id}/ego_planner_node/topo_detour_path",
                Path,
                queue_size=1,
                latch=True,
            )
            for ugv_id in range(self.ugv_num)
        ]
        self.required_pub = rospy.Publisher("topo_detour_required", Int32MultiArray, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("topo_detour_route_visual", Marker, queue_size=1)

        rospy.Subscriber("ego_planner_node/grid_map/topo_closures", Float32MultiArray, self.closure_callback, queue_size=1)
        rospy.Subscriber("broadcast/ugv_odom", Odometry, self.odom_callback, queue_size=50)
        for ugv_id in range(self.ugv_num):
            rospy.Subscriber(
                f"/ugv_{ugv_id}/ego_planner_node/topo_current_goal",
                PoseStamped,
                self.goal_callback,
                callback_args=ugv_id,
                queue_size=1,
            )
            rospy.Subscriber(
                f"/ugv_{ugv_id}/ego_planner_node/topo_route_ack",
                Int32,
                self.ack_callback,
                callback_args=ugv_id,
                queue_size=5,
            )

        self.timer = rospy.Timer(rospy.Duration(max(0.1, self.period)), self.timer_callback)
        rospy.logwarn("[TopoDetour] enabled for %d UGVs", self.ugv_num)

    def closure_callback(self, msg: Float32MultiArray) -> None:
        row_size = self.ROW_SIZE
        if msg.layout.dim and len(msg.layout.dim) >= 2 and msg.layout.dim[1].size:
            row_size = int(msg.layout.dim[1].size)
        if row_size < 13:
            rospy.logwarn_throttle(3.0, "[TopoDetour] ignored topo_closures with row_size=%d", row_size)
            return

        closures: List[Closure] = []
        data = list(msg.data)
        for offset in range(0, len(data) - row_size + 1, row_size):
            row = data[offset:offset + row_size]
            wall_dir = normalize((row[3], row[4]), (1.0, 0.0))
            branch_dir = normalize((row[5], row[6]), (0.0, 1.0))
            closures.append(
                Closure(
                    closure_id=int(round(row[0])),
                    center=(row[1], row[2]),
                    wall_dir=wall_dir,
                    branch_dir=branch_dir,
                    left_width=max(0.0, row[7]),
                    right_width=max(0.0, row[8]),
                    thickness=max(0.1, row[9]),
                    z_min=row[10],
                    z_max=row[11],
                    cell_count=int(round(row[12])),
                    branch_back_depth=max(0.0, row[13]) if row_size > 13 else 0.0,
                    branch_front_depth=max(0.0, row[14]) if row_size > 14 else 0.0,
                )
            )
        self.closures = closures

    def odom_callback(self, msg: Odometry) -> None:
        ugv_id = self.parse_ugv_id(msg.child_frame_id)
        if ugv_id is None or ugv_id >= self.ugv_num:
            return
        self.odom[ugv_id] = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def goal_callback(self, msg: PoseStamped, ugv_id: int) -> None:
        self.goals[ugv_id] = (msg.pose.position.x, msg.pose.position.y)
        self.goal_z[ugv_id] = msg.pose.position.z

    def ack_callback(self, msg: Int32, ugv_id: int) -> None:
        self.acks[ugv_id] = msg.data

    @staticmethod
    def parse_ugv_id(child_frame_id: str) -> Optional[int]:
        if not child_frame_id.startswith("ugv_"):
            return None
        try:
            return int(child_frame_id.split("_", 1)[1])
        except ValueError:
            return None

    def timer_callback(self, _event) -> None:
        required: Dict[int, int] = {}
        for ugv_id in range(self.ugv_num):
            if ugv_id not in self.odom or ugv_id not in self.goals or not self.closures:
                continue
            route = self.compute_route(self.odom[ugv_id], self.goals[ugv_id])
            if route is None:
                continue
            if self.should_publish_route(ugv_id, route):
                self.route_seq[ugv_id] += 1
                self.last_routes[ugv_id] = route
                self.publish_route(ugv_id, route, self.route_seq[ugv_id])
            required[ugv_id] = self.route_seq[ugv_id]

        self.last_required = required
        self.publish_required(required)
        self.publish_markers()

    def compute_route(self, start: Vec2, goal: Vec2) -> Optional[List[Vec2]]:
        blocking = self.blocking_closures(start, goal)
        if not blocking:
            return None

        travel = normalize(sub(goal, start), (0.0, -1.0))
        blocking.sort(key=lambda closure: dot(sub(closure.center, start), travel))

        candidates = []
        for side in (-1.0, 1.0):
            route = self.build_side_route(start, goal, blocking, side, travel)
            if route is None:
                continue
            if self.route_intersects_any([start] + route):
                continue
            candidates.append((self.route_score(start, route), route))

        if not candidates:
            rospy.logwarn_throttle(2.0, "[TopoDetour] no valid side route around %d closure(s)", len(blocking))
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def blocking_closures(self, start: Vec2, goal: Vec2) -> List[Closure]:
        return [
            closure
            for closure in self.closures
            if self.segment_hits_closure(start, goal, closure, self.intersection_margin)
        ]

    def build_side_route(
        self,
        start: Vec2,
        goal: Vec2,
        closures: Sequence[Closure],
        side: float,
        travel: Vec2,
    ) -> Optional[List[Vec2]]:
        route: List[Vec2] = []
        for closure in closures:
            lateral_width = closure.left_width if side > 0.0 else closure.right_width
            lateral = mul(closure.wall_dir, side * (lateral_width + self.side_clearance))
            front_depth = max(closure.branch_front_depth, closure.thickness * 0.5) + self.front_clearance
            back_depth = max(closure.branch_back_depth, closure.thickness * 0.5) + self.rear_clearance
            p_front = add(add(closure.center, lateral), mul(closure.branch_dir, front_depth))
            p_back = add(add(closure.center, lateral), mul(closure.branch_dir, -back_depth))
            pair = [p_front, p_back]
            pair.sort(key=lambda p: dot(sub(p, start), travel))
            for p in pair:
                if not self.point_in_bounds(p):
                    return None
                if not route or dist(route[-1], p) > 0.2:
                    route.append(p)
        route.append(goal)
        return route

    def route_intersects_any(self, points: Sequence[Vec2]) -> bool:
        for a, b in zip(points[:-1], points[1:]):
            for closure in self.closures:
                if self.segment_hits_closure(a, b, closure, self.intersection_margin):
                    return True
        return False

    def segment_hits_closure(self, a: Vec2, b: Vec2, closure: Closure, margin: float) -> bool:
        ax, ay = self.to_closure_frame(a, closure)
        bx, by = self.to_closure_frame(b, closure)
        min_x = -closure.right_width - margin
        max_x = closure.left_width + margin
        half_t = closure.thickness * 0.5 + margin
        return self.segment_intersects_aabb((ax, ay), (bx, by), min_x, max_x, -half_t, half_t)

    @staticmethod
    def segment_intersects_aabb(a: Vec2, b: Vec2, min_x: float, max_x: float, min_y: float, max_y: float) -> bool:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        t0 = 0.0
        t1 = 1.0
        for p, q in ((-dx, a[0] - min_x), (dx, max_x - a[0]), (-dy, a[1] - min_y), (dy, max_y - a[1])):
            if abs(p) < 1e-9:
                if q < 0.0:
                    return False
                continue
            r = q / p
            if p < 0.0:
                if r > t1:
                    return False
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                if r < t1:
                    t1 = r
        return True

    @staticmethod
    def to_closure_frame(point: Vec2, closure: Closure) -> Vec2:
        delta = sub(point, closure.center)
        return (dot(delta, closure.wall_dir), dot(delta, closure.branch_dir))

    def route_score(self, start: Vec2, route: Sequence[Vec2]) -> float:
        points = [start] + list(route)
        length = sum(dist(a, b) for a, b in zip(points[:-1], points[1:]))
        clearance = min(self.boundary_clearance(p) for p in route[:-1]) if len(route) > 1 else 0.0
        return length - 0.2 * clearance

    def boundary_clearance(self, point: Vec2) -> float:
        return min(
            point[0] - self.map_min_x,
            self.map_max_x - point[0],
            point[1] - self.map_min_y,
            self.map_max_y - point[1],
        )

    def point_in_bounds(self, point: Vec2) -> bool:
        return (
            self.map_min_x <= point[0] <= self.map_max_x
            and self.map_min_y <= point[1] <= self.map_max_y
        )

    def should_publish_route(self, ugv_id: int, route: Sequence[Vec2]) -> bool:
        old = self.last_routes.get(ugv_id)
        if not old or len(old) != len(route):
            return True
        max_delta = max(dist(a, b) for a, b in zip(old, route))
        return max_delta > self.min_route_change

    def publish_route(self, ugv_id: int, route: Sequence[Vec2], seq: int) -> None:
        path = Path()
        path.header.seq = seq
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "world"
        z = self.goal_z.get(ugv_id, self.route_z)
        for idx, xy in enumerate(route):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = xy[0]
            pose.pose.position.y = xy[1]
            pose.pose.position.z = z if idx == len(route) - 1 else self.route_z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pubs[ugv_id].publish(path)
        rospy.logwarn(
            "[TopoDetour] ugv_%d route seq=%d waypoints=%s",
            ugv_id,
            seq,
            " -> ".join("(%.2f, %.2f)" % p for p in route),
        )

    def publish_required(self, required: Dict[int, int]) -> None:
        msg = Int32MultiArray()
        for ugv_id in sorted(required):
            seq = required[ugv_id]
            if seq > 0:
                msg.data.extend([ugv_id, seq])
        self.required_pub.publish(msg)

    def publish_markers(self) -> None:
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "topo_detour_route"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.color.r = 0.2
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        for ugv_id, route in self.last_routes.items():
            if ugv_id not in self.odom:
                continue
            points = [self.odom[ugv_id]] + route
            for a, b in zip(points[:-1], points[1:]):
                marker.points.append(point3(a, self.route_z + 0.15))
                marker.points.append(point3(b, self.route_z + 0.15))
        self.marker_pub.publish(marker)


def main() -> None:
    rospy.init_node("topo_detour_router")
    TopoDetourRouter()
    rospy.spin()


if __name__ == "__main__":
    main()
