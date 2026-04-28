#!/usr/bin/env python3
"""Generate legacy-style structured ASCII PCD maps.

The output intentionally keeps the same map mechanism as the bundled
40obs/60obs/80obs maps:
- full z=0 ground plane
- 0.1 m grid resolution
- short square box obstacles
- side walls at z=0.1..0.5
- top surfaces at z=0.6
- ASCII PCD files written under map_generator/resource/

Only the obstacle arrangement changes. The supported layouts are V-shaped,
U-shaped, and dead-end trap scenes for stress-testing planners. By default,
each map contains several small copies of the selected trap shape instead of
one large scene-sized structure.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


RESOLUTION = 0.1
GROUND_Z = 0.0
SIDE_Z_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)
TOP_Z = 0.6
SMALL_OBS_CELLS = 7
LARGE_OBS_CELLS = 9
LARGE_OBS_RATIO = 79.0 / 180.0
RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resource"
DEFAULT_INSTANCE_COUNT = 3
DEFAULT_SHAPE_SCALE = 0.25
DEFAULT_OPENING_SCALE = 1.9
DEFAULT_DEPTH_SCALE = 0.95
DEFAULT_BACKGROUND_SIZE_X = 16.0
DEFAULT_BACKGROUND_SIZE_Y = 10.0
DEFAULT_ENTRANCE_CLEARANCE = 1.4
DEFAULT_SHAPE_CLEARANCE = 1.8

SCENARIO_OUTPUTS: Dict[str, str] = {
    "v": "v_shape.pcd",
    "u": "u_shape.pcd",
    "deadend": "deadend.pcd",
}

PROTECTED_POINTS = (
    (0.0, 16.0),    # ugv_0 start
    (-2.0, 16.0),   # ugv_1 start
    (2.0, 16.0),    # ugv_2 start
    (0.0, -16.0),   # ugv_0 target
    (-2.0, -16.0),  # ugv_1 target
    (2.0, -16.0),   # ugv_2 target
    (0.0, 15.0),    # uav start
)

CENTRAL_STRUCTURE_POSES: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 9.5, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, -9.5, 0.0),
)


Point3 = Tuple[float, float, float]
Point2 = Tuple[float, float]
Segment = Tuple[Point2, Point2, str]


def quantize(value: float) -> float:
    return round(value + 0.0, 4)


def ensure_grid_aligned(value: float, name: str) -> float:
    steps = round(value / RESOLUTION)
    aligned = steps * RESOLUTION
    if not math.isclose(value, aligned, abs_tol=1e-9):
        raise ValueError(f"{name} must be a multiple of {RESOLUTION:.1f} m, got {value}.")
    return round(aligned, 4)


def snap_center(value: float) -> float:
    """Snap to the half-cell center lattice used by the legacy obstacle generator."""
    return quantize(round((value - RESOLUTION / 2.0) / RESOLUTION) * RESOLUTION + RESOLUTION / 2.0)


def build_axis_points(size: float) -> List[float]:
    steps = int(round(size / RESOLUTION))
    start = -size / 2.0
    return [quantize(start + idx * RESOLUTION) for idx in range(steps + 1)]


def build_center_candidates(size: float, obs_cells: int) -> List[float]:
    half_cells = obs_cells // 2
    start = -size / 2.0 + RESOLUTION / 2.0 + half_cells * RESOLUTION
    end = size / 2.0 - RESOLUTION / 2.0 - half_cells * RESOLUTION
    count = int(round((end - start) / RESOLUTION)) + 1
    if count <= 0:
        raise ValueError(
            f"map size {size} is too small for {obs_cells * RESOLUTION:.1f} m obstacles."
        )
    return [quantize(start + idx * RESOLUTION) for idx in range(count)]


@dataclass(frozen=True)
class Obstacle:
    center_x: float
    center_y: float
    size_cells: int
    label: str = "shape"

    @property
    def half_cells(self) -> int:
        return self.size_cells // 2

    def top_xy(self) -> List[Point2]:
        pts: List[Point2] = []
        for dx in range(-self.half_cells, self.half_cells + 1):
            for dy in range(-self.half_cells, self.half_cells + 1):
                pts.append(
                    (
                        quantize(self.center_x + dx * RESOLUTION),
                        quantize(self.center_y + dy * RESOLUTION),
                    )
                )
        return pts

    def points(self) -> List[Point3]:
        pts: List[Point3] = []
        for z in SIDE_Z_LEVELS:
            for dx in range(-self.half_cells, self.half_cells + 1):
                for dy in range(-self.half_cells, self.half_cells + 1):
                    if abs(dx) != self.half_cells and abs(dy) != self.half_cells:
                        continue
                    pts.append(
                        (
                            quantize(self.center_x + dx * RESOLUTION),
                            quantize(self.center_y + dy * RESOLUTION),
                            z,
                        )
                    )

        for x, y in self.top_xy():
            pts.append((x, y, TOP_Z))

        return pts


def iter_ground_points(size_x: float, size_y: float) -> Iterable[Point3]:
    for x in build_axis_points(size_x):
        for y in build_axis_points(size_y):
            yield (x, y, GROUND_Z)


def build_size_plan(obs_num: int, rng: random.Random) -> List[int]:
    large_count = int(round(obs_num * LARGE_OBS_RATIO))
    small_count = obs_num - large_count
    plan = [SMALL_OBS_CELLS] * small_count + [LARGE_OBS_CELLS] * large_count
    rng.shuffle(plan)
    return plan


def line_obstacles(
    start: Point2,
    end: Point2,
    size_cells: int,
    spacing: float,
    label: str,
) -> List[Obstacle]:
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    steps = max(1, int(math.ceil(distance / spacing)))
    obstacles: List[Obstacle] = []
    seen: Set[Point2] = set()

    for idx in range(steps + 1):
        ratio = idx / steps
        x = snap_center(start[0] + (end[0] - start[0]) * ratio)
        y = snap_center(start[1] + (end[1] - start[1]) * ratio)
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        obstacles.append(Obstacle(x, y, size_cells, label=label))

    return obstacles


def local_shape_segments(scenario: str) -> List[Segment]:
    if scenario == "u":
        return [
            ((-4.2, 6.5), (-4.2, -7.5), "u_left"),
            ((4.2, 6.5), (4.2, -7.5), "u_right"),
            ((-4.2, -7.5), (4.2, -7.5), "u_back"),
        ]

    if scenario == "v":
        return [
            ((-5.4, 7.5), (0.0, -7.5), "v_left"),
            ((5.4, 7.5), (0.0, -7.5), "v_right"),
        ]

    if scenario == "deadend":
        return [
            ((-2.8, 7.0), (-2.8, -7.5), "deadend_left"),
            ((2.8, 7.0), (2.8, -7.5), "deadend_right"),
            ((-2.8, -7.5), (2.8, -7.5), "deadend_back"),
        ]

    raise ValueError(f"Unknown scenario: {scenario}")


def local_entrance_points(scenario: str) -> List[Point2]:
    if scenario == "u":
        xs = (-4.2, -2.1, 0.0, 2.1, 4.2)
        ys = (6.5, 8.5)
        return [(x, y) for y in ys for x in xs]

    if scenario == "v":
        xs = (-5.4, -2.7, 0.0, 2.7, 5.4)
        ys = (7.5, 9.5)
        return [(x, y) for y in ys for x in xs]

    if scenario == "deadend":
        xs = (-2.8, -1.4, 0.0, 1.4, 2.8)
        ys = (7.0, 9.0)
        return [(x, y) for y in ys for x in xs]

    raise ValueError(f"Unknown scenario: {scenario}")


def transform_local_point(
    point: Point2,
    center_x: float,
    center_y: float,
    scale: float,
    opening_scale: float,
    depth_scale: float,
    yaw: float,
) -> Point2:
    scaled_x = point[0] * scale * opening_scale
    scaled_y = point[1] * scale * depth_scale
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        center_x + scaled_x * cos_yaw - scaled_y * sin_yaw,
        center_y + scaled_x * sin_yaw + scaled_y * cos_yaw,
    )


def build_entrance_clearance_points(
    scenario: str,
    instance_count: int,
    shape_scale: float,
    opening_scale: float,
    depth_scale: float,
) -> List[Point2]:
    points: List[Point2] = []
    for center_x, center_y, yaw in CENTRAL_STRUCTURE_POSES[:instance_count]:
        for point in local_entrance_points(scenario):
            points.append(
                transform_local_point(
                    point,
                    center_x,
                    center_y,
                    shape_scale,
                    opening_scale,
                    depth_scale,
                    yaw,
                )
            )
    return points


def build_shape_obstacles(
    scenario: str,
    size_cells: int,
    instance_count: int,
    shape_scale: float,
    opening_scale: float,
    depth_scale: float,
) -> List[Obstacle]:
    spacing = size_cells * RESOLUTION * 0.85
    shape_obstacles: List[Obstacle] = []
    segments = local_shape_segments(scenario)

    for instance_idx, (center_x, center_y, yaw) in enumerate(
        CENTRAL_STRUCTURE_POSES[:instance_count],
        start=1,
    ):
        for start, end, label in segments:
            start_xy = transform_local_point(
                start,
                center_x,
                center_y,
                shape_scale,
                opening_scale,
                depth_scale,
                yaw,
            )
            end_xy = transform_local_point(
                end,
                center_x,
                center_y,
                shape_scale,
                opening_scale,
                depth_scale,
                yaw,
            )
            shape_obstacles.extend(
                line_obstacles(
                    start_xy,
                    end_xy,
                    size_cells,
                    spacing,
                    f"{label}_{instance_idx}",
                )
            )

    return shape_obstacles


def within_map(obstacle: Obstacle, size_x: float, size_y: float) -> bool:
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    return all(
        -half_x <= x <= half_x and -half_y <= y <= half_y
        for x, y in obstacle.top_xy()
    )


def min_distance_ok(
    candidate_x: float,
    candidate_y: float,
    placed: Sequence[Obstacle],
    min_distance: float,
) -> bool:
    min_sq = min_distance * min_distance
    for obs in placed:
        dx = candidate_x - obs.center_x
        dy = candidate_y - obs.center_y
        if dx * dx + dy * dy < min_sq - 1e-9:
            return False
    return True


def protected_clearance_ok(obstacle: Obstacle, protected_radius: float) -> bool:
    radius_sq = protected_radius * protected_radius
    for x, y in obstacle.top_xy():
        for px, py in PROTECTED_POINTS:
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy < radius_sq - 1e-9:
                return False
    return True


def entrance_clearance_ok(
    obstacle: Obstacle,
    entrance_points: Sequence[Point2],
    entrance_clearance: float,
) -> bool:
    if entrance_clearance <= 0.0:
        return True

    radius_sq = entrance_clearance * entrance_clearance
    for x, y in obstacle.top_xy():
        for px, py in entrance_points:
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy < radius_sq - 1e-9:
                return False
    return True


def shape_clearance_ok(
    obstacle: Obstacle,
    shape_obstacles: Sequence[Obstacle],
    shape_clearance: float,
) -> bool:
    if shape_clearance <= 0.0:
        return True

    clearance_sq = shape_clearance * shape_clearance
    obstacle_half = obstacle.half_cells * RESOLUTION
    for shape_obstacle in shape_obstacles:
        shape_half = shape_obstacle.half_cells * RESOLUTION
        gap_x = max(
            abs(obstacle.center_x - shape_obstacle.center_x) - obstacle_half - shape_half,
            0.0,
        )
        gap_y = max(
            abs(obstacle.center_y - shape_obstacle.center_y) - obstacle_half - shape_half,
            0.0,
        )
        if gap_x * gap_x + gap_y * gap_y < clearance_sq - 1e-9:
            return False

    return True


def add_random_background(
    placed: List[Obstacle],
    shape_obstacles: Sequence[Obstacle],
    occupied_top_xy: Set[Point2],
    entrance_points: Sequence[Point2],
    entrance_clearance: float,
    shape_clearance: float,
    background_obs: int,
    seed: int,
    min_distance: float,
    protected_radius: float,
    size_x: float,
    size_y: float,
    background_size_x: float,
    background_size_y: float,
) -> List[Obstacle]:
    if background_obs <= 0:
        return []

    rng = random.Random(seed)
    size_plan = build_size_plan(background_obs, rng)

    def central_candidates(size: float, obs_cells: int, central_size: float) -> List[float]:
        candidates = build_center_candidates(size, obs_cells)
        if central_size <= 0.0 or central_size >= size:
            return candidates

        half_limit = central_size / 2.0
        filtered = [value for value in candidates if abs(value) <= half_limit]
        if not filtered:
            raise ValueError(
                f"central background size {central_size:.1f} m is too small "
                f"for {obs_cells * RESOLUTION:.1f} m obstacles."
            )
        return filtered

    small_x = central_candidates(size_x, SMALL_OBS_CELLS, background_size_x)
    small_y = central_candidates(size_y, SMALL_OBS_CELLS, background_size_y)
    large_x = central_candidates(size_x, LARGE_OBS_CELLS, background_size_x)
    large_y = central_candidates(size_y, LARGE_OBS_CELLS, background_size_y)

    candidate_centers = {
        SMALL_OBS_CELLS: [(x, y) for x in small_x for y in small_y],
        LARGE_OBS_CELLS: [(x, y) for x in large_x for y in large_y],
    }

    background: List[Obstacle] = []
    max_attempts = max(5000, background_obs * 1000)

    for idx, size_cells in enumerate(size_plan, start=1):
        candidates = candidate_centers[size_cells]
        placed_this_round = False

        for _ in range(max_attempts):
            center_x, center_y = candidates[rng.randrange(len(candidates))]
            candidate = Obstacle(center_x, center_y, size_cells, label="background")
            top_xy = candidate.top_xy()
            if any(cell in occupied_top_xy for cell in top_xy):
                continue
            if not shape_clearance_ok(candidate, shape_obstacles, shape_clearance):
                continue
            if not min_distance_ok(center_x, center_y, placed, min_distance):
                continue
            if not protected_clearance_ok(candidate, protected_radius):
                continue
            if not entrance_clearance_ok(candidate, entrance_points, entrance_clearance):
                continue

            placed.append(candidate)
            background.append(candidate)
            occupied_top_xy.update(top_xy)
            placed_this_round = True
            break

        if not placed_this_round:
            raise RuntimeError(
                "Could not place background obstacle "
                f"{idx}/{background_obs}. Try reducing --background-obs, "
                "--shape-clearance, or --min-distance."
            )

    return background


def collect_obstacle_points(obstacles: Sequence[Obstacle]) -> List[Point3]:
    points: List[Point3] = []
    seen: Set[Point3] = set()
    for obstacle in obstacles:
        for point in obstacle.points():
            if point in seen:
                continue
            seen.add(point)
            points.append(point)
    return points


def write_ascii_pcd(output_path: Path, points: Sequence[Point3]) -> None:
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        "DATA ascii",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as fh:
        fh.write("\n".join(header))
        fh.write("\n")
        for x, y, z in points:
            fh.write(f"{x:.7f} {y:.7f} {z:.7f}\n")


def build_output_path(scenario: str, raw_output: str, output_prefix: str) -> Path:
    if raw_output:
        name = Path(raw_output).name
        if not name.endswith(".pcd"):
            name += ".pcd"
        return RESOURCE_DIR / name

    if output_prefix:
        return RESOURCE_DIR / f"{output_prefix}_{scenario}.pcd"

    return RESOURCE_DIR / SCENARIO_OUTPUTS[scenario]


def build_map(
    scenario: str,
    size_cells: int,
    instance_count: int,
    shape_scale: float,
    opening_scale: float,
    depth_scale: float,
    background_obs: int,
    seed: int,
    min_distance: float,
    protected_radius: float,
    entrance_clearance: float,
    shape_clearance: float,
    size_x: float,
    size_y: float,
    background_size_x: float,
    background_size_y: float,
) -> Tuple[List[Point3], List[Obstacle], List[Obstacle]]:
    shape_obstacles = build_shape_obstacles(
        scenario=scenario,
        size_cells=size_cells,
        instance_count=instance_count,
        shape_scale=shape_scale,
        opening_scale=opening_scale,
        depth_scale=depth_scale,
    )
    entrance_points = build_entrance_clearance_points(
        scenario=scenario,
        instance_count=instance_count,
        shape_scale=shape_scale,
        opening_scale=opening_scale,
        depth_scale=depth_scale,
    )
    for obstacle in shape_obstacles:
        if not within_map(obstacle, size_x, size_y):
            raise ValueError(
                f"{scenario} obstacle at ({obstacle.center_x}, {obstacle.center_y}) "
                "does not fit in the requested map size."
            )
        if not protected_clearance_ok(obstacle, protected_radius):
            raise ValueError(
                f"{scenario} obstacle at ({obstacle.center_x}, {obstacle.center_y}) "
                "is too close to a protected start/target point. Reduce --protected-radius."
            )

    occupied_top_xy: Set[Point2] = set()
    for obstacle in shape_obstacles:
        occupied_top_xy.update(obstacle.top_xy())

    all_obstacles = list(shape_obstacles)
    background = add_random_background(
        placed=all_obstacles,
        shape_obstacles=shape_obstacles,
        occupied_top_xy=occupied_top_xy,
        entrance_points=entrance_points,
        entrance_clearance=entrance_clearance,
        shape_clearance=shape_clearance,
        background_obs=background_obs,
        seed=seed,
        min_distance=min_distance,
        protected_radius=protected_radius,
        size_x=size_x,
        size_y=size_y,
        background_size_x=background_size_x,
        background_size_y=background_size_y,
    )

    points = collect_obstacle_points(all_obstacles)
    points.extend(iter_ground_points(size_x, size_y))
    return points, shape_obstacles, background


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate V/U/dead-end legacy-style ASCII PCD maps into "
            "map_generator/resource without changing the launch flow."
        )
    )
    parser.add_argument(
        "output_name",
        nargs="?",
        default="",
        help=(
            "Output file name for a single scenario. '.pcd' is appended when omitted. "
            "Ignored when --scenario all is used."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("v", "u", "deadend", "all"),
        default="all",
        help="Structured layout to generate. Default: all",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix used with --scenario all, e.g. prefix_v.pcd. Default names are v_shape/u_shape/deadend.",
    )
    parser.add_argument(
        "--block-size",
        choices=("small", "large"),
        default="large",
        help="Use the same 0.7 m or 0.9 m square box primitives as legacy maps. Default: large",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=DEFAULT_INSTANCE_COUNT,
        help=(
            "Number of small trap structures to place in each generated map. "
            f"Default: {DEFAULT_INSTANCE_COUNT}"
        ),
    )
    parser.add_argument(
        "--shape-scale",
        type=float,
        default=DEFAULT_SHAPE_SCALE,
        help=(
            "Base scale applied to the original large V/U/dead-end geometry before placement. "
            f"Default: {DEFAULT_SHAPE_SCALE}"
        ),
    )
    parser.add_argument(
        "--opening-scale",
        type=float,
        default=DEFAULT_OPENING_SCALE,
        help=(
            "Horizontal multiplier for the trap opening. Larger values make wider mouths. "
            f"Default: {DEFAULT_OPENING_SCALE}"
        ),
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEFAULT_DEPTH_SCALE,
        help=(
            "Vertical multiplier for trap depth. Smaller values make shallower traps. "
            f"Default: {DEFAULT_DEPTH_SCALE}"
        ),
    )
    parser.add_argument(
        "--background-obs",
        type=int,
        default=18,
        help="Number of extra random legacy-style box obstacles. Use 0 for shape-only maps. Default: 18",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed for background obstacles.")
    parser.add_argument(
        "--min-distance",
        type=float,
        default=1.4,
        help="Minimum distance from random background obstacle centers to existing obstacles. Default: 1.4",
    )
    parser.add_argument(
        "--protected-radius",
        type=float,
        default=2.0,
        help="Keep generated obstacles away from default UGV/UAV starts and UGV targets. Default: 2.0",
    )
    parser.add_argument(
        "--entrance-clearance",
        type=float,
        default=DEFAULT_ENTRANCE_CLEARANCE,
        help=(
            "Keep random background obstacles away from each trap entrance. "
            f"Default: {DEFAULT_ENTRANCE_CLEARANCE}"
        ),
    )
    parser.add_argument(
        "--shape-clearance",
        type=float,
        default=DEFAULT_SHAPE_CLEARANCE,
        help=(
            "Minimum XY clearance from random background obstacle footprints "
            "to structured trap obstacle footprints. Default: "
            f"{DEFAULT_SHAPE_CLEARANCE}"
        ),
    )
    parser.add_argument(
        "--size-x",
        type=float,
        default=35.0,
        help="Map width in meters. Must be a multiple of 0.1. Default: 35.0",
    )
    parser.add_argument(
        "--size-y",
        type=float,
        default=35.0,
        help="Map height in meters. Must be a multiple of 0.1. Default: 35.0",
    )
    parser.add_argument(
        "--background-size-x",
        type=float,
        default=DEFAULT_BACKGROUND_SIZE_X,
        help=(
            "Width of the central region used for random background obstacles. "
            f"Use 0 for the full map. Default: {DEFAULT_BACKGROUND_SIZE_X}"
        ),
    )
    parser.add_argument(
        "--background-size-y",
        type=float,
        default=DEFAULT_BACKGROUND_SIZE_Y,
        help=(
            "Height of the central region used for random background obstacles. "
            f"Use 0 for the full map. Default: {DEFAULT_BACKGROUND_SIZE_Y}"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.background_obs < 0:
        raise ValueError("--background-obs must be non-negative.")
    if args.instances <= 0:
        raise ValueError("--instances must be positive.")
    if args.instances > len(CENTRAL_STRUCTURE_POSES):
        raise ValueError(f"--instances supports at most {len(CENTRAL_STRUCTURE_POSES)} structures.")
    if args.shape_scale <= 0.0:
        raise ValueError("--shape-scale must be positive.")
    if args.opening_scale <= 0.0:
        raise ValueError("--opening-scale must be positive.")
    if args.depth_scale <= 0.0:
        raise ValueError("--depth-scale must be positive.")
    if args.min_distance < 0.0:
        raise ValueError("--min-distance must be non-negative.")
    if args.protected_radius < 0.0:
        raise ValueError("--protected-radius must be non-negative.")
    if args.entrance_clearance < 0.0:
        raise ValueError("--entrance-clearance must be non-negative.")
    if args.shape_clearance < 0.0:
        raise ValueError("--shape-clearance must be non-negative.")
    if args.background_size_x < 0.0:
        raise ValueError("--background-size-x must be non-negative.")
    if args.background_size_y < 0.0:
        raise ValueError("--background-size-y must be non-negative.")

    size_x = ensure_grid_aligned(args.size_x, "--size-x")
    size_y = ensure_grid_aligned(args.size_y, "--size-y")
    min_required_size = LARGE_OBS_CELLS * RESOLUTION
    if size_x < min_required_size or size_y < min_required_size:
        raise ValueError(
            f"map size must be at least {min_required_size:.1f} m in both directions."
        )

    size_cells = LARGE_OBS_CELLS if args.block_size == "large" else SMALL_OBS_CELLS
    scenarios = ("v", "u", "deadend") if args.scenario == "all" else (args.scenario,)

    for index, scenario in enumerate(scenarios):
        points, shape_obstacles, background = build_map(
            scenario=scenario,
            size_cells=size_cells,
            instance_count=args.instances,
            shape_scale=args.shape_scale,
            opening_scale=args.opening_scale,
            depth_scale=args.depth_scale,
            background_obs=args.background_obs,
            seed=args.seed + index * 101,
            min_distance=args.min_distance,
            protected_radius=args.protected_radius,
            entrance_clearance=args.entrance_clearance,
            shape_clearance=args.shape_clearance,
            size_x=size_x,
            size_y=size_y,
            background_size_x=args.background_size_x,
            background_size_y=args.background_size_y,
        )
        output_path = build_output_path(
            scenario=scenario,
            raw_output="" if args.scenario == "all" else args.output_name,
            output_prefix=args.output_prefix,
        )
        write_ascii_pcd(output_path, points)
        print(f"Wrote {output_path}")
        print(
            "summary: "
            f"scenario={scenario}, "
            f"instances={args.instances}, "
            f"shape_scale={args.shape_scale:.3f}, "
            f"opening_scale={args.opening_scale:.3f}, "
            f"depth_scale={args.depth_scale:.3f}, "
            f"structural_blocks={len(shape_obstacles)}, "
            f"background_blocks={len(background)}, "
            f"entrance_clearance={args.entrance_clearance:.2f}, "
            f"shape_clearance={args.shape_clearance:.2f}, "
            f"background_region=({args.background_size_x:.1f}m, {args.background_size_y:.1f}m), "
            f"size=({size_x:.1f}m, {size_y:.1f}m), "
            f"points={len(points)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
