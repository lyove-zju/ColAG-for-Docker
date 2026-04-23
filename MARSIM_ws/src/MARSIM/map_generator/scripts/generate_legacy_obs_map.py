#!/usr/bin/env python3
"""Generate legacy 40/60/80obs-style ASCII PCD maps.

The generated maps follow the same high-level structure as the bundled
40obs/60obs/80obs family:
- a full z=0 ground plane
- short square box obstacles centered on a 0.1 m grid
- obstacle side walls at z=0.1..0.5
- obstacle top surfaces at z=0.6
- ASCII PCD output under map_generator/resource/
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple


RESOLUTION = 0.1
GROUND_Z = 0.0
SIDE_Z_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)
TOP_Z = 0.6
SMALL_OBS_CELLS = 7
LARGE_OBS_CELLS = 9
LARGE_OBS_RATIO = 79.0 / 180.0
RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resource"


def quantize(value: float) -> float:
    return round(value + 0.0, 4)


def ensure_grid_aligned(value: float, name: str) -> float:
    steps = round(value / RESOLUTION)
    aligned = steps * RESOLUTION
    if not math.isclose(value, aligned, abs_tol=1e-9):
        raise ValueError(f"{name} must be a multiple of {RESOLUTION:.1f} m, got {value}.")
    return round(aligned, 4)


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

    @property
    def half_cells(self) -> int:
        return self.size_cells // 2

    def top_xy(self) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for dx in range(-self.half_cells, self.half_cells + 1):
            for dy in range(-self.half_cells, self.half_cells + 1):
                pts.append(
                    (
                        quantize(self.center_x + dx * RESOLUTION),
                        quantize(self.center_y + dy * RESOLUTION),
                    )
                )
        return pts

    def points(self) -> List[Tuple[float, float, float]]:
        pts: List[Tuple[float, float, float]] = []
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


def build_size_plan(obs_num: int, rng: random.Random) -> List[int]:
    large_count = int(round(obs_num * LARGE_OBS_RATIO))
    small_count = obs_num - large_count
    plan = [SMALL_OBS_CELLS] * small_count + [LARGE_OBS_CELLS] * large_count
    rng.shuffle(plan)
    return plan


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


def place_obstacles(
    obs_num: int,
    seed: int,
    min_distance: float,
    size_x: float,
    size_y: float,
) -> List[Obstacle]:
    rng = random.Random(seed)
    size_plan = build_size_plan(obs_num, rng)
    candidate_centers = {
        SMALL_OBS_CELLS: [
            (x, y)
            for x in build_center_candidates(size_x, SMALL_OBS_CELLS)
            for y in build_center_candidates(size_y, SMALL_OBS_CELLS)
        ],
        LARGE_OBS_CELLS: [
            (x, y)
            for x in build_center_candidates(size_x, LARGE_OBS_CELLS)
            for y in build_center_candidates(size_y, LARGE_OBS_CELLS)
        ],
    }

    placed: List[Obstacle] = []
    occupied_top_xy: Set[Tuple[float, float]] = set()
    max_attempts = max(5000, obs_num * 500)

    for idx, size_cells in enumerate(size_plan, start=1):
        candidates = candidate_centers[size_cells]
        if not candidates:
            raise RuntimeError("No valid obstacle centers are available for the current map size.")

        placed_this_round = False
        for _ in range(max_attempts):
            center_x, center_y = candidates[rng.randrange(len(candidates))]
            if not min_distance_ok(center_x, center_y, placed, min_distance):
                continue

            candidate = Obstacle(center_x=center_x, center_y=center_y, size_cells=size_cells)
            top_xy = candidate.top_xy()
            if any(cell in occupied_top_xy for cell in top_xy):
                continue

            placed.append(candidate)
            occupied_top_xy.update(top_xy)
            placed_this_round = True
            break

        if not placed_this_round:
            raise RuntimeError(
                "Could not place obstacle "
                f"{idx}/{obs_num}. Try reducing obs_num or min_distance, "
                "or increase the map size."
            )

    return placed


def iter_ground_points(size_x: float, size_y: float) -> Iterable[Tuple[float, float, float]]:
    for x in build_axis_points(size_x):
        for y in build_axis_points(size_y):
            yield (x, y, GROUND_Z)


def write_ascii_pcd(output_path: Path, points: Sequence[Tuple[float, float, float]]) -> None:
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


def build_output_name(raw_name: str, obs_num: int, seed: int) -> str:
    if raw_name:
        name = Path(raw_name).name
    else:
        name = f"generated_{obs_num}obs_seed{seed}"

    if not name.endswith(".pcd"):
        name += ".pcd"
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a legacy 40/60/80obs-style ASCII PCD map into map_generator/resource."
    )
    parser.add_argument(
        "output_name",
        nargs="?",
        default="",
        help="Output map name. '.pcd' is appended automatically when omitted.",
    )
    parser.add_argument("--obs-num", type=int, required=True, help="Number of square obstacles.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for obstacle placement.")
    parser.add_argument(
        "--min-distance",
        type=float,
        default=1.5,
        help="Minimum Euclidean distance between obstacle centers in meters. Default: 1.5",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.obs_num <= 0:
        raise ValueError("--obs-num must be positive.")
    if args.min_distance < 0.0:
        raise ValueError("--min-distance must be non-negative.")

    size_x = ensure_grid_aligned(args.size_x, "--size-x")
    size_y = ensure_grid_aligned(args.size_y, "--size-y")
    min_required_size = LARGE_OBS_CELLS * RESOLUTION
    if size_x < min_required_size or size_y < min_required_size:
        raise ValueError(
            f"map size must be at least {min_required_size:.1f} m in both directions."
        )

    obstacles = place_obstacles(
        obs_num=args.obs_num,
        seed=args.seed,
        min_distance=args.min_distance,
        size_x=size_x,
        size_y=size_y,
    )

    points: List[Tuple[float, float, float]] = []
    for obs in obstacles:
        points.extend(obs.points())
    points.extend(iter_ground_points(size_x, size_y))

    output_name = build_output_name(args.output_name, args.obs_num, args.seed)
    output_path = RESOURCE_DIR / output_name
    write_ascii_pcd(output_path, points)

    small_count = sum(1 for obs in obstacles if obs.size_cells == SMALL_OBS_CELLS)
    large_count = len(obstacles) - small_count
    print(f"Wrote {output_path}")
    print(
        "summary: "
        f"obs_num={len(obstacles)}, "
        f"small_obs={small_count}, "
        f"large_obs={large_count}, "
        f"size=({size_x:.1f}m, {size_y:.1f}m), "
        f"points={len(points)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
