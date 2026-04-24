#!/usr/bin/env python3
"""VRPTW expert utilities mirroring the current UAV dispatch baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except ImportError:  # pragma: no cover
    pywrapcp = None
    routing_enums_pb2 = None

from dispatch_dataset import save_jsonl
from dispatch_env import SyntheticDispatchGenerator
from dispatch_runtime import (
    DispatchObservationBuilder,
    DispatchTask,
    sanitize_slot_order,
    slot_order_to_ugv_ids,
)


def require_ortools() -> None:
    if pywrapcp is None or routing_enums_pb2 is None:
        raise RuntimeError("OR-Tools is required for VRPTW expert generation. Install it before running this script.")


@dataclass
class ExpertSolution:
    slot_order: List[int]
    ugv_order: List[int]
    time_matrix: List[List[int]]
    time_windows: List[List[int]]
    solved: bool
    relax_iterations: int


class VRPTWExpertPlanner:
    def __init__(
        self,
        ugv_num: int,
        uav_num: int = 1,
        v_max: float = 3.0,
        a_max: float = 1.0,
        time_resolution: int = 100,
        max_relax_iterations: int = 100,
    ):
        require_ortools()
        self.ugv_num = ugv_num
        self.uav_num = uav_num
        self.max_relax_iterations = max_relax_iterations
        self.builder = DispatchObservationBuilder(
            ugv_num=ugv_num,
            v_max=v_max,
            a_max=a_max,
            time_resolution=time_resolution,
        )

    def solve(
        self,
        tasks: Sequence[DispatchTask],
        pos: Sequence[float],
        vel: Sequence[float],
        now_sec: float,
    ) -> ExpertSolution:
        if self.uav_num != 1:
            raise NotImplementedError("RL dispatch v1 only supports single UAV expert routes.")

        observation = self.builder.build_observation(tasks, pos, vel, now_sec)
        time_matrix = [row[:] for row in observation["time_matrix"]]
        time_windows = [window[:] for window in observation["time_windows"]]
        active_ids = observation["active_ugv_ids"]

        if not active_ids:
            return ExpertSolution([], [], time_matrix, time_windows, True, 0)

        route_order: List[int] = []
        solved = False
        relax_iterations = 0

        while relax_iterations < self.max_relax_iterations:
            manager = pywrapcp.RoutingIndexManager(len(time_matrix), self.uav_num, 0)
            routing = pywrapcp.RoutingModel(manager)

            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return time_matrix[from_node][to_node]

            transit_callback_index = routing.RegisterTransitCallback(time_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            routing.AddDimension(
                transit_callback_index,
                30000,
                500000,
                False,
                "Time",
            )
            time_dimension = routing.GetDimensionOrDie("Time")

            for location_idx in range(len(time_windows)):
                if location_idx == 0:
                    continue
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(
                    int(time_windows[location_idx][0]),
                    int(time_windows[location_idx][1]),
                )
            start_index = routing.Start(0)
            time_dimension.CumulVar(start_index).SetRange(int(time_windows[0][0]), int(time_windows[0][1]))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(start_index))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(0)))

            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            solution = routing.SolveWithParameters(search_parameters)
            if solution:
                index = routing.Start(0)
                route_order = []
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index > 0:
                        route_order.append(node_index - 1)
                    index = solution.Value(routing.NextVar(index))
                solved = True
                break

            relax_iterations += 1
            for idx in range(len(time_windows) - 1):
                time_windows[idx + 1][1] = int(time_windows[idx + 1][1] * 1.5 + 0.999)

        route_order = sanitize_slot_order(route_order, observation["active_mask"])
        return ExpertSolution(
            slot_order=route_order,
            ugv_order=slot_order_to_ugv_ids(route_order, observation["slot_task_ids"]),
            time_matrix=time_matrix,
            time_windows=time_windows,
            solved=solved,
            relax_iterations=relax_iterations,
        )


def generate_dataset(args: argparse.Namespace) -> int:
    generator = SyntheticDispatchGenerator(
        ugv_num=args.ugv_num,
        seed=args.seed,
        map_size_x=args.map_size_x,
        map_size_y=args.map_size_y,
    )
    expert = VRPTWExpertPlanner(
        ugv_num=args.ugv_num,
        uav_num=1,
        v_max=args.v_max,
        a_max=args.a_max,
        time_resolution=args.time_resolution,
    )

    records: List[Dict[str, object]] = []
    skipped = 0
    for sample_idx in range(args.num_samples):
        scenario = generator.sample()
        solution = expert.solve(scenario.tasks, scenario.self_pos, scenario.self_vel, now_sec=0.0)
        if not solution.solved:
            skipped += 1
            continue
        record = {
            "global_features": scenario.observation["global_features"],
            "node_features": scenario.observation["node_features"],
            "pairwise_time_matrix": scenario.observation["pairwise_time_matrix"],
            "active_mask": scenario.observation["active_mask"],
            "expert_order_slots": solution.slot_order,
            "meta": {
                "sample_idx": sample_idx,
                "ugv_order": solution.ugv_order,
                "relax_iterations": solution.relax_iterations,
                "active_ids": scenario.observation["active_ugv_ids"],
            },
        }
        records.append(record)

    save_jsonl(records, args.output)
    print(f"wrote {len(records)} expert samples to {args.output}, skipped_unsolved={skipped}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="VRPTW expert data exporter for RL dispatch.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--ugv-num", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-size-x", type=float, default=35.0)
    parser.add_argument("--map-size-y", type=float, default=35.0)
    parser.add_argument("--v-max", type=float, default=3.0)
    parser.add_argument("--a-max", type=float, default=1.0)
    parser.add_argument("--time-resolution", type=int, default=100)
    args = parser.parse_args()
    return generate_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
