#!/usr/bin/env python3
"""Create dependency-free SVG plots for RL dispatch training metrics."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple


COLOR_PALETTE = [
    "#146C94",
    "#C85C5C",
    "#4F8A5B",
    "#D28B26",
    "#594A9E",
    "#3F7D7B",
]


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    if not path or not os.path.exists(path):
        return []
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_json(path: str) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _read_bc_log(path: str) -> List[Dict[str, object]]:
    if not path or not os.path.exists(path):
        return []
    records: List[Dict[str, object]] = []
    pattern = re.compile(r"\[BC\]\s+epoch=(\d+)\s+avg_nll=([-+0-9.eE]+)")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = pattern.search(line)
            if match:
                records.append(
                    {
                        "phase": "bc",
                        "epoch": int(match.group(1)),
                        "avg_nll": float(match.group(2)),
                    }
                )
    return records


def _read_ppo_log(path: str) -> List[Dict[str, object]]:
    if not path or not os.path.exists(path):
        return []
    records: List[Dict[str, object]] = []
    ppo_pattern = re.compile(r"\[PPO\]\s+update=(\d+)\s+avg_reward=([-+0-9.eE]+)")
    event_pattern = re.compile(
        r"\[PPO_EVENT\]\s+update=(\d+)\s+avg_reward=([-+0-9.eE]+)\s+"
        r"avg_hits=([-+0-9.eE]+)\s+avg_misses=([-+0-9.eE]+)\s+transitions=(\d+)"
    )
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            event_match = event_pattern.search(line)
            if event_match:
                records.append(
                    {
                        "phase": "ppo_event",
                        "update": int(event_match.group(1)),
                        "avg_reward": float(event_match.group(2)),
                        "avg_hits": float(event_match.group(3)),
                        "avg_misses": float(event_match.group(4)),
                        "transitions": int(event_match.group(5)),
                    }
                )
                continue
            ppo_match = ppo_pattern.search(line)
            if ppo_match:
                records.append(
                    {
                        "phase": "ppo",
                        "update": int(ppo_match.group(1)),
                        "avg_reward": float(ppo_match.group(2)),
                    }
                )
    return records


def _series(records: Sequence[Dict[str, object]], x_key: str, y_key: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for idx, record in enumerate(records):
        x = _finite_float(record.get(x_key, idx))
        y = _finite_float(record.get(y_key))
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def _format_number(value: float) -> str:
    if abs(value) >= 1000.0 or (0.0 < abs(value) < 0.001):
        return f"{value:.2e}"
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _range(values: Iterable[float]) -> Tuple[float, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-9:
        pad = max(1.0, abs(high) * 0.1)
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _polyline(points: Sequence[Tuple[float, float]], x_range, y_range, box) -> str:
    left, top, width, height = box
    x_min, x_max = x_range
    y_min, y_max = y_range
    coords = []
    for x, y in points:
        px = left + (x - x_min) / max(x_max - x_min, 1e-9) * width
        py = top + height - (y - y_min) / max(y_max - y_min, 1e-9) * height
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def _draw_panel(
    title: str,
    named_series: Sequence[Tuple[str, List[Tuple[float, float]]]],
    top: int,
    width: int,
    panel_height: int = 210,
) -> List[str]:
    left = 68
    right = 24
    chart_top = top + 38
    chart_height = panel_height - 72
    chart_width = width - left - right
    box = (left, chart_top, chart_width, chart_height)
    all_points = [point for _, points in named_series for point in points]
    x_range = _range(point[0] for point in all_points)
    y_range = _range(point[1] for point in all_points)

    lines = [
        f'<text x="24" y="{top + 22}" class="title">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{chart_top}" width="{chart_width}" height="{chart_height}" class="plot-bg"/>',
    ]
    for idx in range(5):
        y = chart_top + chart_height * idx / 4.0
        value = y_range[1] - (y_range[1] - y_range[0]) * idx / 4.0
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{_format_number(value)}</text>')
    for idx in range(3):
        x = left + chart_width * idx / 2.0
        value = x_range[0] + (x_range[1] - x_range[0]) * idx / 2.0
        lines.append(f'<text x="{x:.1f}" y="{chart_top + chart_height + 22}" text-anchor="middle" class="tick">{_format_number(value)}</text>')

    lines.append(f'<line x1="{left}" y1="{chart_top}" x2="{left}" y2="{chart_top + chart_height}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{chart_top + chart_height}" x2="{left + chart_width}" y2="{chart_top + chart_height}" class="axis"/>')

    legend_x = left + 10
    for idx, (name, points) in enumerate(named_series):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        if len(points) == 1:
            point = points[0]
            coords = _polyline([point], x_range, y_range, box).split(",")
            lines.append(f'<circle cx="{coords[0]}" cy="{coords[1]}" r="3" fill="{color}"/>')
        elif len(points) > 1:
            lines.append(
                f'<polyline points="{_polyline(points, x_range, y_range, box)}" '
                f'fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        legend_y = top + 22 + idx * 18
        lines.append(f'<line x1="{legend_x + 610}" y1="{legend_y - 4}" x2="{legend_x + 632}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 640}" y="{legend_y}" class="legend">{html.escape(name)}</text>')
    return lines


def _eval_summary_lines(metrics: Dict[str, object]) -> List[str]:
    if not metrics:
        return []
    keys = [
        "eval_mode",
        "rl_vs_expert_ratio",
        "rl_deadline_hit_rate",
        "expert_deadline_hit_rate",
        "rl_success_per_episode",
        "expert_success_per_episode",
        "rl_avg_response_time_seconds",
        "expert_avg_response_time_seconds",
        "rl_flight_distance_per_episode",
        "expert_flight_distance_per_episode",
    ]
    lines = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, (int, float)):
            value = _format_number(float(value))
        lines.append(f"{key}: {value}")
    return lines


def build_svg(
    bc_records: Sequence[Dict[str, object]],
    ppo_records: Sequence[Dict[str, object]],
    eval_metrics: Dict[str, object],
) -> str:
    panels: List[Tuple[str, Sequence[Tuple[str, List[Tuple[float, float]]]]]] = []
    if bc_records:
        panels.append(("BC pretraining loss", [("avg_nll", _series(bc_records, "epoch", "avg_nll"))]))
    if ppo_records:
        x_key = "update"
        panels.append(("PPO reward", [("avg_reward", _series(ppo_records, x_key, "avg_reward"))]))
        loss_series = [
            ("loss", _series(ppo_records, x_key, "loss")),
            ("policy_loss", _series(ppo_records, x_key, "policy_loss")),
            ("value_loss", _series(ppo_records, x_key, "value_loss")),
            ("entropy", _series(ppo_records, x_key, "entropy")),
        ]
        panels.append(("PPO optimization terms", [(name, points) for name, points in loss_series if points]))
        event_count_series = [
            ("avg_hits", _series(ppo_records, x_key, "avg_hits")),
            ("avg_misses", _series(ppo_records, x_key, "avg_misses")),
            ("avg_success_count", _series(ppo_records, x_key, "avg_success_count")),
            ("avg_missed_count", _series(ppo_records, x_key, "avg_missed_count")),
        ]
        event_count_series = [(name, points) for name, points in event_count_series if points]
        if event_count_series:
            panels.append(("Dispatch success and miss counts", event_count_series))

    summary = _eval_summary_lines(eval_metrics)
    width = 980
    panel_height = 210
    summary_height = 58 + max(len(summary), 1) * 18 if summary else 0
    height = 54 + len(panels) * panel_height + summary_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:DejaVu Sans,Arial,sans-serif;fill:#1f2933}",
        ".heading{font-size:22px;font-weight:700}",
        ".title{font-size:15px;font-weight:700}",
        ".tick{font-size:11px;fill:#64748b}",
        ".legend{font-size:12px;fill:#334155}",
        ".plot-bg{fill:#fbfaf7;stroke:#cbd5e1;stroke-width:1}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        ".axis{stroke:#94a3b8;stroke-width:1.2}",
        ".summary-bg{fill:#f8fafc;stroke:#cbd5e1;stroke-width:1}",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#fffdf8"/>',
        '<text x="24" y="34" class="heading">UAV RL Dispatch Training Curves</text>',
    ]
    top = 54
    for title, named_series in panels:
        if named_series:
            lines.extend(_draw_panel(title, named_series, top, width, panel_height))
            top += panel_height

    if summary:
        lines.append(f'<rect x="24" y="{top + 10}" width="{width - 48}" height="{summary_height - 20}" rx="8" class="summary-bg"/>')
        lines.append(f'<text x="44" y="{top + 38}" class="title">Final evaluation summary</text>')
        for idx, text in enumerate(summary):
            x = 44 if idx < (len(summary) + 1) // 2 else 510
            y = top + 64 + (idx % ((len(summary) + 1) // 2)) * 18
            lines.append(f'<text x="{x}" y="{y}" class="legend">{html.escape(text)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot UAV RL dispatch training curves as SVG.")
    parser.add_argument("--bc-jsonl", default="", help="BC metrics JSONL produced by train_dispatch_rl.py.")
    parser.add_argument("--ppo-jsonl", default="", help="PPO metrics JSONL produced by train_dispatch_rl.py.")
    parser.add_argument("--bc-log", default="", help="Fallback BC log path for older runs without metrics JSONL.")
    parser.add_argument("--ppo-log", default="", help="Fallback PPO log path for older runs without metrics JSONL.")
    parser.add_argument("--eval-json", default="", help="Final evaluation metrics JSON.")
    parser.add_argument("--output", required=True, help="Output SVG path.")
    args = parser.parse_args()

    bc_records = _read_jsonl(args.bc_jsonl) or _read_bc_log(args.bc_log)
    ppo_records = _read_jsonl(args.ppo_jsonl) or _read_ppo_log(args.ppo_log)
    svg = build_svg(
        bc_records=bc_records,
        ppo_records=ppo_records,
        eval_metrics=_read_json(args.eval_json),
    )
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(svg)
        fh.write("\n")
    print(f"wrote training plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
