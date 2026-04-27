#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)

MAP_NAME=""
UGV_NUM=3
RUNS=3
RL_MODEL_PATH=""
RL_DEVICE=cpu
OUTPUT_ROOT=/work/online_eval
RUN_DURATION=180
STARTUP_TIMEOUT=180
TRIGGER_DELAY=5
BETWEEN_RUN_DELAY=5
ARRIVAL_RADIUS=0.8
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash Air_ws/src/swarm_support/scripts/run_online_eval.sh --map MAP --rl-model MODEL [options]

Options:
  --map NAME                 Map name passed to run.sh, for example 96obs.
  --rl-model PATH            RL checkpoint path used by run.sh in RL mode.
  --ugv-num N                Number of UGVs. Default: 3.
  --runs N                   Number of paired VRPTW/RL runs. Default: 3.
  --rl-device cpu|cuda       RL inference device for the simulation container. Default: cpu.
  --output-root DIR          Root output directory. Default: /work/online_eval.
  --run-duration SEC         Seconds to record after sending the start trigger. Default: 180.
  --startup-timeout SEC      Max seconds to wait for UAV odom before triggering. Default: 180.
  --trigger-delay SEC        Extra seconds after odom appears before trigger. Default: 5.
  --between-run-delay SEC    Pause between runs. Default: 5.
  --arrival-radius M         Arrival radius passed to analyzer. Default: 0.8.
  --overwrite                Replace existing bag/json/log files for the requested runs.
  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --map)
      MAP_NAME="$2"
      shift 2
      ;;
    --rl-model)
      RL_MODEL_PATH="$2"
      shift 2
      ;;
    --ugv-num)
      UGV_NUM="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --rl-device)
      RL_DEVICE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-duration)
      RUN_DURATION="$2"
      shift 2
      ;;
    --startup-timeout)
      STARTUP_TIMEOUT="$2"
      shift 2
      ;;
    --trigger-delay)
      TRIGGER_DELAY="$2"
      shift 2
      ;;
    --between-run-delay)
      BETWEEN_RUN_DELAY="$2"
      shift 2
      ;;
    --arrival-radius)
      ARRIVAL_RADIUS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MAP_NAME" ]]; then
  echo "--map is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "$RL_MODEL_PATH" ]]; then
  echo "--rl-model is required." >&2
  usage >&2
  exit 1
fi

if [[ "$RL_DEVICE" != "cpu" && "$RL_DEVICE" != "cuda" ]]; then
  echo "--rl-device must be cpu or cuda." >&2
  exit 1
fi

if [[ ! -f "$RL_MODEL_PATH" ]]; then
  echo "RL model not found: $RL_MODEL_PATH" >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash
source "$REPO_ROOT/MARSIM_ws/devel/setup.bash"
source "$REPO_ROOT/Ground_ws/devel/setup.bash"
source "$REPO_ROOT/Air_ws/devel/setup.bash"

RUN_DIR="$OUTPUT_ROOT/$MAP_NAME"
mkdir -p "$RUN_DIR"

ROSCORE_PID=""
SIM_PGID=""
REC_PID=""

cleanup_processes() {
  set +e
  if [[ -n "${REC_PID:-}" ]] && kill -0 "$REC_PID" 2>/dev/null; then
    kill -INT "$REC_PID" 2>/dev/null
    wait "$REC_PID" 2>/dev/null
  fi
  REC_PID=""

  if [[ -n "${SIM_PGID:-}" ]] && kill -0 "-$SIM_PGID" 2>/dev/null; then
    kill -TERM "-$SIM_PGID" 2>/dev/null
    sleep 3
    if kill -0 "-$SIM_PGID" 2>/dev/null; then
      kill -KILL "-$SIM_PGID" 2>/dev/null
    fi
    wait "$SIM_PGID" 2>/dev/null
  fi
  SIM_PGID=""
  set -e
}

cleanup_all() {
  cleanup_processes
  if [[ -n "${ROSCORE_PID:-}" ]] && kill -0 "$ROSCORE_PID" 2>/dev/null; then
    kill -TERM "$ROSCORE_PID" 2>/dev/null
    wait "$ROSCORE_PID" 2>/dev/null || true
  fi
}

trap cleanup_all EXIT INT TERM

ensure_roscore() {
  if rostopic list >/dev/null 2>&1; then
    return
  fi

  roscore >"$RUN_DIR/roscore.log" 2>&1 &
  ROSCORE_PID=$!
  for _ in $(seq 1 30); do
    if rostopic list >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done

  echo "Timed out waiting for roscore." >&2
  exit 1
}

wait_for_topic() {
  local topic="$1"
  local timeout_sec="$2"
  local waited=0
  while (( waited < timeout_sec )); do
    if rostopic list 2>/dev/null | grep -qx "$topic"; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

publish_start_trigger() {
  rostopic pub -1 /traj_start_trigger geometry_msgs/PoseStamped \
    "{pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}}}"
}

prepare_output_file() {
  local path="$1"
  if [[ -e "$path" || -e "$path.active" ]]; then
    if [[ "$OVERWRITE" -eq 1 ]]; then
      rm -f "$path" "$path.active"
    else
      echo "Output already exists: $path" >&2
      echo "Use --overwrite if you want to replace existing outputs." >&2
      exit 1
    fi
  fi
}

run_one() {
  local method="$1"
  local run_idx="$2"
  local bag_path="$RUN_DIR/${method}_run${run_idx}.bag"
  local run_log="$RUN_DIR/${method}_run${run_idx}.log"
  local record_log="$RUN_DIR/${method}_run${run_idx}_record.log"

  prepare_output_file "$bag_path"
  if [[ "$OVERWRITE" -eq 1 ]]; then
    rm -f "$run_log" "$record_log"
  elif [[ -e "$run_log" || -e "$record_log" ]]; then
    echo "Log file already exists for ${method}_run${run_idx}; use --overwrite to replace it." >&2
    exit 1
  fi

  echo
  echo "=== [$MAP_NAME] $method run $run_idx/$RUNS ==="

  rosbag record -O "$bag_path" \
    /drone_0/broadcast/blind_info \
    /drone_0/lidar_slam/odom >"$record_log" 2>&1 &
  REC_PID=$!

  if [[ "$method" == "rl" ]]; then
    setsid bash -lc "cd '$REPO_ROOT' && ./run.sh '$UGV_NUM' '$MAP_NAME' rl '$RL_MODEL_PATH' '$RL_DEVICE'" >"$run_log" 2>&1 &
  else
    setsid bash -lc "cd '$REPO_ROOT' && ./run.sh '$UGV_NUM' '$MAP_NAME' vrptw" >"$run_log" 2>&1 &
  fi
  SIM_PGID=$!

  if ! wait_for_topic "/drone_0/lidar_slam/odom" "$STARTUP_TIMEOUT"; then
    echo "Timed out waiting for /drone_0/lidar_slam/odom. See $run_log" >&2
    cleanup_processes
    exit 1
  fi

  sleep "$TRIGGER_DELAY"
  publish_start_trigger >/dev/null

  echo "Recording $method run $run_idx for ${RUN_DURATION}s..."
  sleep "$RUN_DURATION"

  cleanup_processes
  sleep "$BETWEEN_RUN_DELAY"
}

compare_one() {
  local run_idx="$1"
  local compare_json="$RUN_DIR/compare_run${run_idx}.json"
  if [[ "$OVERWRITE" -eq 1 ]]; then
    rm -f "$compare_json"
  elif [[ -e "$compare_json" ]]; then
    echo "Compare output already exists: $compare_json" >&2
    echo "Use --overwrite if you want to replace existing outputs." >&2
    exit 1
  fi

  python3 "$SCRIPT_DIR/analyze_online_dispatch.py" compare \
    --baseline-bag "$RUN_DIR/vrptw_run${run_idx}.bag" \
    --candidate-bag "$RUN_DIR/rl_run${run_idx}.bag" \
    --baseline-label vrptw \
    --candidate-label rl \
    --arrival-radius "$ARRIVAL_RADIUS" \
    --output-json "$compare_json"
}

ensure_roscore

for run_idx in $(seq 1 "$RUNS"); do
  run_one vrptw "$run_idx"
  run_one rl "$run_idx"
  compare_one "$run_idx"
done

echo
echo "Online evaluation finished."
echo "Output directory: $RUN_DIR"
echo "Compare JSON files:"
for run_idx in $(seq 1 "$RUNS"); do
  echo "  $RUN_DIR/compare_run${run_idx}.json"
done
