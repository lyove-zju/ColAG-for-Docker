#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)

UGV_NUM=3
SEED=7
NUM_SAMPLES=1000
DEVICE=cpu
MAP_SIZE_X=35.0
MAP_SIZE_Y=35.0
BC_EPOCHS=5
BC_BATCH_SIZE=64
BC_LR=1e-3
PPO_UPDATES=10
PPO_EPOCHS=3
PPO_BATCH_SIZE=32
PPO_LR=3e-4
PPO_MODE=ppo
PPO_HORIZON_SEC=150.0
PPO_REPLAN_PERIOD=10.0
PPO_ARRIVAL_RADIUS=0.8
PPO_GAMMA=0.98
PPO_MAX_DECISIONS_PER_EPISODE=80
PPO_SUCCESS_REWARD=4.0
PPO_MISS_PENALTY=6.0
PPO_TARDINESS_WEIGHT=0.25
PPO_ONGOING_TARDINESS_WEIGHT=0.04
PPO_RESPONSE_TIME_WEIGHT=0.05
PPO_FLIGHT_TIME_WEIGHT=0.01
PPO_FLIGHT_DISTANCE_WEIGHT=0.005
PPO_ROUTE_CHURN_WEIGHT=0.02
EVAL_EPISODES=200
OUTPUT_ROOT="$REPO_ROOT/rl_dispatch_runs"
EXPERIMENT_NAME=""

usage() {
  cat <<'EOF'
Usage:
  bash Air_ws/src/swarm_support/scripts/run_dispatch_pipeline.sh [options]

Options:
  --ugv-num N
  --seed N
  --num-samples N
  --device cpu|cuda
  --map-size-x FLOAT
  --map-size-y FLOAT
  --bc-epochs N
  --bc-batch-size N
  --bc-lr FLOAT
  --ppo-updates N
  --ppo-epochs N
  --ppo-batch-size N
  --ppo-lr FLOAT
  --ppo-mode ppo|ppo_event
  --ppo-horizon-sec FLOAT
  --ppo-replan-period FLOAT
  --ppo-arrival-radius FLOAT
  --ppo-gamma FLOAT
  --ppo-max-decisions-per-episode N
  --ppo-success-reward FLOAT
  --ppo-miss-penalty FLOAT
  --ppo-tardiness-weight FLOAT
  --ppo-ongoing-tardiness-weight FLOAT
  --ppo-response-time-weight FLOAT
  --ppo-flight-time-weight FLOAT
  --ppo-flight-distance-weight FLOAT
  --ppo-route-churn-weight FLOAT
  --eval-episodes N
  --output-root DIR
  --experiment-name NAME
  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ugv-num)
      UGV_NUM="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --map-size-x)
      MAP_SIZE_X="$2"
      shift 2
      ;;
    --map-size-y)
      MAP_SIZE_Y="$2"
      shift 2
      ;;
    --bc-epochs)
      BC_EPOCHS="$2"
      shift 2
      ;;
    --bc-batch-size)
      BC_BATCH_SIZE="$2"
      shift 2
      ;;
    --bc-lr)
      BC_LR="$2"
      shift 2
      ;;
    --ppo-updates)
      PPO_UPDATES="$2"
      shift 2
      ;;
    --ppo-epochs)
      PPO_EPOCHS="$2"
      shift 2
      ;;
    --ppo-batch-size)
      PPO_BATCH_SIZE="$2"
      shift 2
      ;;
    --ppo-lr)
      PPO_LR="$2"
      shift 2
      ;;
    --ppo-mode)
      PPO_MODE="$2"
      shift 2
      ;;
    --ppo-horizon-sec)
      PPO_HORIZON_SEC="$2"
      shift 2
      ;;
    --ppo-replan-period)
      PPO_REPLAN_PERIOD="$2"
      shift 2
      ;;
    --ppo-arrival-radius)
      PPO_ARRIVAL_RADIUS="$2"
      shift 2
      ;;
    --ppo-gamma)
      PPO_GAMMA="$2"
      shift 2
      ;;
    --ppo-max-decisions-per-episode)
      PPO_MAX_DECISIONS_PER_EPISODE="$2"
      shift 2
      ;;
    --ppo-success-reward)
      PPO_SUCCESS_REWARD="$2"
      shift 2
      ;;
    --ppo-miss-penalty)
      PPO_MISS_PENALTY="$2"
      shift 2
      ;;
    --ppo-tardiness-weight)
      PPO_TARDINESS_WEIGHT="$2"
      shift 2
      ;;
    --ppo-ongoing-tardiness-weight)
      PPO_ONGOING_TARDINESS_WEIGHT="$2"
      shift 2
      ;;
    --ppo-response-time-weight)
      PPO_RESPONSE_TIME_WEIGHT="$2"
      shift 2
      ;;
    --ppo-flight-time-weight)
      PPO_FLIGHT_TIME_WEIGHT="$2"
      shift 2
      ;;
    --ppo-flight-distance-weight)
      PPO_FLIGHT_DISTANCE_WEIGHT="$2"
      shift 2
      ;;
    --ppo-route-churn-weight)
      PPO_ROUTE_CHURN_WEIGHT="$2"
      shift 2
      ;;
    --eval-episodes)
      EVAL_EPISODES="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --experiment-name)
      EXPERIMENT_NAME="$2"
      shift 2
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

if [[ "$PPO_MODE" != "ppo" && "$PPO_MODE" != "ppo_event" ]]; then
  echo "--ppo-mode must be one of: ppo, ppo_event" >&2
  exit 1
fi
EVAL_MODE=bandit
if [[ "$PPO_MODE" == "ppo_event" ]]; then
  EVAL_MODE=event
fi

if [[ -z "$EXPERIMENT_NAME" ]]; then
  EXPERIMENT_NAME="$(date +%Y%m%d_%H%M%S)"
fi

python3 - <<'PY'
import importlib.util
import sys

required = ["torch", "ortools"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python dependencies: " + ", ".join(missing), file=sys.stderr)
    print("Install them before running the pipeline.", file=sys.stderr)
    sys.exit(1)
PY

EXPERIMENT_DIR="$OUTPUT_ROOT/$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"

EXPERT_JSONL="$EXPERIMENT_DIR/expert_ugv${UGV_NUM}.jsonl"
BC_CKPT="$EXPERIMENT_DIR/dispatch_bc_ugv${UGV_NUM}.pt"
PPO_CKPT="$EXPERIMENT_DIR/dispatch_ppo_ugv${UGV_NUM}.pt"
EVAL_JSON="$EXPERIMENT_DIR/eval_metrics.json"
EXPERT_LOG="$EXPERIMENT_DIR/expert.log"
BC_LOG="$EXPERIMENT_DIR/bc.log"
PPO_LOG="$EXPERIMENT_DIR/ppo.log"
EVAL_LOG="$EXPERIMENT_DIR/eval.log"
PIPELINE_SUMMARY="$EXPERIMENT_DIR/pipeline_summary.txt"
BC_METRICS_JSONL="$EXPERIMENT_DIR/bc_metrics.jsonl"
PPO_METRICS_JSONL="$EXPERIMENT_DIR/ppo_metrics.jsonl"
TRAINING_PLOT_SVG="$EXPERIMENT_DIR/training_curves.svg"

cd "$REPO_ROOT"

echo "[1/5] Generating VRPTW expert dataset..."
python3 "$SCRIPT_DIR/dispatch_expert.py" \
  --output "$EXPERT_JSONL" \
  --num-samples "$NUM_SAMPLES" \
  --ugv-num "$UGV_NUM" \
  --seed "$SEED" \
  --map-size-x "$MAP_SIZE_X" \
  --map-size-y "$MAP_SIZE_Y" | tee "$EXPERT_LOG"

echo "[2/5] Running BC pretraining..."
python3 "$SCRIPT_DIR/train_dispatch_rl.py" bc \
  --dataset "$EXPERT_JSONL" \
  --output "$BC_CKPT" \
  --ugv-num "$UGV_NUM" \
  --epochs "$BC_EPOCHS" \
  --batch-size "$BC_BATCH_SIZE" \
  --lr "$BC_LR" \
  --device "$DEVICE" \
  --metrics-jsonl "$BC_METRICS_JSONL" | tee "$BC_LOG"

echo "[3/5] Running PPO finetuning..."
PPO_EXTRA_ARGS=()
if [[ "$PPO_MODE" == "ppo_event" ]]; then
  PPO_EXTRA_ARGS=(
    --horizon-sec "$PPO_HORIZON_SEC"
    --replan-period "$PPO_REPLAN_PERIOD"
    --arrival-radius "$PPO_ARRIVAL_RADIUS"
    --gamma "$PPO_GAMMA"
    --max-decisions-per-episode "$PPO_MAX_DECISIONS_PER_EPISODE"
    --success-reward "$PPO_SUCCESS_REWARD"
    --miss-penalty "$PPO_MISS_PENALTY"
    --tardiness-weight "$PPO_TARDINESS_WEIGHT"
    --ongoing-tardiness-weight "$PPO_ONGOING_TARDINESS_WEIGHT"
    --response-time-weight "$PPO_RESPONSE_TIME_WEIGHT"
    --flight-time-weight "$PPO_FLIGHT_TIME_WEIGHT"
    --flight-distance-weight "$PPO_FLIGHT_DISTANCE_WEIGHT"
    --route-churn-weight "$PPO_ROUTE_CHURN_WEIGHT"
  )
fi

python3 "$SCRIPT_DIR/train_dispatch_rl.py" "$PPO_MODE" \
  --init-checkpoint "$BC_CKPT" \
  --output "$PPO_CKPT" \
  --ugv-num "$UGV_NUM" \
  --updates "$PPO_UPDATES" \
  --ppo-epochs "$PPO_EPOCHS" \
  --batch-size "$PPO_BATCH_SIZE" \
  --lr "$PPO_LR" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --map-size-x "$MAP_SIZE_X" \
  --map-size-y "$MAP_SIZE_Y" \
  --metrics-jsonl "$PPO_METRICS_JSONL" \
  "${PPO_EXTRA_ARGS[@]}" | tee "$PPO_LOG"

echo "[4/5] Evaluating RL policy..."
EVAL_EXTRA_ARGS=(
  --eval-mode "$EVAL_MODE"
  --map-size-x "$MAP_SIZE_X"
  --map-size-y "$MAP_SIZE_Y"
)
if [[ "$PPO_MODE" == "ppo_event" ]]; then
  EVAL_EXTRA_ARGS=(
    --eval-mode event
    --map-size-x "$MAP_SIZE_X"
    --map-size-y "$MAP_SIZE_Y"
    --horizon-sec "$PPO_HORIZON_SEC"
    --replan-period "$PPO_REPLAN_PERIOD"
    --arrival-radius "$PPO_ARRIVAL_RADIUS"
    --max-decisions-per-episode "$PPO_MAX_DECISIONS_PER_EPISODE"
  )
fi

python3 "$SCRIPT_DIR/eval_dispatch_rl.py" \
  --checkpoint "$PPO_CKPT" \
  --ugv-num "$UGV_NUM" \
  --episodes "$EVAL_EPISODES" \
  --seed "$SEED" \
  --device "$DEVICE" \
  "${EVAL_EXTRA_ARGS[@]}" \
  --output-json "$EVAL_JSON" | tee "$EVAL_LOG"

echo "[5/5] Plotting training curves..."
python3 "$SCRIPT_DIR/plot_dispatch_training.py" \
  --bc-jsonl "$BC_METRICS_JSONL" \
  --ppo-jsonl "$PPO_METRICS_JSONL" \
  --bc-log "$BC_LOG" \
  --ppo-log "$PPO_LOG" \
  --eval-json "$EVAL_JSON" \
  --output "$TRAINING_PLOT_SVG"

cat > "$PIPELINE_SUMMARY" <<EOF
experiment_dir=$EXPERIMENT_DIR
expert_jsonl=$EXPERT_JSONL
bc_checkpoint=$BC_CKPT
ppo_checkpoint=$PPO_CKPT
eval_metrics_json=$EVAL_JSON
bc_metrics_jsonl=$BC_METRICS_JSONL
ppo_metrics_jsonl=$PPO_METRICS_JSONL
training_plot_svg=$TRAINING_PLOT_SVG
ppo_mode=$PPO_MODE
eval_mode=$EVAL_MODE
ppo_replan_period=$PPO_REPLAN_PERIOD
ppo_success_reward=$PPO_SUCCESS_REWARD
ppo_miss_penalty=$PPO_MISS_PENALTY
ppo_tardiness_weight=$PPO_TARDINESS_WEIGHT
ppo_ongoing_tardiness_weight=$PPO_ONGOING_TARDINESS_WEIGHT
ppo_response_time_weight=$PPO_RESPONSE_TIME_WEIGHT
ppo_flight_time_weight=$PPO_FLIGHT_TIME_WEIGHT
ppo_flight_distance_weight=$PPO_FLIGHT_DISTANCE_WEIGHT
ppo_route_churn_weight=$PPO_ROUTE_CHURN_WEIGHT

run_rl_sim_command=./run.sh $UGV_NUM 80obs rl $PPO_CKPT $DEVICE
run_vrptw_sim_command=./run.sh $UGV_NUM 80obs vrptw
EOF

echo
echo "Pipeline finished."
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Expert dataset:       $EXPERT_JSONL"
echo "BC checkpoint:        $BC_CKPT"
echo "PPO checkpoint:       $PPO_CKPT"
echo "Eval metrics JSON:    $EVAL_JSON"
echo "Training plot SVG:    $TRAINING_PLOT_SVG"
echo "Next RL sim command:  ./run.sh $UGV_NUM 80obs rl $PPO_CKPT $DEVICE"
