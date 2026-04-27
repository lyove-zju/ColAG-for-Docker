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

cd "$REPO_ROOT"

echo "[1/4] Generating VRPTW expert dataset..."
python3 "$SCRIPT_DIR/dispatch_expert.py" \
  --output "$EXPERT_JSONL" \
  --num-samples "$NUM_SAMPLES" \
  --ugv-num "$UGV_NUM" \
  --seed "$SEED" \
  --map-size-x "$MAP_SIZE_X" \
  --map-size-y "$MAP_SIZE_Y" | tee "$EXPERT_LOG"

echo "[2/4] Running BC pretraining..."
python3 "$SCRIPT_DIR/train_dispatch_rl.py" bc \
  --dataset "$EXPERT_JSONL" \
  --output "$BC_CKPT" \
  --ugv-num "$UGV_NUM" \
  --epochs "$BC_EPOCHS" \
  --batch-size "$BC_BATCH_SIZE" \
  --lr "$BC_LR" \
  --device "$DEVICE" | tee "$BC_LOG"

echo "[3/4] Running PPO finetuning..."
python3 "$SCRIPT_DIR/train_dispatch_rl.py" ppo \
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
  --map-size-y "$MAP_SIZE_Y" | tee "$PPO_LOG"

echo "[4/4] Evaluating RL policy..."
python3 "$SCRIPT_DIR/eval_dispatch_rl.py" \
  --checkpoint "$PPO_CKPT" \
  --ugv-num "$UGV_NUM" \
  --episodes "$EVAL_EPISODES" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --output-json "$EVAL_JSON" | tee "$EVAL_LOG"

cat > "$PIPELINE_SUMMARY" <<EOF
experiment_dir=$EXPERIMENT_DIR
expert_jsonl=$EXPERT_JSONL
bc_checkpoint=$BC_CKPT
ppo_checkpoint=$PPO_CKPT
eval_metrics_json=$EVAL_JSON

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
echo "Next RL sim command:  ./run.sh $UGV_NUM 80obs rl $PPO_CKPT $DEVICE"
