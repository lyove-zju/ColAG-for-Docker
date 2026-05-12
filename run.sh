#!/bin/bash

if ! pgrep -x "roscore" > /dev/null
then
    roscore &
fi

ugv_num=${1:-1}
map_arg=${2:-}
dispatch_method=${3:-vrptw}
rl_model_arg=${4:-}
rl_device=${5:-cpu}
ugv_side_pair=${UGV_SIDE_PAIR:-false}
if [ "$ugv_side_pair" = "1" ]; then
    ugv_side_pair=true
fi
topo_deadend_enable=${TOPO_DEADEND:-false}
if [ "$topo_deadend_enable" = "1" ]; then
    topo_deadend_enable=true
fi
topo_detour_enable=${TOPO_DETOUR:-false}
if [ "$topo_detour_enable" = "1" ]; then
    topo_detour_enable=true
fi
topo_auto_trigger=${TOPO_AUTO_TRIGGER:-false}
if [ "$topo_auto_trigger" = "1" ]; then
    topo_auto_trigger=true
fi
topo_ready_timeout=${TOPO_READY_TIMEOUT:-180}
topo_ready_frames=${TOPO_READY_FRAMES:-2}
topo_ready_min_cells=${TOPO_READY_MIN_CELLS:-}
topo_ready_stable_frames=${TOPO_READY_STABLE_FRAMES:-3}
topo_ready_stable_seconds=${TOPO_READY_STABLE_SECONDS:-8.0}
topo_ready_ratio=${TOPO_READY_RATIO:-0.98}

WORKSPACE_DIR=$(pwd)

MARSIM="$WORKSPACE_DIR/MARSIM_ws"
UGV="$WORKSPACE_DIR/Ground_ws"
UAV="$WORKSPACE_DIR/Air_ws"
MAP_RESOURCE_DIR="$WORKSPACE_DIR/MARSIM_ws/src/MARSIM/map_generator/resource"
DEFAULT_MAP="$MAP_RESOURCE_DIR/80obs.pcd"

absolute_path() {
    local candidate="$1"

    case "$candidate" in
        /*) printf '%s\n' "$candidate" ;;
        *) printf '%s\n' "$WORKSPACE_DIR/$candidate" ;;
    esac
}

resolve_map_path() {
    local candidate="$1"

    if [ -z "$candidate" ]; then
        printf '%s\n' "$DEFAULT_MAP"
        return 0
    fi

    if [ -f "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    if [ -f "$WORKSPACE_DIR/$candidate" ]; then
        printf '%s\n' "$WORKSPACE_DIR/$candidate"
        return 0
    fi

    if [[ "$candidate" != *.pcd ]]; then
        candidate="${candidate}.pcd"
    fi

    if [ -f "$MAP_RESOURCE_DIR/$candidate" ]; then
        printf '%s\n' "$MAP_RESOURCE_DIR/$candidate"
        return 0
    fi

    return 1
}

if ! map_name=$(resolve_map_path "$map_arg"); then
    echo "Map not found: $map_arg" >&2
    echo "Try a file under $MAP_RESOURCE_DIR, for example: 80obs, 60obs, small_forest01cutoff, randomcube" >&2
    exit 1
fi

topo_deadend_scenario=""
map_basename=$(basename "$map_name")
case "$map_basename" in
    u_shape.pcd|u_shape_new.pcd|u_shape_new2.pcd) topo_deadend_scenario="u" ;;
    v_shape.pcd) topo_deadend_scenario="v" ;;
    deadend.pcd) topo_deadend_scenario="deadend" ;;
esac

if [ -z "$topo_ready_min_cells" ]; then
    case "$topo_deadend_scenario" in
        u) topo_ready_min_cells=1500 ;;
        *) topo_ready_min_cells=2500 ;;
    esac
fi

if [ "$topo_deadend_enable" = "true" ] && [ -z "$topo_deadend_scenario" ]; then
    echo "TOPO_DEADEND requested, but $map_basename is not a structured U/V/deadend map; topo closures disabled for this run." >&2
    topo_deadend_enable=false
fi

if [ "$dispatch_method" != "vrptw" ] && [ "$dispatch_method" != "rl" ]; then
    echo "Unsupported dispatch method: $dispatch_method" >&2
    echo "Use one of: vrptw, rl" >&2
    exit 1
fi

rl_model_path=""
if [ "$dispatch_method" = "rl" ]; then
    if [ -z "$rl_model_arg" ]; then
        echo "dispatch_method=rl requires a model path as the 4th argument." >&2
        echo "Example: ./run.sh 3 80obs rl checkpoints/dispatch_bc.pt" >&2
        exit 1
    fi
    if [ -f "$rl_model_arg" ]; then
        rl_model_path=$(absolute_path "$rl_model_arg")
    elif [ -f "$WORKSPACE_DIR/$rl_model_arg" ]; then
        rl_model_path=$(absolute_path "$rl_model_arg")
    else
        echo "RL model not found: $rl_model_arg" >&2
        exit 1
    fi
fi

cd "$MARSIM" && catkin_make
cd "$UGV" && catkin_make
cd "$UAV" && catkin_make

source "$UAV/devel/setup.sh" && roslaunch ego_planner rviz.launch &
source "$UAV/devel/setup.sh" && roslaunch ego_planner swarm_sim.launch ugv_num:="$ugv_num" dispatch_method:="$dispatch_method" rl_model_path:="$rl_model_path" rl_device:="$rl_device" topo_deadend_enable:="$topo_deadend_enable" topo_deadend_scenario:="$topo_deadend_scenario" topo_detour_enable:="$topo_detour_enable" &
source "$UGV/devel/setup.sh" && roslaunch ego_planner swarm_sim.launch ugv_num:="$ugv_num" ugv_side_pair:="$ugv_side_pair" topo_detour_enable:="$topo_detour_enable" &
if [ "$topo_deadend_enable" = "true" ] && [ "$topo_auto_trigger" = "true" ]; then
    echo "TOPO_AUTO_TRIGGER enabled: waiting for topo obstacle cells in every UGV grid_map before /traj_start_trigger." >&2
    detour_args=()
    if [ "$topo_detour_enable" = "true" ]; then
        detour_args+=(--detour-enable)
    fi
    source "$UAV/devel/setup.sh" && python3 "$UAV/src/planner/plan_env/scripts/topo_ready_trigger.py" --ugv-num "$ugv_num" --timeout "$topo_ready_timeout" --ready-frames "$topo_ready_frames" --min-closure-cells "$topo_ready_min_cells" --stable-frames "$topo_ready_stable_frames" --stable-seconds "$topo_ready_stable_seconds" --required-ratio "$topo_ready_ratio" "${detour_args[@]}" &
fi
source "$MARSIM/devel/setup.sh" && roslaunch test_interface single_drone_vlp32.launch map_name:="$map_name"
