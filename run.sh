#!/bin/bash

source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}

wait_for_ros_master() {
    local attempt
    for attempt in $(seq 1 40); do
        if rosparam get /run_id >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.25
    done

    return 1
}

if ! rosnode list >/dev/null 2>&1; then
    roscore &
fi

if ! wait_for_ros_master; then
    echo "ROS master did not become ready on $ROS_MASTER_URI" >&2
    echo "If port 11311 is occupied by a stale process, stop it before rerunning run.sh." >&2
    exit 1
fi

ugv_num=${1:-1}
map_arg=${2:-}
dispatch_method=${3:-vrptw}
rl_model_arg=${4:-}
rl_device=${5:-cpu}

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

source "$UAV/devel/setup.sh" && roslaunch --wait ego_planner rviz.launch &
source "$UAV/devel/setup.sh" && roslaunch --wait ego_planner swarm_sim.launch ugv_num:="$ugv_num" dispatch_method:="$dispatch_method" rl_model_path:="$rl_model_path" rl_device:="$rl_device" &
source "$UGV/devel/setup.sh" && roslaunch --wait ego_planner swarm_sim.launch ugv_num:="$ugv_num" &
source "$MARSIM/devel/setup.sh" && roslaunch --wait test_interface single_drone_vlp32.launch map_name:="$map_name"
