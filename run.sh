#!/bin/bash

if ! pgrep -x "roscore" > /dev/null
then
    roscore &
fi

ugv_num=${1:-1}
map_arg=${2:-}

WORKSPACE_DIR=$(pwd)

MARSIM="$WORKSPACE_DIR/MARSIM_ws"
UGV="$WORKSPACE_DIR/Ground_ws"
UAV="$WORKSPACE_DIR/Air_ws"
MAP_RESOURCE_DIR="$WORKSPACE_DIR/MARSIM_ws/src/MARSIM/map_generator/resource"
DEFAULT_MAP="$MAP_RESOURCE_DIR/80obs.pcd"

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

cd "$MARSIM" && catkin_make
cd "$UGV" && catkin_make
cd "$UAV" && catkin_make

source "$UAV/devel/setup.sh" && roslaunch ego_planner rviz.launch &
source "$UAV/devel/setup.sh" && roslaunch ego_planner swarm_sim.launch ugv_num:="$ugv_num" &
source "$UGV/devel/setup.sh" && roslaunch ego_planner swarm_sim.launch ugv_num:="$ugv_num" &
source "$MARSIM/devel/setup.sh" && roslaunch test_interface single_drone_vlp32.launch map_name:="$map_name"
