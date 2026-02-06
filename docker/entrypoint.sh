#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash

if [ -f /work/MARSIM_ws/devel/setup.bash ]; then
  source /work/MARSIM_ws/devel/setup.bash
fi
if [ -f /work/Air_ws/devel/setup.bash ]; then
  source /work/Air_ws/devel/setup.bash
fi
if [ -f /work/Ground_ws/devel/setup.bash ]; then
  source /work/Ground_ws/devel/setup.bash
fi

exec "$@"
