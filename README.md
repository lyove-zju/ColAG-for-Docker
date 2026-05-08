# ColAG

**ColAG:** A **Col**laborative **A**ir-**G**round Framework for Perception-Limited UGVs' Navigation

## Table of Contents

1. [About](#1-about)
2. [How to Use](#2-how-to-use)
3. [Acknowledgments](#3-acknowledgments)
4. [License](#4-license)
5. [Maintenance](#5-maintenance)

## 1. About

**Author**: Zhehan Li $^\dagger$, Rui Mao $^\dagger$, Nanhe Chen, Chao Xu, Fei Gao, Yanjun Cao *

**Preprint Paper**: [ColAG: A Collaborative Air-Ground Framework for Perception-Limited UGVs' Navigation](https://arxiv.org/abs/2310.13324).

**Related Video**: [Bilibili](https://www.bilibili.com/video/BV1by421a73g/).
 
Accepted in [ICRA2024](https://2024.ieee-icra.org/).

```bib
@article{li2023colag,
  title={ColAG: A Collaborative Air-Ground Framework for Perception-Limited UGVs' Navigation},
  author={Li, Zhehan and Mao, Rui and Chen, Nanhe and Xu, Chao and Gao, Fei and Cao, Yanjun},
  journal={arXiv preprint arXiv:2310.13324},
  year={2023}
}
```

If our source code is used in your academic projects, please cite our paper. Thank you!

## 2. How to Use

Compiling tests passed on Ubuntu 20.04 with ros1 installed.

- Follow the [Ubuntu install of ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
- Follow the installation of [MARSIM](https://github.com/hku-mars/MARSIM)
- Follow the installation of [EGO-Swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)
- Follow the installation of [OR-Tools](https://github.com/google/or-tools), use the Python version
- Follow the installation of [diablo_mpc](https://github.com/GaoLon/diablo_mpc)

For convenience, we listed the installation steps here

```sh
sudo apt install libglfw3-dev libglew-dev libarmadillo-dev libzmqpp-dev ros-noetic-mavros

git clone https://github.com/osqp/osqp
cd osqp
mkdir build
cd build
cmake -G "Unix Makefiles" .. 
cmake --build .
sudo cmake --build . --target install
# if meet "CMake 3.18 or higher is required. 
# You are running version 3.16.3", 
# change "cmake_minimum_required(VERSION 3.18)" 
# to "cmake_minimum_required(VERSION 3.16)"
# in all CMakeLists.txt.

git clone https://github.com/robotology/osqp-eigen.git
cd osqp-eigen
mkdir build
cd build
cmake ..
make
sudo make install
# if meet error, may caused by version mismatch
# we backup osqp and osqp-eigen source code in folder .bak,
# which is tested on Ubuntu 20.04

sudo apt install python3-pip
python3 -m pip install --upgrade --user ortools
```

The Air_ws, Ground_ws, MARSIM_ws are separated catkin workspace, since they based on ego-swarm

```sh
cd MARSIM_ws
catkin_make
```

```sh
cd Air_ws
catkin_make
```

```sh
cd Ground_ws
catkin_make
```

Open rviz for visualization

```sh
cd Air_ws
source devel/setup.sh
roslaunch ego_planner rviz.launch
```

Run the following in different command windows

Ensure the `ugv_num` is same in both swarm_sim.launch

```sh
cd MARSIM_ws
source devel/setup.sh
roslaunch test_interface single_drone_vlp32.launch
```

```sh
cd Ground_ws
source devel/setup.sh
roslaunch ego_planner swarm_sim.launch ugv_num:=3
```

```sh
cd Air_ws
source devel/setup.sh
roslaunch ego_planner swarm_sim.launch ugv_num:=3
```

Or use the [run.sh](run.sh) to run the launches above

```sh
./run.sh 3 # the ugv num
```

Optionally, pass a second argument to choose another MARSIM map while keeping the original launch flow unchanged:

```sh
./run.sh 3 60obs
./run.sh 3 small_forest01cutoff
./run.sh 3 MARSIM_ws/src/MARSIM/map_generator/resource/randomcube.pcd
```

The UAV dispatch baseline remains VRPTW by default. You can keep the original behavior or explicitly switch to the new RL dispatch module:

```sh
./run.sh 3 80obs vrptw
./run.sh 3 80obs rl /absolute/path/to/dispatch_policy.pt cpu
```

You can also launch the Air side directly with ROS parameters:

```sh
cd Air_ws
source devel/setup.sh
roslaunch ego_planner swarm_sim.launch \
  ugv_num:=3 \
  dispatch_method:=rl \
  rl_model_path:=/absolute/path/to/dispatch_policy.pt \
  rl_device:=cpu
```

The RL module only replaces the UAV dispatch order generator. UGV planning, collision detection, bridge logic, `blind_info`, and UAV/UGV map sharing remain unchanged. See [RL_DISPATCH_RULES.md](RL_DISPATCH_RULES.md) for the hard constraints.

Training and evaluation utilities are provided under `Air_ws/src/swarm_support/scripts/`:

```sh
python3 Air_ws/src/swarm_support/scripts/dispatch_expert.py \
  --output tmp_dispatch_expert.jsonl \
  --num-samples 2000 \
  --ugv-num 3

python3 Air_ws/src/swarm_support/scripts/train_dispatch_rl.py bc \
  --dataset tmp_dispatch_expert.jsonl \
  --output dispatch_bc.pt \
  --ugv-num 3

python3 Air_ws/src/swarm_support/scripts/train_dispatch_rl.py ppo \
  --init-checkpoint dispatch_bc.pt \
  --output dispatch_ppo.pt \
  --ugv-num 3

python3 Air_ws/src/swarm_support/scripts/eval_dispatch_rl.py \
  --checkpoint dispatch_ppo.pt \
  --ugv-num 3
```

These scripts require PyTorch, and the expert exporter additionally requires OR-Tools.

If you want the whole chain in one command, use the pipeline wrapper:

```sh
bash Air_ws/src/swarm_support/scripts/run_dispatch_pipeline.sh --ugv-num 3
```

If you set `--output-root /work/rl_dispatch_runs/formal_ugv3` and do not pass `--experiment-name`, the script will create a timestamped run directory such as `/work/rl_dispatch_runs/formal_ugv3/20260424_153000/`.

It will automatically generate:

- expert dataset JSONL
- BC checkpoint `.pt`
- PPO checkpoint `.pt`
- evaluation logs
- `eval_metrics.json`

By default the outputs are written to `tmp_rl_dispatch_runs/<experiment_name>/`.

### GPU Training Container

If you want to keep the ROS Noetic simulation container unchanged and run RL training in a separate GPU-enabled container, build the dedicated trainer image:

```sh
docker build -f docker/Dockerfile.train-gpu -t colag-train-gpu .
```

Then launch training with a bind-mounted workspace so outputs are still written back to `/work`:

```sh
docker run --rm -it \
  --gpus all \
  -v "$PWD":/work \
  -w /work \
  --name colag_train_gpu \
  colag-train-gpu \
  bash -lc 'bash Air_ws/src/swarm_support/scripts/run_dispatch_pipeline.sh --ugv-num 3 --seed 7 --num-samples 5000 --device cuda --map-size-x 35.0 --map-size-y 35.0 --bc-epochs 20 --bc-batch-size 128 --bc-lr 1e-3 --ppo-updates 50 --ppo-epochs 4 --ppo-batch-size 64 --ppo-lr 3e-4 --eval-episodes 500 --output-root /work/rl_dispatch_runs/formal_ugv3'
```

If you do not pass `--experiment-name`, the outputs will be written to `/work/rl_dispatch_runs/formal_ugv3/<timestamp>/`.

### Online Dispatch Metrics

To compare the online VRPTW baseline and the RL replacement in the real ROS/MARSIM loop, record the UAV odometry and dispatch requests during each run:

```sh
rosbag record -O /work/dispatch_vrptw_80obs.bag \
  /drone_0/broadcast/blind_info \
  /drone_0/lidar_slam/odom
```

Run the VRPTW baseline:

```sh
./run.sh 3 80obs vrptw
```

Then record the RL run in a second bag:

```sh
rosbag record -O /work/dispatch_rl_80obs.bag \
  /drone_0/broadcast/blind_info \
  /drone_0/lidar_slam/odom
```

```sh
./run.sh 3 80obs rl /work/rl_dispatch_runs/formal_ugv3/<timestamp>/dispatch_ppo_ugv3.pt cpu
```

After both bags are saved, analyze one run:

```sh
python3 Air_ws/src/swarm_support/scripts/analyze_online_dispatch.py analyze \
  --bag /work/dispatch_vrptw_80obs.bag \
  --label vrptw \
  --output-json /work/dispatch_vrptw_80obs_metrics.json
```

Or compare two runs directly:

```sh
python3 Air_ws/src/swarm_support/scripts/analyze_online_dispatch.py compare \
  --baseline-bag /work/dispatch_vrptw_80obs.bag \
  --candidate-bag /work/dispatch_rl_80obs.bag \
  --baseline-label vrptw \
  --candidate-label rl \
  --output-json /work/dispatch_compare_80obs.json
```

The analyzer reports:

- `deadline_hit_rate` / `deadline_hit_count`
- `missed_count`
- `avg_tardiness_seconds`
- `avg_response_time_seconds`
- `uav_flight_distance_m`
- `uav_flight_time_seconds`

It treats each positive `blind_info` message as one online support request and checks whether the UAV reaches the requested support point within the configured XY arrival radius before the request deadline.

For repeated experiments, use the wrapper below to run three VRPTW/RL pairs automatically on one map:

```sh
bash Air_ws/src/swarm_support/scripts/run_online_eval.sh \
  --map 96obs \
  --ugv-num 3 \
  --runs 3 \
  --rl-model /work/rl_dispatch_runs/formal_ugv3/<timestamp>/dispatch_ppo_ugv3.pt \
  --rl-device cpu \
  --output-root /work/online_eval \
  --run-duration 150
```

This creates:

- `/work/online_eval/96obs/vrptw_run1.bag`
- `/work/online_eval/96obs/rl_run1.bag`
- `/work/online_eval/96obs/compare_run1.json`
- the same files for `run2` and `run3`

If you are repeating an experiment with the same names, add `--overwrite`.

You can also generate more `40obs/60obs/80obs`-style ASCII maps offline without changing the launch logic:

```sh
python3 MARSIM_ws/src/MARSIM/map_generator/scripts/generate_legacy_obs_map.py custom_72 \
  --obs-num 72 \
  --seed 7 \
  --min-distance 1.6

./run.sh 3 custom_72
```

Optional map size:

```sh
python3 MARSIM_ws/src/MARSIM/map_generator/scripts/generate_legacy_obs_map.py custom_wide \
  --obs-num 96 \
  --seed 11 \
  --min-distance 1.4 \
  --size-x 45.0 \
  --size-y 35.0
```

Structured V/U/dead-end maps can also be generated offline. These maps use the
same ASCII PCD, ground plane, grid resolution, box obstacle, and launch flow as
the legacy maps; only the obstacle arrangement changes:

```sh
python3 MARSIM_ws/src/MARSIM/map_generator/scripts/generate_structured_obs_map.py

./run.sh 3 v_shape vrptw
./run.sh 3 u_shape vrptw
./run.sh 3 deadend vrptw
```

Use `--shape-clearance` to move random background obstacles farther from the
V/U/dead-end structures while keeping the structures themselves unchanged.

Topo dead-end closure is disabled by default. For structured topo tests, enable
it explicitly:

```sh
TOPO_DEADEND=1 ./run.sh 3 u_shape vrptw
```

To avoid starting UGVs before the UAV has generated and shared the virtual
topo obstacles, use the optional ready gate. It waits until the topo obstacle
cloud is stable and every `/ugv_i/broadcast/grid_map.occu_address` contains
those cells, then publishes the normal `/traj_start_trigger` automatically:

```sh
TOPO_DEADEND=1 TOPO_AUTO_TRIGGER=1 ./run.sh 3 u_shape vrptw
```

To test only the original left/right UGV starts and remove the middle start,
run two UGVs with the side-pair switch:

```sh
UGV_SIDE_PAIR=1 TOPO_DEADEND=1 TOPO_AUTO_TRIGGER=1 ./run.sh 2 u_shape vrptw
```

For `u_shape`, the gate waits by default for at least 1500 topo obstacle cells
and 8 seconds of stable topo cloud before checking UGV maps. You can override
these with `TOPO_READY_MIN_CELLS` and `TOPO_READY_STABLE_SECONDS`.

For RL dispatch, keep the same map-name position and pass the dispatch method
and model arguments as usual:

```sh
./run.sh 3 v_shape rl /absolute/path/to/dispatch_policy.pt cpu
```

Then send a trigger to start blind navigation

```sh
rostopic pub /traj_start_trigger geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0"
```

### Docker (Ubuntu 24.04 Host)

This project targets Ubuntu 20.04 + ROS Noetic. If your host is Ubuntu 24.04, you can run it in Docker.

Build the image:

```sh
docker build -f docker/Dockerfile -t colag:noetic .
```

Run the container (GUI enabled for RViz):

```sh
xhost +local:root
docker run -it --rm \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/work \
  --name colag \
  colag:noetic
```

If you need NVIDIA GPU acceleration, add `--gpus all`:

```sh
docker run -it --rm \
  --net=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/work \
  --name colag \
  colag:noetic
```

Inside the container, run the simulation:

```sh
cd /work
./run.sh 3
```

If the FSM stays at `WAIT_TARGET`, open another terminal into the running container:

```sh
docker exec -it colag bash
```
Then send the trigger:

```sh
rostopic pub /traj_start_trigger geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0"
```

If `rostopic` is not found in the new terminal, source ROS:

```sh
source /opt/ros/noetic/setup.bash
source /work/MARSIM_ws/devel/setup.bash
source /work/Ground_ws/devel/setup.bash
source /work/Air_ws/devel/setup.bash
```
```

## 3. Acknowledgments

**There are several important works which support this project:**

- [MARSIM](https://github.com/hku-mars/MARSIM): A lightweight point-realistic simulator for LiDAR-based UAVs.
- [EGO-Swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm): A Fully Autonomous and Decentralized Quadrotor Swarm System in Cluttered Environments.
- [OR-Tools](https://github.com/google/or-tools): Google's software suite for combinatorial optimization.
- [diablo_mpc](https://github.com/GaoLon/diablo_mpc): A MPC for diablo configuration(see as differential car).

## 4. License

The source code is released under the [GPLv3](https://www.gnu.org/licenses/) license.

## 5. Maintenance

We are still working on extending the proposed system and improving code reliability.

For any technical issues, please contact Zhehan Li (<zhehanli@zju.edu.cn>) or Yanjun Cao (<yanjunhi@zju.edu.cn>).

For commercial inquiries, please contact Yanjun Cao (<yanjunhi@zju.edu.cn>).
