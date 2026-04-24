#!/usr/bin/env python3
"""ROS runtime node for RL-based UAV dispatch."""

from __future__ import annotations

import os

import rospy
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from custom_msgs.msg import blind_info

from dispatch_dataset import tensorize_observation
from dispatch_model import load_policy_checkpoint
from dispatch_runtime import (
    BlindTaskStore,
    DispatchObservationBuilder,
    build_path_msg,
    build_waypoint_marker,
    sanitize_slot_order,
    slot_order_to_ugv_ids,
    slot_order_to_waypoints,
)


class RLGuideManager:
    def __init__(self):
        self.drone_id_ = rospy.get_param("~drone_id", 0)
        self.ugv_num_ = rospy.get_param("~ugv_num", 1)
        self.uav_num_ = rospy.get_param("~uav_num", 1)
        self.replan_period_ = rospy.get_param("~replan_period", 1.0)
        self.v_max_ = rospy.get_param("~v_max", 1.0)
        self.a_max_ = rospy.get_param("~a_max", 1.0)
        self.time_reslolution_ = rospy.get_param("~time_reslolution", 100)
        self.dispatch_method_ = rospy.get_param("~dispatch_method", "rl")
        self.rl_model_path_ = rospy.get_param("~rl_model_path", "")
        self.rl_device_ = rospy.get_param("~rl_device", "cpu")

        if self.dispatch_method_ != "rl":
            raise RuntimeError("guide_plan_rl.py must only run with dispatch_method=rl.")
        if self.uav_num_ != 1:
            raise RuntimeError("RL dispatch v1 only supports uav_num=1.")
        if self.ugv_num_ > 10:
            raise RuntimeError("RL dispatch v1 only supports ugv_num <= 10.")
        if not self.rl_model_path_:
            raise RuntimeError("dispatch_method=rl requires a non-empty rl_model_path.")
        if not os.path.isfile(self.rl_model_path_):
            raise RuntimeError(f"RL model not found: {self.rl_model_path_}")

        self.pos_ = [0.0, 0.0, 0.0]
        self.vel_ = [0.0, 0.0, 0.0]
        self.have_odom_ = False
        self.last_order_slots_ = []

        self.task_store_ = BlindTaskStore(self.ugv_num_)
        self.obs_builder_ = DispatchObservationBuilder(
            ugv_num=self.ugv_num_,
            v_max=self.v_max_,
            a_max=self.a_max_,
            time_resolution=self.time_reslolution_,
        )
        self.model_ = load_policy_checkpoint(self.rl_model_path_, device=self.rl_device_)

        self.ugv_info_sub_ = rospy.Subscriber("blind_info", blind_info, self.UGVinfoCallback, queue_size=200)
        self.self_odom_sub_ = rospy.Subscriber(
            "self_odom",
            Odometry,
            self.selfOdomCallback,
            queue_size=10,
            tcp_nodelay=True,
        )
        self.waypoint_pub_ = rospy.Publisher("guide_waypoint", Path, queue_size=10)
        self.waypoint_marker_pub_ = rospy.Publisher("guide_waypoint_visual", Marker, queue_size=10)

        rospy.loginfo(
            "[RL Dispatch] Loaded model %s on %s for drone_%d.",
            self.rl_model_path_,
            self.rl_device_,
            self.drone_id_,
        )

    def publish_route(self, observation, order_slots):
        sanitized = sanitize_slot_order(order_slots, observation["active_mask"])
        waypoints = slot_order_to_waypoints(sanitized, observation["slot_waypoints"])
        ugv_order = slot_order_to_ugv_ids(sanitized, observation["slot_task_ids"])
        self.last_order_slots_ = sanitized

        self.waypoint_marker_pub_.publish(build_waypoint_marker(waypoints))
        self.waypoint_pub_.publish(build_path_msg(waypoints))
        rospy.loginfo("[RL Dispatch] published order %s", ugv_order)

    def infer_route(self, observation):
        batch = tensorize_observation(observation, device=self.rl_device_)
        output = self.model_.greedy_order(batch)
        return output["actions"][0].detach().cpu().tolist()

    def UGVinfoCallback(self, msg):
        self.task_store_.update_from_blind_info(msg)
        if not self.have_odom_:
            rospy.logwarn_throttle(2.0, "[RL Dispatch] waiting for self odom before planning.")
            return

        tasks = self.task_store_.active_tasks()
        if not tasks:
            return

        observation = self.obs_builder_.build_observation(
            tasks=tasks,
            pos=self.pos_,
            vel=self.vel_,
            now_sec=rospy.Time.now().to_sec(),
            previous_order_slots=self.last_order_slots_,
        )
        route_order = self.infer_route(observation)
        self.publish_route(observation, route_order)

    def selfOdomCallback(self, msg):
        self.pos_[0] = msg.pose.pose.position.x
        self.pos_[1] = msg.pose.pose.position.y
        self.pos_[2] = msg.pose.pose.position.z
        self.vel_[0] = msg.twist.twist.linear.x
        self.vel_[1] = msg.twist.twist.linear.y
        self.vel_[2] = msg.twist.twist.linear.z
        self.have_odom_ = True


if __name__ == "__main__":
    rospy.init_node("guide_planner_rl")
    manager = RLGuideManager()
    rospy.spin()
