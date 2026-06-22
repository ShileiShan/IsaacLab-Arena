# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

_LEFT_ARM_JOINT_PATTERN = re.compile(r"^joint[1-6]_l$")
_RIGHT_ARM_JOINT_PATTERN = re.compile(r"^joint[1-6]_r$")
_LEFT_GRIPPER_PATTERN = re.compile(r"^finger_joint.*_l$")
_RIGHT_GRIPPER_PATTERN = re.compile(r"^finger_joint.*_r$")


def left_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns left arm joint positions (joint1_l … joint6_l)."""
    robot = env.scene[asset_cfg.name]
    indices = [i for i, n in enumerate(robot.data.joint_names) if _LEFT_ARM_JOINT_PATTERN.match(n)]
    return wp.to_torch(robot.data.joint_pos)[:, indices]


def right_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns right arm joint positions (joint1_r … joint6_r)."""
    robot = env.scene[asset_cfg.name]
    indices = [i for i, n in enumerate(robot.data.joint_names) if _RIGHT_ARM_JOINT_PATTERN.match(n)]
    return wp.to_torch(robot.data.joint_pos)[:, indices]


def left_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns left gripper joint positions (finger_joint*_l), normalised to [0, 1]."""
    robot = env.scene[asset_cfg.name]
    indices = [i for i, n in enumerate(robot.data.joint_names) if _LEFT_GRIPPER_PATTERN.match(n)]
    return wp.to_torch(robot.data.joint_pos)[:, indices] / 0.035


def right_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns right gripper joint positions (finger_joint*_r), normalised to [0, 1]."""
    robot = env.scene[asset_cfg.name]
    indices = [i for i, n in enumerate(robot.data.joint_names) if _RIGHT_GRIPPER_PATTERN.match(n)]
    return wp.to_torch(robot.data.joint_pos)[:, indices] / 0.035


def left_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns left end-effector (hand_link_l) position (x, y, z) in world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("hand_link_l")
    return wp.to_torch(robot.data.body_pos_w)[:, body_idx, :]


def left_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns left end-effector (hand_link_l) orientation as quaternion (w, x, y, z) in world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("hand_link_l")
    return wp.to_torch(robot.data.body_quat_w)[:, body_idx, :]


def right_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns right end-effector (hand_link_r) position (x, y, z) in world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("hand_link_r")
    return wp.to_torch(robot.data.body_pos_w)[:, body_idx, :]


def right_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns right end-effector (hand_link_r) orientation as quaternion (w, x, y, z) in world frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("hand_link_r")
    return wp.to_torch(robot.data.body_quat_w)[:, body_idx, :]
