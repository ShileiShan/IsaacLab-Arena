# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Double Piper dual-arm robot embodiment for Isaac Lab Arena.

Hardware reference: Lightwheel double-Piper setup (two AgileX Piper 6-DOF arms
mounted on a common base, each with a two-finger parallel gripper).
"""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)

from isaaclab_arena.embodiments.double_piper.actions import SymmetricGripperPositionActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

from isaaclab_arena.embodiments.double_piper.observations import (
    left_arm_joint_pos,
    left_ee_pos,
    left_ee_quat,
    left_gripper_pos,
    right_arm_joint_pos,
    right_ee_pos,
    right_ee_quat,
    right_gripper_pos,
)

# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

_DOUBLE_PIPER_USD = "/workspaces/isaaclab_arena/assets/double_piper.usd"

# Default robot base orientation (xyzw quaternion).
# Adjust this if the USD model faces the wrong direction in simulation.
# Common values:
#   identity (no rotation):     (0, 0, 0, 1)
#   180° around Z (face -X→+X): (0, 0, 1, 0)
#    90° around Z (face +Y→+X): (0, 0, 0.7071068, 0.7071068)
#   -90° around Z (face -Y→+X): (0, 0, -0.7071068, 0.7071068)
_DEFAULT_ROT_XYZW = (0.0, 0.0, 0.0, 1.0)  # — robot faces -X toward the table


@configclass
class DoublePiperSceneCfg:
    """Scene assets contributed by the Double Piper embodiment."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_DOUBLE_PIPER_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),
            rot=_DEFAULT_ROT_XYZW,
            joint_pos={
                # Left arm — default to real-robot home pose (matches training data)
                "joint1_l": -0.6379,
                "joint2_l": 0.0215,
                "joint3_l": -0.4208,
                "joint4_l": 0.3144,
                "joint5_l": 0.7449,
                "joint6_l": -0.3596,
                # Right arm
                "joint1_r": 0.3084,
                "joint2_r": 0.0,
                "joint3_r": -0.4139,
                "joint4_r": -0.2013,
                "joint5_r": 0.6952,
                "joint6_r": 0.2756,
                # Left gripper
                "finger_joint_left_l": 0.035,
                "finger_joint_right_l": -0.035,
                # Right gripper
                "finger_joint_left_r": 0.035,
                "finger_joint_right_r": -0.035,
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]_l"],
                effort_limit=50.0,
                velocity_limit=20.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]_r"],
                effort_limit=50.0,
                velocity_limit=20.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint.*_l"],
                effort_limit=500.0,
                velocity_limit=0.5,
                stiffness=5000.0,
                damping=200.0,
            ),
            "right_gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint.*_r"],
                effort_limit=500.0,
                velocity_limit=0.5,
                stiffness=5000.0,
                damping=200.0,
            ),
        },
    )

    # End-effector frame tracker for both arms
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/piper_L/hand_link_l",
                name="ee_tcp_l",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/piper_R/hand_link_r",
                name="ee_tcp_r",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------


@configclass
class DoublePiperCameraCfg:
    """Three on-robot cameras: left wrist, right wrist and first-person."""

    left_hand_camera: CameraCfg | TiledCameraCfg = MISSING
    right_hand_camera: CameraCfg | TiledCameraCfg = MISSING
    first_person_camera: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        is_tiled = getattr(self, "_is_tiled_camera", True)
        CameraClass = TiledCameraCfg if is_tiled else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        # ------------------------------------------------------------------
        # Real hardware intrinsics (640×480 @ 30 fps)
        # Conversion: horizontal_aperture = focal_length_mm / fx * width
        #             vertical_aperture   = focal_length_mm / fy * height
        # focal_length reference = 19.3 mm
        # rot xyzw 
        # ------------------------------------------------------------------

        # 相机2 — D405  (fx=394.517, fy=394.177, cx=316.56, cy=237.26)
        # horizontal_aperture = 19.3 / 394.517 * 640 = 31.31 mm
        # vertical_aperture   = 19.3 / 394.177 * 480 = 23.50 mm
        self.left_hand_camera = CameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/piper_L/hand_link_l/left_hand_camera",
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=19.3,
                focus_distance=400.0,
                horizontal_aperture=31.31,
                vertical_aperture=23.50,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=OffsetClass(
                pos=(-0.053, 0.0, 0.054),
                # rot=(-0.68301, 0.68301, 0.18301, -0.18301),
                rot=(-0.69636, 0.69636, 0.12279, -0.12279),
                convention="opengl",
            ),
        )

        # 相机3 — D405  (fx=394.858, fy=394.635, cx=315.12, cy=245.37)
        # horizontal_aperture = 19.3 / 394.858 * 640 = 31.29 mm
        # vertical_aperture   = 19.3 / 394.635 * 480 = 23.49 mm
        self.right_hand_camera = CameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/piper_R/hand_link_r/right_hand_camera",
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=19.3,
                focus_distance=400.0,
                horizontal_aperture=31.29,
                vertical_aperture=23.49,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=OffsetClass(
                pos=(-0.053, 0.0, 0.054),
                # rot=(-0.68301, 0.68301, 0.18301, -0.18301),
                rot = (-0.69636, 0.69636, 0.12279, -0.12279),
                convention="opengl",
            ),
        )
        # 腕部相机影响夹爪位置 
        # 测试腕部相机具体影响：
        # 0.0  右侧夹爪碰到物体
        # 0.01 默认值 左侧夹爪碰到物体
        # 0.009 夹爪左测会碰撞到物体
        # 0.005 左侧碰到物体
        # 0.002 左侧碰到物体
        # 0.001 基本在中间

        # 头相机 — D435I  (fx=608.366, fy=607.084, cx=312.87, cy=230.46)
        # horizontal_aperture = 19.3 / 608.366 * 640 = 20.31 mm
        # vertical_aperture   = 19.3 / 607.084 * 480 = 15.25 mm
        self.first_person_camera = CameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/piper_R/dummy_link/first_person_camera",
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=19.3,
                focus_distance=400.0,
                horizontal_aperture=20.31,
                vertical_aperture=15.25,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=OffsetClass(
                # pos=(-0.02, 0.3, 0.52),
                # rot=(0.28761, -0.28761, -0.64597, 0.64597),
                pos = (-0.02, 0.33, 0.8),
                # rot = (0.20083, -0.20083, -0.67799, 0.67799),
                rot = (0.17106, -0.17106, -0.6861, 0.6861),
                convention="opengl",
            ),
        )


# ---------------------------------------------------------------------------
# Action configurations
# ---------------------------------------------------------------------------


@configclass
class DoublePiperAbsoluteJointPositionActionsCfg:
    """Absolute joint-position actions for both arms and grippers."""

    left_arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1_l", "joint2_l", "joint3_l", "joint4_l", "joint5_l", "joint6_l"],
        preserve_order=True,
        use_default_offset=False,
    )
    right_arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1_r", "joint2_r", "joint3_r", "joint4_r", "joint5_r", "joint6_r"],
        preserve_order=True,
        use_default_offset=False,
    )
    left_gripper_action: ActionTermCfg = SymmetricGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint.*_l"],
        open_command_expr={"finger_joint_left_l": 0.035, "finger_joint_right_l": -0.035},
        close_command_expr={"finger_joint_left_l": -0.07, "finger_joint_right_l": 0.07},
        max_opening=0.035,
    )
    right_gripper_action: ActionTermCfg = SymmetricGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint.*_r"],
        open_command_expr={"finger_joint_left_r": 0.035, "finger_joint_right_r": -0.035},
        close_command_expr={"finger_joint_left_r": -0.07, "finger_joint_right_r": 0.07},
        max_opening=0.035,
    )


# ---------------------------------------------------------------------------
# Observation configuration
# ---------------------------------------------------------------------------


@configclass
class DoublePiperObservationsCfg:
    """Observation specifications for the Double Piper MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive observations exposed to the policy."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)

        left_joint_pos = ObsTerm(func=left_arm_joint_pos)
        right_joint_pos = ObsTerm(func=right_arm_joint_pos)
        left_gripper_pos = ObsTerm(func=left_gripper_pos)
        right_gripper_pos = ObsTerm(func=right_gripper_pos)
        left_eef_pos = ObsTerm(func=left_ee_pos)
        left_eef_quat = ObsTerm(func=left_ee_quat)
        right_eef_pos = ObsTerm(func=right_ee_pos)
        right_eef_quat = ObsTerm(func=right_ee_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Event (reset) configuration
# ---------------------------------------------------------------------------


@configclass
class DoublePiperEventCfg:
    """Reset events for the Double Piper robot."""

    reset_robot_joints = EventTerm(
        func=mdp_isaac_lab.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.01, 0.01),
            "velocity_range": (0.0, 0.0),
        },
    )


# ---------------------------------------------------------------------------
# Base embodiment class
# ---------------------------------------------------------------------------


class DoublePiperEmbodimentBase(EmbodimentBase):
    """Base class for all Double Piper embodiment variants.

    Subclasses must assign ``self.action_config`` to a concrete action cfg.
    """

    name = "double_piper"
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = DoublePiperSceneCfg()
        self.camera_config = DoublePiperCameraCfg()
        self.action_config = None  # must be set by subclass
        self.observation_config = DoublePiperObservationsCfg()
        self.event_config = DoublePiperEventCfg()
        self.reward_config = None
        self.mimic_env = None

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        if arm_mode == ArmMode.DUAL_ARM:
            return "ee_frame"
        return "ee_frame"


# ---------------------------------------------------------------------------
# Concrete registered embodiment variants
# ---------------------------------------------------------------------------


@register_asset
class DoublePiperAbsoluteJointPositionEmbodiment(DoublePiperEmbodimentBase):
    """Double Piper embodiment with absolute joint-position action control."""

    name = "double_piper_abs_joint_pos"
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.action_config = DoublePiperAbsoluteJointPositionActionsCfg()

