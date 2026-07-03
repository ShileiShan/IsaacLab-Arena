# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""IsaacTeleop pipeline for Double Piper XR teleoperation.

Output: 16D action vector
  [left_ee_pose(7), right_ee_pose(7), left_gripper(1), right_gripper(1)]

left_ee_pose / right_ee_pose: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
  Absolute SE3 target in robot root frame, consumed by DoublePiperDiffIKActionsCfg.
  Note: Isaac Lab's DifferentialInverseKinematicsActionCfg expects the quaternion
  in wxyz (w-first) order; Se3AbsRetargeter natively emits xyzw, so the reorderer
  below places quat_w ahead of quat_x/y/z in the output stream.

Rotation offsets (intrinsic XYZ Euler, degrees) align the OpenXR controller wrist
frame to the piper hand_link frame. Values mirror LW-BenchHub's
``lw_openxr_device.py`` ``_retarget_abs`` (left: 90 deg about Y; right: 180 deg
about (-x+z)/sqrt(2)) — the previous ARENA values produced a rotation matrix that
sent the gripper axis the wrong direction (manifested as "raise hand -> gripper
points down")."""


def _build_double_piper_teleop_pipeline():
    """Build an IsaacTeleop retargeting pipeline for Double Piper XR teleoperation.

    Returns:
        OutputCombiner with a single "action" output (16D flattened tensor).
    """
    from isaacteleop.retargeters import (
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
        TriHandMotionControllerConfig,
        TriHandMotionControllerRetargeter,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # -------------------------------------------------------------------------
    # SE3 Absolute Pose Retargeters
    # Rotation offsets align XR controller frame to piper hand_link frame.
    # Values match LW-BenchHub lw_openxr_device.py _retarget_abs():
    #   Left  : 90 deg about +Y         -> (roll=0,    pitch=90,  yaw=0)
    #   Right : 180 deg about (-x+z)/√2 -> (roll=-180, pitch=-90, yaw=0)
    # -------------------------------------------------------------------------
    left_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=45.0,      # 从 0 改成 45
        target_offset_pitch=180.0,    # 从 90 改成 180
        target_offset_yaw=-90.0,      # 从 0 改成 -90
        # target_offset_roll=0.0,
        # target_offset_pitch=90.0,
        # target_offset_yaw=0.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )

    right_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=-135.0,    # 从 -180 改成 -135
        target_offset_pitch=0.0,      # 从 -90 改成 0
        target_offset_yaw=90.0,       # 从 0 改成 90
        # target_offset_roll=-180.0,
        # target_offset_pitch=-90.0,
        # target_offset_yaw=0.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # -------------------------------------------------------------------------
    # TriHand Retargeters — use trigger scalar as gripper command (index 0)
    # -------------------------------------------------------------------------
    gripper_joint_names = [
        "gripper",
        "thumb_proximal",
        "thumb_distal",
        "index_proximal",
        "index_distal",
        "middle_proximal",
        "middle_distal",
    ]
    left_trihand_cfg = TriHandMotionControllerConfig(
        hand_joint_names=gripper_joint_names,
        controller_side="left",
    )
    left_trihand = TriHandMotionControllerRetargeter(left_trihand_cfg, name="trihand_left")
    connected_left_trihand = left_trihand.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )

    right_trihand_cfg = TriHandMotionControllerConfig(
        hand_joint_names=gripper_joint_names,
        controller_side="right",
    )
    right_trihand = TriHandMotionControllerRetargeter(right_trihand_cfg, name="trihand_right")
    connected_right_trihand = right_trihand.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # -------------------------------------------------------------------------
    # TensorReorderer: 16D output
    # [left_ee(7), right_ee(7), left_gripper(1), right_gripper(1)]
    #
    # Se3AbsRetargeter emits [pos(3), quat_xyzw(4)]; Isaac Lab DiffIK expects
    # [pos(3), quat_wxyz(4)]. We name the input elements in the retargeter's
    # native xyzw order (so the reorderer can index them correctly) and then
    # emit them in wxyz order via the output_order list.
    # -------------------------------------------------------------------------
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    left_ee_output = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_w", "l_quat_x", "l_quat_y", "l_quat_z"]
    right_ee_output = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_w", "r_quat_x", "r_quat_y", "r_quat_z"]
    left_hand_elements = [
        "l_gripper",
        "l_thumb_proximal",
        "l_thumb_distal",
        "l_index_proximal",
        "l_index_distal",
        "l_middle_proximal",
        "l_middle_distal",
    ]
    right_hand_elements = [
        "r_gripper",
        "r_thumb_proximal",
        "r_thumb_distal",
        "r_index_proximal",
        "r_index_distal",
        "r_middle_proximal",
        "r_middle_distal",
    ]

    output_order = left_ee_output + right_ee_output + ["l_gripper", "r_gripper"]

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_elements,
            "right_hand_joints": right_hand_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect({
        "left_ee_pose": connected_left_se3.output("ee_pose"),
        "right_ee_pose": connected_right_se3.output("ee_pose"),
        "left_hand_joints": connected_left_trihand.output("hand_joints"),
        "right_hand_joints": connected_right_trihand.output("hand_joints"),
    })

    return OutputCombiner({"action": connected_reorderer.output("output")})
