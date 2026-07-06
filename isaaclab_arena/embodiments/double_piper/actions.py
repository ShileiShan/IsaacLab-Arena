# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 1
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Config for BinaryJointPositionZeroToOneAction.

    Uses 0/1 convention: value > 0.5 → close, value ≤ 0.5 → open.
    This is the correct convention for policies that output 0.0 (open) or 1.0 (close).
    """

    class_type: type = BinaryJointPositionZeroToOneAction


class SymmetricGripperPositionAction(BinaryJointPositionAction):
    """Continuous gripper control: single scalar → two symmetric finger joints.

    Input: g ∈ [0, max_opening], where 0 = fully closed, max_opening = fully open.
    Maps to: finger_left = g, finger_right = -g (mirrors open_command scaling).
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # actions: (N, 1), clamp to valid range
        g = actions.clamp(0.0, self.cfg.max_opening)  # (N, 1)
        # _open_command: (N, num_joints) = [+max_opening, -max_opening]
        # scale by g/max_opening to get [+g, -g]
        scale = g / self.cfg.max_opening  # (N, 1)
        self._processed_actions = scale * self._open_command
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class SymmetricGripperPositionActionCfg(BinaryJointPositionActionCfg):
    """Gripper action for policy evaluation.

    Policy outputs g ∈ [0, max_opening]: 0 = closed, max_opening = fully open.
    """

    class_type: type = SymmetricGripperPositionAction
    max_opening: float = 0.035


class XRGripperPositionAction(BinaryJointPositionAction):
    """Continuous gripper control for XR teleoperation.

    Input: g ∈ [0, max_opening], where 0 = fully open, max_opening = fully closed.
    Interpolates linearly from open_command (g=0) to close_command (g=max_opening).
    Convention matches VR trigger: no press → open, full press → closed.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        g = actions.clamp(0.0, self.cfg.max_opening)  # (N, 1)
        t = g / self.cfg.max_opening  # (N, 1)
        self._processed_actions = (1.0 - t) * self._open_command + t * self._close_command
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class XRGripperPositionActionCfg(BinaryJointPositionActionCfg):
    """Gripper action for XR teleoperation.

    Trigger input g ∈ [0, max_opening=1.0]: no press (g=0) → open; full press (g=1) → closed.
    """

    class_type: type = XRGripperPositionAction
    max_opening: float = 1.0


# ---------------------------------------------------------------------------
# Pinocchio DLS IK action term
# ---------------------------------------------------------------------------


class PiperArmIKAction(ActionTerm):
    """Action term that solves Piper arm IK via Pinocchio DLS.

    Accepts a 7D absolute EE pose [pos(3), quat_wxyz(4)] in the **robot root**
    frame (where the user wants ``hand_link_X`` to be) and outputs joint
    position targets for joints 1, 2, 3, 5, 6.  Joint4 is locked in the
    reduced model at ``cfg.locked_joint4_value`` — this MUST match the actual
    joint4 value in simulation, otherwise FK is wrong.

    Two optional frame transforms are composed before IK is solved:
    - ``arm_base_*``: pose of the arm's URDF base frame in robot-root coordinates
      (pinocchio's "world" is the URDF root; the single-arm URDF is reused for
      both left and right arms in a dual-arm scene).
    - ``ee_offset_*``: pose of ``hand_link`` relative to ``gripper_base`` (the
      URDF end-effector frame the IK targets).  Compensates for the difference
      between the URDF EE and the USD ``hand_link_X`` body the user is aiming at.

    Both default to identity, in which case the action behaves like a direct
    single-arm IK.
    """

    cfg: PiperArmIKActionCfg
    _asset: object  # Articulation

    def __init__(self, cfg: PiperArmIKActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        import pinocchio as pin

        from isaaclab_arena.embodiments.double_piper.piper_ik import PiperDLSIK

        self._ik = PiperDLSIK(
            urdf_path=cfg.urdf_path if cfg.urdf_path else None,
            package_dirs=cfg.package_dirs if cfg.package_dirs else None,
            locked_joint4_value=cfg.locked_joint4_value,
            lock_joint4=cfg.lock_joint4,
        )

        # Resolve the 5 controllable joint indices in the articulation
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        assert len(self._joint_ids) == self._ik.nq, (
            f"PiperArmIKAction: expected {self._ik.nq} joints, got {len(self._joint_ids)}"
        )

        # Pre-build SE3 transforms for frame composition
        def _se3(pos, quat_wxyz):
            w, x, y, z = quat_wxyz
            R = pin.Quaternion(float(w), float(x), float(y), float(z)).normalized().toRotationMatrix()
            return pin.SE3(R, np.array(pos, dtype=np.float64))

        self._T_root_armbase = _se3(cfg.arm_base_pos, cfg.arm_base_quat_wxyz)
        self._T_armbase_root = self._T_root_armbase.inverse()
        # T_grbase_handlink: hand_link expressed in gripper_base frame
        self._T_grbase_handlink = _se3(cfg.ee_offset_pos, cfg.ee_offset_quat_wxyz)
        self._T_handlink_grbase = self._T_grbase_handlink.inverse()

        # Rotation offset applied by Se3AbsRetargeter (target_offset_roll/pitch/yaw).
        # Stored so we can deconjugate it from the orientation delta; see process_actions.
        w, x, y, z = cfg.ee_rotation_offset_quat_wxyz
        self._R_offset = pin.Quaternion(float(w), float(x), float(y), float(z)).normalized().toRotationMatrix()

        # Axis-permutation matrix P such that root_axis[i] = retargeter_axis[perm[i]].
        # Used to remap the retargeter (anchor-frame) delta into the robot root
        # frame when XrCfg.anchor_rot swaps axes rather than being identity.
        # Applied to position delta as ``P · Δp`` and to rotation delta as the
        # similarity transform ``P · ΔR · Pᵀ``.
        perm = np.asarray(cfg.world_axis_permutation, dtype=np.int64)
        assert perm.shape == (3,) and set(perm.tolist()) == {0, 1, 2}, (
            f"world_axis_permutation must be a permutation of (0,1,2); got {tuple(cfg.world_axis_permutation)}"
        )
        self._axis_perm = perm
        self._P = np.zeros((3, 3), dtype=np.float64)
        for out_idx, in_idx in enumerate(perm):
            self._P[out_idx, in_idx] = 1.0

        self._raw_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._ik.nq, device=self.device)
        self._dbg_step: int = 0
        self._prev_raw: np.ndarray | None = None  # (num_envs, 7) for relative mode
        # Persistent IK reference (arm-base frame) for relative_mode.  Integrates
        # delta_pos / delta_rot from teleop instead of re-reading FK each tick,
        # so the target does not drift under gravity when input is idle.
        self._ik_ref_pos: np.ndarray | None = None  # (num_envs, 3)
        self._ik_ref_rot: np.ndarray | None = None  # (num_envs, 3, 3)
        # Init anchoring for absolute mode.  On the first call after a reset we
        # capture (raw_target, current_EE_in_root) and later frames output
        #   hand_root = init_ee_root + (raw - init_raw)
        # so the arm starts from wherever it happens to be and tracks operator
        # hand motion in root-frame world-delta terms.  This mirrors what G1's
        # pelvis-mounted anchor gives it via biology (arm length ~ human arm) —
        # for a table-top Piper the biology doesn't align, so we do it in code.
        self._init_raw_pos: np.ndarray | None = None  # (num_envs, 3)
        self._init_raw_rot: np.ndarray | None = None  # (num_envs, 3, 3)
        self._init_ee_pos: np.ndarray | None = None  # (num_envs, 3), in root frame
        self._init_ee_rot: np.ndarray | None = None  # (num_envs, 3, 3), in root frame
        self._init_captured: np.ndarray | None = None  # (num_envs,) bool

    # -- properties --------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # -- operations --------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        import pinocchio as pin
        import warp as wp

        self._raw_actions[:] = actions
        targets_np = actions.detach().cpu().numpy().astype(np.float64)  # (N, 7)
        jp = self._asset.data.joint_pos
        if not isinstance(jp, torch.Tensor):
            jp = wp.to_torch(jp)
        cur_q = jp[:, self._joint_ids].detach().cpu().numpy().astype(np.float64)

        N = targets_np.shape[0]
        ik_targets = np.empty((N, 7), dtype=np.float64)

        for i in range(N):
            pos = targets_np[i, :3]
            qw, qx, qy, qz = targets_np[i, 3:7]
            R = pin.Quaternion(float(qw), float(qx), float(qy), float(qz)).normalized().toRotationMatrix()
            T_root_hand = pin.SE3(R, pos)
            T_armbase_target = self._T_armbase_root * T_root_hand * self._T_handlink_grbase

            if self.cfg.relative_mode:
                # Current FK in arm-base frame (copy before solve() overwrites _data)
                pin.forwardKinematics(self._ik._model, self._ik._data, cur_q[i])
                pin.updateFramePlacements(self._ik._model, self._ik._data)
                FK_pos = self._ik._data.oMf[self._ik._frame_id].translation.copy()
                FK_rot = self._ik._data.oMf[self._ik._frame_id].rotation.copy()

                # FK diagnostic — diff tells us arm_base_pos error for future calibration
                if self._dbg_step % 30 == 0:
                    diff = T_armbase_target.translation - FK_pos
                    print(
                        f"[IK diag {self.cfg.joint_names[0][-1]}]"
                        f"  FK={FK_pos.round(4)}"
                        f"  abs_target={T_armbase_target.translation.round(4)}"
                        f"  diff(arm_base误差)={diff.round(4)}",
                        flush=True,
                    )

                if self._prev_raw is None:
                    self._prev_raw = targets_np.copy()
                    self._ik_ref_pos = np.zeros((N, 3), dtype=np.float64)
                    self._ik_ref_rot = np.tile(np.eye(3, dtype=np.float64), (N, 1, 1))
                    self._ik_ref_pos[i] = FK_pos
                    self._ik_ref_rot[i] = FK_rot
                    ik_pos = FK_pos
                    ik_rot = FK_rot
                    delta_pos_dbg = np.zeros(3)
                    rot_delta_mag_dbg = 0.0
                else:
                    # Lazy-init per-env reference on first visit (batch may grow)
                    if self._ik_ref_pos is None or self._ik_ref_pos.shape[0] != N:
                        self._ik_ref_pos = np.tile(FK_pos, (N, 1))
                        self._ik_ref_rot = np.tile(FK_rot, (N, 1, 1))

                    delta_pos = targets_np[i, :3] - self._prev_raw[i, :3]
                    self._ik_ref_pos[i] = self._ik_ref_pos[i] + delta_pos

                    # Leash: clamp reference to stay within max_track_err of FK
                    # so it can't run ahead of the arm and trip err_gate.
                    lead = self._ik_ref_pos[i] - FK_pos
                    lead_mag = float(np.linalg.norm(lead))
                    max_te = float(self.cfg.max_track_err)
                    if lead_mag > max_te and lead_mag > 1e-9:
                        self._ik_ref_pos[i] = FK_pos + lead * (max_te / lead_mag)

                    ik_pos = self._ik_ref_pos[i]
                    delta_pos_dbg = delta_pos

                    if self.cfg.track_rotation:
                        qw0, qx0, qy0, qz0 = self._prev_raw[i, 3:7]
                        R_prev = pin.Quaternion(
                            float(qw0), float(qx0), float(qy0), float(qz0)
                        ).normalized().toRotationMatrix()
                        delta_rot = R @ R_prev.T
                        # Slew-limit angle-axis magnitude to keep 6D IK error small
                        # (rotation contribution stays below err_gate_thresh).
                        aa = pin.log3(delta_rot)  # (3,) angle-axis
                        aa_mag = float(np.linalg.norm(aa))
                        max_rd = float(self.cfg.max_rot_delta_per_step)
                        if aa_mag > max_rd and aa_mag > 1e-9:
                            aa = aa * (max_rd / aa_mag)
                            delta_rot = pin.exp3(aa)
                            aa_mag = max_rd
                        self._ik_ref_rot[i] = delta_rot @ self._ik_ref_rot[i]

                        # Rotation leash: clamp ref rotation to stay within
                        # 2 * max_track_err (rad) of current FK rotation.
                        lead_R = self._ik_ref_rot[i] @ FK_rot.T
                        lead_aa = pin.log3(lead_R)
                        lead_aa_mag = float(np.linalg.norm(lead_aa))
                        max_te_rot = 2.0 * float(self.cfg.max_track_err)
                        if lead_aa_mag > max_te_rot and lead_aa_mag > 1e-9:
                            lead_aa = lead_aa * (max_te_rot / lead_aa_mag)
                            self._ik_ref_rot[i] = pin.exp3(lead_aa) @ FK_rot

                        ik_rot = self._ik_ref_rot[i]
                        rot_delta_mag_dbg = aa_mag
                    else:
                        # Freeze rotation reference at initial FK
                        ik_rot = self._ik_ref_rot[i]
                        rot_delta_mag_dbg = 0.0

                if self._dbg_step % 30 == 0:
                    track_err = float(np.linalg.norm(ik_pos - FK_pos))
                    print(
                        f"[IK diag {self.cfg.joint_names[0][-1]}]"
                        f"  |Δpos_raw|={float(np.linalg.norm(delta_pos_dbg)):.4f}m"
                        f"  |Δrot_raw|={rot_delta_mag_dbg:.4f}rad"
                        f"  track_err(ref-FK)={track_err:.4f}m",
                        flush=True,
                    )

                self._prev_raw[i] = targets_np[i]
                q_pin = pin.Quaternion(ik_rot)
                ik_targets[i, :3] = ik_pos
                ik_targets[i, 3:7] = (q_pin.w, q_pin.x, q_pin.y, q_pin.z)

            else:
                # Absolute mode with init-anchor.  On the very first frame after
                # a reset we capture (raw, EE_in_root) as the anchor pair; every
                # subsequent frame outputs
                #     hand_root = init_ee_root + (raw - init_raw)
                # and applies the raw-frame rotation delta on top of init_ee_rot.
                # No per-frame accumulation → no drift.  The init pair simply
                # translates the operator's VR-world so the current controller
                # position corresponds to the current EE position, so IK targets
                # always start inside the arm's reachable workspace.
                pin.forwardKinematics(self._ik._model, self._ik._data, cur_q[i])
                pin.updateFramePlacements(self._ik._model, self._ik._data)
                T_armbase_grbase_fk = self._ik._data.oMf[self._ik._frame_id].copy()
                T_root_hand_fk = (
                    self._T_root_armbase * T_armbase_grbase_fk * self._T_grbase_handlink
                )

                if self._init_captured is None:
                    self._init_raw_pos = np.zeros((N, 3), dtype=np.float64)
                    self._init_raw_rot = np.tile(np.eye(3, dtype=np.float64), (N, 1, 1))
                    self._init_ee_pos = np.zeros((N, 3), dtype=np.float64)
                    self._init_ee_rot = np.tile(np.eye(3, dtype=np.float64), (N, 1, 1))
                    self._init_captured = np.zeros(N, dtype=bool)

                if not self._init_captured[i]:
                    self._init_raw_pos[i] = pos
                    self._init_raw_rot[i] = R
                    self._init_ee_pos[i] = T_root_hand_fk.translation
                    self._init_ee_rot[i] = T_root_hand_fk.rotation
                    self._init_captured[i] = True
                    hand_pos_root = T_root_hand_fk.translation
                    hand_rot_root = T_root_hand_fk.rotation
                    T_armbase_grbase_fk = self._ik._data.oMf[self._ik._frame_id].copy()
                    print(
                        f"[IK anchor {self.cfg.joint_names[0][-1]}] init"
                        f"  cur_q={cur_q[i].round(4)}"
                        f"  FK_in_armbase={T_armbase_grbase_fk.translation.round(4)}"
                        f"  arm_base_pos={np.array(self.cfg.arm_base_pos).round(4)}"
                        f"  init_raw={pos.round(4)}"
                        f"  init_ee_root={hand_pos_root.round(4)}",
                        flush=True,
                    )
                else:
                    signs = np.asarray(self.cfg.world_delta_signs, dtype=np.float64)
                    # Position: reorder retargeter-frame delta into root-frame axes,
                    # then apply per-axis sign flip.
                    delta_pos_r = pos - self._init_raw_pos[i]
                    delta_pos = delta_pos_r[self._axis_perm] * signs

                    # Rotation: same axis permutation applied as similarity transform
                    # so ΔR expressed in root axes rotates the arm consistently.
                    delta_rot_r = R @ self._init_raw_rot[i].T
                    delta_rot_raw = self._P @ delta_rot_r @ self._P.T
                    rot_signs = np.asarray(self.cfg.world_rot_delta_signs, dtype=np.float64)
                    aa = pin.log3(delta_rot_raw)
                    aa_flipped = aa * rot_signs
                    delta_rot = pin.exp3(aa_flipped)
                    hand_pos_root = self._init_ee_pos[i] + delta_pos
                    hand_rot_root = delta_rot @ self._init_ee_rot[i]
                    if self._dbg_step % 30 == 0:
                        aa_r = pin.log3(delta_rot_r)
                        aa_r_mag = float(np.linalg.norm(aa_r))
                        aa_mag = float(np.linalg.norm(aa))
                        aa_flipped_mag = float(np.linalg.norm(aa_flipped))
                        print(
                            f"[IK rot {self.cfg.joint_names[0][-1]}]"
                            f" rot_pre_perm={aa_r.round(4)} (mag={aa_r_mag:.4f})"
                            f" rot_delta_raw={aa.round(4)} (mag={aa_mag:.4f})"
                            f" rot_delta_flipped={aa_flipped.round(4)} (mag={aa_flipped_mag:.4f})",
                            flush=True,
                        )
                        print(
                            f"[IK pos {self.cfg.joint_names[0][-1]}]"
                            f" pos_pre_perm={delta_pos_r.round(4)}"
                            f" pos_delta={delta_pos.round(4)}",
                            flush=True,
                        )

                T_root_hand_anchored = pin.SE3(hand_rot_root, hand_pos_root)
                T_armbase_target_anchored = (
                    self._T_armbase_root * T_root_hand_anchored * self._T_handlink_grbase
                )
                ik_targets[i, :3] = T_armbase_target_anchored.translation
                q_out = pin.Quaternion(T_armbase_target_anchored.rotation)
                ik_targets[i, 3:7] = (q_out.w, q_out.x, q_out.y, q_out.z)

        if not self.cfg.relative_mode and self._dbg_step % 30 == 0:
            ik_q_out = pin.Quaternion(ik_targets[0,3], ik_targets[0,4], ik_targets[0,5], ik_targets[0,6])
            print(
                f"[{self.cfg.joint_names[0]}] step={self._dbg_step}"
                f" raw_pos={targets_np[0,:3].round(4)} raw_quat_wxyz={targets_np[0,3:7].round(4)}"
                f" ik_pos={ik_targets[0,:3].round(4)} ik_quat_wxyz={ik_targets[0,3:7].round(4)}",
                flush=True,
            )

        q_np = self._ik.solve(ik_targets, warm_q=cur_q)

        if self._dbg_step % 30 == 0:
            print(
                f"[{self.cfg.joint_names[0]}] |Δq|={np.linalg.norm(q_np[0]-cur_q[0]):.4f}",
                flush=True,
            )
        self._dbg_step += 1

        self._processed_actions = torch.tensor(q_np, dtype=torch.float32, device=self.device)

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids_np = np.array(env_ids, dtype=np.intp) if env_ids is not None else None
        self._ik.reset(ids_np)
        self._prev_raw = None  # reinitialize on next call regardless of which envs reset
        self._ik_ref_pos = None
        self._ik_ref_rot = None
        # Clear init-anchor cache so absolute mode re-captures (raw, EE) on the
        # next call.  A partial reset (env_ids != None) invalidates only those
        # envs, but with num_envs == 1 for teleop this is equivalent to full clear.
        if self._init_captured is not None:
            if env_ids is None:
                self._init_captured[:] = False
            else:
                self._init_captured[ids_np] = False


@configclass
class PiperArmIKActionCfg(ActionTermCfg):
    """Config for PiperArmIKAction.

    joint_names must list the 5 controllable arm joints in order:
    [joint1_X, joint2_X, joint3_X, joint5_X, joint6_X].
    """

    class_type: type = PiperArmIKAction

    joint_names: list[str] = MISSING
    urdf_path: str | None = None
    package_dirs: list[str] | None = None

    # Whether to lock joint4 (wrist yaw) in the pinocchio reduced model.
    # Default False → 6-DOF IK using all wrist joints, arbitrary orientation
    # reachable within joint limits.  Set True to fall back to LW-BenchHub's
    # 5-DOF variant, in which case ``locked_joint4_value`` must match sim.
    lock_joint4: bool = False

    # Only used when ``lock_joint4=True``: the joint4 value the reduced model
    # assumes is held constant.  MUST match the actual joint4 position in the
    # USD/sim, otherwise FK is wrong and the gripper tracks a shifted pose.
    locked_joint4_value: float = 0.0

    # Pose of the arm's URDF base frame in robot-root coordinates.  Default
    # identity assumes the URDF root coincides with the robot root prim.
    arm_base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_base_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Pose of USD ``hand_link_X`` relative to URDF ``gripper_base``.  Default
    # identity assumes they coincide.  Measure once and override if not.
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_offset_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # When True, the action uses delta-pose control: each step applies the
    # difference between the current and previous target to the current FK pose.
    # This bypasses arm_base_pos calibration and lets the arm follow controller
    # movements without needing precise frame alignment.
    relative_mode: bool = False

    # In relative_mode, whether to also track rotation deltas.  When False, the
    # gripper orientation is frozen at whatever the initial FK gives (position-only
    # tracking).  When True, delta rotations are applied on top of FK_rot with a
    # per-step slew limit to keep IK error small enough to avoid err_gate trips.
    track_rotation: bool = True

    # Max rotation delta (rad) per action tick when track_rotation=True.  Clips
    # angle-axis magnitude of delta_rot; ~0.15 rad ≈ 9° is a safe default at 60 Hz
    # (max 540 deg/s wrist speed) that keeps rotation contribution to IK 6D error
    # well below err_gate_thresh.
    max_rot_delta_per_step: float = 0.15

    # Leash length: max distance (m) the integrated IK reference is allowed to
    # sit ahead of the current FK.  When user moves faster than the arm can
    # follow, the reference is clamped along the FK→ref direction so the arm
    # stays within reach and err_gate does not trip.  Should be strictly less
    # than PiperDLSIK.err_gate_thresh (currently 0.6 m) with margin for rotation.
    max_track_err: float = 0.15

    # Sign flips applied to the anchor-frame delta before applying it to the
    # initial EE pose (absolute mode with init anchor).  Use to correct axis
    # inversions between the VR anchor frame and the robot root frame — e.g.
    # if hand-left maps to arm-right, flip the corresponding component to -1.
    # Applied to position delta only.  Default identity: no flip.
    world_delta_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Rotation offset quaternion (wxyz) applied by Se3AbsRetargeter
    # (target_offset_roll/pitch/yaw).  Must match the rotation configured in
    # teleop_pipeline.py for this arm.  Used to deconjugate the retargeter
    # output so rotation tracking is in the raw controller world frame.
    # Default identity: no offset (or offset already accounted for elsewhere).
    ee_rotation_offset_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Sign flips applied to the rotation delta after deconjugating the
    # retargeter offset.  Tune independently from world_delta_signs.
    # Default identity: no flip.
    world_rot_delta_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Axis permutation between the retargeter output frame (anchor-frame) and
    # the robot root frame.  Represents ``root_axis[i] = retargeter_axis[perm[i]]``.
    # Applied to both the position delta (as index reordering) and to the
    # rotation delta (as a similarity transform ``P · ΔR · Pᵀ``).
    #
    # Needed when ``XrCfg.anchor_rot`` swaps axes rather than being identity —
    # e.g. double_piper's ``(0.5,-0.5,-0.5,0.5)`` (a 120° cyclic rotation)
    # produces a retargeter output where user hand X↔Z is swapped relative to
    # the piper root frame.  Set to ``(2, 1, 0)`` there to swap X↔Z back.
    #
    # Default identity ``(0, 1, 2)`` = no permutation, backward compatible.
    world_axis_permutation: tuple[int, int, int] = (0, 1, 2)
