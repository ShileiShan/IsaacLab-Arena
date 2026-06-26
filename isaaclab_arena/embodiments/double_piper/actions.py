# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass


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
    """Config for SymmetricGripperPositionAction.

    max_opening: fully-open joint position (metres). Default matches Piper gripper (0.035 m).
    Policy outputs g ∈ [0, max_opening]: 0 = closed, max_opening = fully open.
    """

    class_type: type = SymmetricGripperPositionAction
    max_opening: float = 0.035
