# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pxr import PhysxSchema, Usd


def add_contact_report(prim: Usd.Prim) -> None:
    """Add a contact report API to a prim.

    Args:
        prim: The prim to add the contact report API to.
    """
    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    cr_api.CreateThresholdAttr().Set(0)


def disable_kinematic_rigid_object_velocity_writes(env) -> None:
    """No-op ``write_root_velocity_to_sim*`` for kinematic rigid objects in the scene.

    ``InteractiveScene.reset_to`` (used by dataset replay, e.g.
    ``isaaclab_arena/scripts/imitation_learning/replay_demos.py``) unconditionally
    writes a zero velocity to every rigid object in the recorded state, regardless of
    whether it is kinematic. PhysX rejects ``setLinearVelocity``/``setAngularVelocity``
    on kinematic bodies with "Body must be non-kinematic!" since their pose is driven
    externally rather than by the solver. This is patched here (per-instance, on the
    Arena side) rather than in the vendored ``submodules/IsaacLab`` scene/asset code.

    Args:
        env: The (unwrapped) ``ManagerBasedEnv`` whose scene should be patched.
    """
    for rigid_object in env.scene.rigid_objects.values():
        rigid_props = getattr(getattr(rigid_object.cfg, "spawn", None), "rigid_props", None)
        if getattr(rigid_props, "kinematic_enabled", False):
            rigid_object.write_root_velocity_to_sim_index = lambda *args, **kwargs: None
            rigid_object.write_root_velocity_to_sim = lambda *args, **kwargs: None
