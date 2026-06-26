# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import sample_object_poses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_poses_and_align_auxiliary_assets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
    fixed_asset_cfg: SceneEntityCfg | None = None,
    auxiliary_asset_cfgs: list[SceneEntityCfg] | None = None,
    randomization_mode: Literal["held_and_fixed_only", "held_fixed_and_auxiliary"] = "held_and_fixed_only",
):
    """
    Randomize object poses and update the poses of related assets accordingly.

    Args:
        randomization_mode:
            - "held_and_fixed_only": Randomize only the fixed and held assets independently.
            - "held_fixed_and_auxiliary": Randomize fixed, held, and auxiliary assets, with auxiliary
              assets positioned relative to the fixed asset.
    """
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])

            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )

            if (
                randomization_mode == "held_fixed_and_auxiliary"
                and auxiliary_asset_cfgs is not None
                and fixed_asset_cfg is not None
                and asset_cfg.name == fixed_asset_cfg.name
            ):
                # Place auxiliary assets at exactly the same pose as the fixed asset (zero offset).
                # NOTE: This assumes the asset USD files have base frames defined such that zero offset creates a valid scene.
                # Currently designed for gear mesh task where all gears share the same center point.
                # For other assets, this may cause geometry intersections. Customers need to adjust it accordingly.
                for j in range(len(auxiliary_asset_cfgs)):
                    rel_asset_cfg = auxiliary_asset_cfgs[j]
                    rel_asset = env.scene[rel_asset_cfg.name]
                    rel_asset.write_root_pose_to_sim(
                        torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
                    )
                    rel_asset.write_root_velocity_to_sim(
                        torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
                    )


def enable_object_ccd(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    object_names: list[str],
) -> None:
    """Enable per-body CCD on the given rigid-body objects.

    Scene-level ``PhysxCfg.enable_ccd=True`` only activates the CCD broad-phase pass.
    Each dynamic rigid body also needs the ``physxRigidBody:enableCCD`` USD attribute
    set to ``true`` to actually participate in CCD sweeps.  Isaac Lab's
    ``RigidBodyPropertiesCfg`` does not expose this field, so we set it here via the
    USD stage after the scene has been built.

    Args:
        object_names: List of scene entity names (keys in ``env.scene``) whose rigid
            body prims should have CCD enabled.
    """
    del env_ids
    from pxr import UsdPhysics

    stage = env.sim.stage
    print(f"[enable_object_ccd] called for objects: {object_names}", flush=True)
    for name in object_names:
        if name not in env.scene.rigid_objects:
            print(f"[enable_object_ccd] WARNING: '{name}' not found in rigid_objects, skipping", flush=True)
            continue
        rigid_obj = env.scene.rigid_objects[name]
        enabled_count = 0
        for prim_path in rigid_obj.root_physx_view.prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            # Walk up to the first ancestor with PhysicsRigidBodyAPI applied.
            while prim.IsValid() and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim = prim.GetParent()
            if not prim.IsValid():
                continue
            from pxr import Sdf
            attr = prim.GetAttribute("physxRigidBody:enableCCD")
            if not attr.IsValid():
                attr = prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool, False)
            attr.Set(True)
            enabled_count += 1
        print(f"[enable_object_ccd] '{name}': CCD enabled on {enabled_count} prim(s)", flush=True)
