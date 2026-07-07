# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
import warp as wp
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


def wait_for_objects_to_settle(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    object_names: list[str],
    velocity_threshold: float = 0.005,
    max_steps: int = 300,
    min_steps: int = 40,
    consecutive_below_threshold: int = 10,
    position_stability_threshold: float = 5e-4,
    position_stability_window: int = 10,
) -> None:
    """Step physics until the listed rigid objects are at rest, or ``max_steps`` is reached.

    MimicGen snapshots each object's pose once per subtask and replays the recorded
    trajectory open-loop against that single snapshot. If an object is still falling
    or settling when the snapshot is taken (e.g. right after being placed into a bin),
    the replayed trajectory targets a stale pose. This event holds the scene idle and
    steps physics forward until every object has settled, so any caller of ``env.reset()``
    (teleop recording or MimicGen generation) only resumes once objects are at rest.

    Must run after any object-placement reset events. ``Task.get_events_cfg()`` is
    combined before the final ``placement_reset`` term in
    ``arena_env_builder.compose_manager_cfg``, so this event should be appended to
    ``env_cfg.events`` afterwards (e.g. via ``env_cfg_callback``) rather than returned
    from a task's ``get_events_cfg()``.

    Only settles when ``env_ids`` covers every environment: ``sim.step()`` advances
    physics for all environments at once, so partially resetting a subset while others
    are mid-episode would corrupt those other environments' state.
    """
    print(
        f"[settle] entry: env_ids={None if env_ids is None else env_ids.tolist()}, "
        f"num_envs={env.num_envs}, object_names={object_names}",
        flush=True,
    )
    if env_ids is None or len(env_ids) != env.num_envs:
        print("[settle] early-return: env_ids does not cover all envs", flush=True)
        return

    rigid_keys = list(env.scene.rigid_objects.keys())
    print(f"[settle] scene.rigid_objects keys={rigid_keys}", flush=True)
    objects = [env.scene.rigid_objects[name] for name in object_names if name in env.scene.rigid_objects]
    print(f"[settle] matched {len(objects)} / {len(object_names)} objects in rigid_objects", flush=True)
    if not objects:
        print("[settle] early-return: no matching rigid objects (settle skipped!)", flush=True)
        return

    # Robot joint pose before physics stepping (to detect arm sag during settle).
    robot_key = None
    for _k in ("robot", "piper", "left_piper", "right_piper"):
        if _k in env.scene.articulations:
            robot_key = _k
            break
    if robot_key is None and len(env.scene.articulations) > 0:
        robot_key = next(iter(env.scene.articulations.keys()))
    joint_pos_before = None
    if robot_key is not None:
        joint_pos_before = wp.to_torch(env.scene.articulations[robot_key].data.joint_pos)[0].clone()
        print(f"[settle] robot='{robot_key}' joint_pos_before={joint_pos_before.cpu().numpy()}", flush=True)

    # Object pose before physics stepping.
    obj_pos_before = [wp.to_torch(o.data.root_state_w)[0, 0:3].clone() for o in objects]
    for name, p in zip(object_names, obj_pos_before):
        print(f"[settle] obj '{name}' pos_before={p.cpu().numpy()}", flush=True)

    # ------------------------------------------------------------------
    # Lock every articulation's joint position target to its CURRENT position
    # before stepping physics. `_reset_idx` runs event terms BEFORE
    # `action_manager.reset()`, so the joint targets currently cached in each
    # articulation are stale (leftover from the previous episode's last teleop
    # step). If we don't overwrite them here, the PD controller will drag the
    # arm toward that stale target during the ~0.5s of settle physics -- which
    # is exactly why we saw the arm shift ~45deg and the grippers auto-close.
    # ------------------------------------------------------------------
    for art_name, art in env.scene.articulations.items():
        cur_joint_pos = wp.to_torch(art.data.joint_pos).clone()
        art.set_joint_position_target_index(target=cur_joint_pos, full_data=True)
        try:
            zero_vel = torch.zeros_like(cur_joint_pos)
            art.set_joint_velocity_target_index(target=zero_vel, full_data=True)
        except Exception:  # noqa: BLE001 - some articulations may not expose velocity targets
            pass
        print(f"[settle] locked joint targets for articulation '{art_name}'", flush=True)

    dt = env.sim.get_physics_dt()
    steps_taken = 0
    max_speed = float("inf")
    max_pos_change = float("inf")
    below_streak = 0
    # Ring buffer of recent per-object positions to detect creeping/drifting
    # objects that pass the velocity threshold momentarily but haven't truly
    # come to rest.
    pos_history: list[torch.Tensor] = []
    for step_idx in range(max_steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt)
        steps_taken += 1

        max_speed = 0.0
        cur_positions = []
        for obj in objects:
            lin_speed = torch.norm(wp.to_torch(obj.data.root_lin_vel_w), dim=-1).max().item()
            ang_speed = torch.norm(wp.to_torch(obj.data.root_ang_vel_w), dim=-1).max().item()
            max_speed = max(max_speed, lin_speed, ang_speed)
            cur_positions.append(wp.to_torch(obj.data.root_state_w)[0, 0:3].clone())
        stacked = torch.stack(cur_positions, dim=0)  # [num_obj, 3]
        pos_history.append(stacked)
        if len(pos_history) > position_stability_window:
            pos_history.pop(0)

        # Compute max position drift across the recent window.
        if len(pos_history) >= 2:
            oldest = pos_history[0]
            newest = pos_history[-1]
            max_pos_change = torch.norm(newest - oldest, dim=-1).max().item()
        else:
            max_pos_change = float("inf")

        # Require: (a) enough physics steps warmup, (b) velocity below threshold
        # for a streak, AND (c) position drift over the recent window is tiny.
        # A freshly placed object trivially passes (b) on step 1; a creeping
        # object passes (b) but fails (c). Only when all three hold do we
        # accept the scene as settled.
        if step_idx + 1 < min_steps:
            continue
        vel_ok = max_speed < velocity_threshold
        pos_ok = (
            len(pos_history) >= position_stability_window
            and max_pos_change < position_stability_threshold
        )
        if vel_ok and pos_ok:
            below_streak += 1
            if below_streak >= consecutive_below_threshold:
                break
        else:
            below_streak = 0
    else:
        print(
            f"[settle] WARNING: hit max_steps={max_steps} without settling. "
            f"Final max_speed={max_speed:.4f} (th={velocity_threshold}), "
            f"max_pos_change={max_pos_change:.6f} (th={position_stability_threshold})",
            flush=True,
        )

    # Force residual object velocity to zero so the recorded initial_state
    # replay-resets with a clean stationary object rather than injecting
    # sub-threshold motion at frame 0.
    for obj in objects:
        obj.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=env_ids,
        )

    print(
        f"[settle] done: steps_taken={steps_taken}/{max_steps}, "
        f"final max_speed={max_speed:.4f}, threshold={velocity_threshold}, "
        f"max_pos_change={max_pos_change:.6f} (th={position_stability_threshold})",
        flush=True,
    )

    # Object pose after settle -- did they actually move?
    for name, p_before, obj in zip(object_names, obj_pos_before, objects):
        p_after = wp.to_torch(obj.data.root_state_w)[0, 0:3]
        delta = (p_after - p_before).cpu().numpy()
        print(f"[settle] obj '{name}' pos_after={p_after.cpu().numpy()}, delta={delta}", flush=True)

    # Robot joint pose after settle -- did the arm sag?
    if robot_key is not None and joint_pos_before is not None:
        joint_pos_after = wp.to_torch(env.scene.articulations[robot_key].data.joint_pos)[0]
        joint_delta = (joint_pos_after - joint_pos_before).cpu().numpy()
        print(f"[settle] robot joint_pos_after={joint_pos_after.cpu().numpy()}", flush=True)
        print(f"[settle] robot joint_delta   ={joint_delta}", flush=True)
