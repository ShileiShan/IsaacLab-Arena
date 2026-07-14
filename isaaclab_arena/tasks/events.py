# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import datetime
import os
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


def _root_prim_paths_for_env_ids(env: "ManagerBasedEnv", object_name: str, env_ids: torch.Tensor) -> list[str]:
    rigid_obj = env.scene.rigid_objects[object_name]
    prim_paths = list(rigid_obj.root_physx_view.prim_paths)
    if len(prim_paths) == env.num_envs:
        return [prim_paths[int(env_id)] for env_id in env_ids.tolist()]
    return prim_paths


def _set_object_collision_enabled(
    env: "ManagerBasedEnv",
    object_name: str,
    env_ids: torch.Tensor,
    enabled: bool,
) -> None:
    from pxr import Usd, UsdPhysics

    stage = env.sim.stage
    changed_count = 0
    for prim_path in _root_prim_paths_for_env_ids(env, object_name, env_ids):
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision_api = UsdPhysics.CollisionAPI(prim)
            attr = collision_api.GetCollisionEnabledAttr()
            if not attr.IsValid():
                attr = collision_api.CreateCollisionEnabledAttr()
            attr.Set(enabled)
            changed_count += 1
    print(f"[drop] collision {'enabled' if enabled else 'disabled'} for '{object_name}' ({changed_count} prims)")


def _find_rigid_body_prim(root_prim):
    from pxr import Usd, UsdPhysics

    prim = root_prim
    while prim.IsValid():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
        parent = prim.GetParent()
        if not parent.IsValid():
            break
        prim = parent

    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    return None


def _set_object_gravity_enabled(
    env: "ManagerBasedEnv",
    object_name: str,
    env_ids: torch.Tensor,
    enabled: bool,
) -> None:
    from pxr import Sdf

    stage = env.sim.stage
    changed_count = 0
    for prim_path in _root_prim_paths_for_env_ids(env, object_name, env_ids):
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            continue
        rigid_body_prim = _find_rigid_body_prim(root_prim)
        if rigid_body_prim is None:
            continue
        attr = rigid_body_prim.GetAttribute("physxRigidBody:disableGravity")
        if not attr.IsValid():
            attr = rigid_body_prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, False)
        attr.Set(not enabled)
        changed_count += 1
    print(f"[drop] gravity {'enabled' if enabled else 'disabled'} for '{object_name}' ({changed_count} bodies)")


def _lock_articulations_at_current_pose(env: "ManagerBasedEnv") -> None:
    for art_name, art in env.scene.articulations.items():
        cur_joint_pos = wp.to_torch(art.data.joint_pos).clone()
        art.set_joint_position_target_index(target=cur_joint_pos)
        try:
            zero_vel = torch.zeros_like(cur_joint_pos)
            art.set_joint_velocity_target_index(target=zero_vel)
        except Exception:  # noqa: BLE001 - some articulations may not expose velocity targets
            pass
        print(f"[drop] locked joint targets for articulation '{art_name}'", flush=True)


def _get_object_bbox_min_max(env: "ManagerBasedEnv", object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    arena_env = getattr(env.cfg, "isaaclab_arena_env", None)
    if arena_env is not None:
        asset = arena_env.scene.assets.get(object_name)
        if asset is not None and hasattr(asset, "get_bounding_box"):
            bbox = asset.get_bounding_box().to(env.device)
            return bbox.min_point[0], bbox.max_point[0]

    # Conservative fallback if the Arena asset metadata is unavailable.
    extent = torch.tensor([0.08, 0.08, 0.08], device=env.device)
    return -0.5 * extent, 0.5 * extent


def _bbox_xy_radius(bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    xy_corners = torch.stack(
        (
            torch.stack((bbox_min[0], bbox_min[1])),
            torch.stack((bbox_min[0], bbox_max[1])),
            torch.stack((bbox_max[0], bbox_min[1])),
            torch.stack((bbox_max[0], bbox_max[1])),
        ),
        dim=0,
    )
    return torch.linalg.norm(xy_corners, dim=-1).max()


def _sample_drop_xy_away_from_released(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    drop_min: torch.Tensor,
    drop_max: torch.Tensor,
    current_radius: torch.Tensor,
    released_names: list[str],
    bbox_by_name: dict[str, tuple[torch.Tensor, torch.Tensor]],
    object_name: str,
    max_attempts: int = 96,
) -> torch.Tensor:
    num_envs = len(env_ids)
    safe_min = drop_min[:, 0:2] + current_radius
    safe_max = drop_max[:, 0:2] - current_radius

    if torch.any(safe_max <= safe_min):
        print(
            f"[drop] WARNING: drop area too small to keep full bbox inside for '{object_name}' "
            f"(xy_radius={float(current_radius.item()):.3f}); sampling root in the configured area",
            flush=True,
        )
        safe_min = drop_min[:, 0:2]
        safe_max = drop_max[:, 0:2]

    env_origins_xy = env.scene.env_origins[env_ids, 0:2]
    released_centers: list[torch.Tensor] = []
    released_radii: list[torch.Tensor] = []
    for released_name in released_names:
        released_obj = env.scene.rigid_objects[released_name]
        released_state = wp.to_torch(released_obj.data.root_state_w)[env_ids]
        released_centers.append(released_state[:, 0:2] - env_origins_xy)
        released_radii.append(_bbox_xy_radius(*bbox_by_name[released_name]))

    def _sample() -> torch.Tensor:
        return torch.rand(num_envs, 2, device=env.device) * (safe_max - safe_min) + safe_min

    if not released_centers:
        return _sample()

    best_xy = _sample()
    best_clearance = torch.full((num_envs,), -1.0e9, device=env.device)
    min_gap = torch.tensor(0.01, device=env.device)

    for _ in range(max_attempts):
        candidate = _sample()
        candidate_clearance = torch.full((num_envs,), 1.0e9, device=env.device)
        for center, radius in zip(released_centers, released_radii):
            clearance = torch.linalg.norm(candidate - center, dim=-1) - (current_radius + radius + min_gap)
            candidate_clearance = torch.minimum(candidate_clearance, clearance)
        better = candidate_clearance > best_clearance
        best_xy[better] = candidate[better]
        best_clearance[better] = candidate_clearance[better]
        if bool(torch.all(candidate_clearance > 0.0).item()):
            return candidate

    print(
        f"[drop] WARNING: no non-overlapping XY sample found for '{object_name}', "
        f"using best clearance {float(best_clearance.min().item()):.3f} m",
        flush=True,
    )
    return best_xy


def drop_objects_into_pick_bin(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    object_names: list[str],
    pick_box_name: str,
    drop_min_xyz: tuple[float, float, float],
    drop_max_xyz: tuple[float, float, float],
    pick_box_min_xyz: tuple[float, float, float],
    pick_box_max_xyz: tuple[float, float, float],
    yaw_half_rad: float = 3.14159,
    drop_interval_s: float = 1.0,
    collision_sync_steps: int = 0,
    final_settle_velocity_threshold: float = 0.05,
    final_settle_max_steps: int = 300,
    debug_dir: str | None = None,
) -> None:
    """Sequentially drop objects from a randomized volume above the pick bin.

    Collisions stay enabled for all target objects. The event first moves all
    objects to a distant non-overlapping holding area with gravity disabled,
    then teleports one object at a time above the pick bin and enables gravity.
    This avoids live collider toggling and prevents unreleased objects from
    overlapping each other near the bin.

    Like ``wait_for_objects_to_settle``, this event only runs for full-environment
    resets because ``sim.step()`` advances every environment together.
    """
    print(
        f"[drop] entry: env_ids={None if env_ids is None else env_ids.tolist()}, "
        f"num_envs={env.num_envs}, object_names={object_names}, pick_box_name={pick_box_name}",
        flush=True,
    )
    # TODO: sequential drop currently only supports full-environment resets.
    # Partial resets would advance the shared simulation and can corrupt other envs.
    if env_ids is None or len(env_ids) != env.num_envs:
        print("[drop] early-return: env_ids does not cover all envs", flush=True)
        return
    if pick_box_name not in env.scene.rigid_objects:
        print(f"[drop] early-return: pick box '{pick_box_name}' not found in rigid_objects", flush=True)
        return

    matched_names = [name for name in object_names if name in env.scene.rigid_objects]
    print(f"[drop] matched {len(matched_names)} / {len(object_names)} objects in rigid_objects", flush=True)
    if not matched_names:
        print("[drop] early-return: no matching rigid objects", flush=True)
        return

    debug_run_dir = None
    if debug_dir:
        debug_run_dir = os.path.join(debug_dir, datetime.datetime.now().strftime("reset_%Y%m%d_%H%M%S_%f"))
        os.makedirs(debug_run_dir, exist_ok=True)
        print(f"[drop] debug images dir: {debug_run_dir}", flush=True)

    _lock_articulations_at_current_pose(env)

    dt = env.sim.get_physics_dt()
    num_envs = len(env_ids)
    zero_velocity = torch.zeros(num_envs, 6, device=env.device)
    drop_min = torch.tensor(drop_min_xyz, device=env.device).unsqueeze(0)
    drop_max = torch.tensor(drop_max_xyz, device=env.device).unsqueeze(0)
    bbox_by_name = {name: _get_object_bbox_min_max(env, name) for name in matched_names}

    holding_origin = torch.tensor(
        [pick_box_min_xyz[0] - 2.0, pick_box_min_xyz[1] - 1.0, pick_box_max_xyz[2] + 0.50],
        device=env.device,
    )
    holding_spacing = max(
        0.35,
        max(float((bbox_max - bbox_min)[0:2].max().item()) for bbox_min, bbox_max in bbox_by_name.values()) + 0.20,
    )
    for index, name in enumerate(matched_names):
        _set_object_gravity_enabled(env, name, env_ids, False)
        bbox_min, _ = bbox_by_name[name]
        holding_position = holding_origin + torch.tensor(
            [0.0, index * holding_spacing, -float(bbox_min[2].item())],
            device=env.device,
        )
        positions = holding_position.unsqueeze(0).repeat(num_envs, 1) + env.scene.env_origins[env_ids]
        quat_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(num_envs, 1)
        pose = torch.cat([positions, quat_xyzw], dim=-1)
        obj = env.scene.rigid_objects[name]
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
        print(f"[drop] staged '{name}' in holding area", flush=True)

    env.scene.write_data_to_sim()
    _save_drop_debug_camera_images(
        env,
        env_ids,
        matched_names,
        pick_box_min_xyz,
        pick_box_max_xyz,
        drop_min_xyz,
        drop_max_xyz,
        debug_run_dir,
        "000_staged",
    )
    drop_steps = max(1, int(round(drop_interval_s / dt)))
    drop_order = torch.randperm(len(matched_names), device=env.device).tolist()
    print(
        f"[drop] releasing objects in order {[matched_names[i] for i in drop_order]} "
        f"for {drop_steps} steps each ({drop_interval_s:.3f}s)",
        flush=True,
    )
    released_names: list[str] = []
    for release_idx, object_index in enumerate(drop_order, start=1):
        name = matched_names[object_index]
        obj = env.scene.rigid_objects[name]
        bbox_min, bbox_max = bbox_by_name[name]
        root_xy_radius = _bbox_xy_radius(bbox_min, bbox_max)
        xy = _sample_drop_xy_away_from_released(
            env,
            env_ids,
            drop_min,
            drop_max,
            root_xy_radius,
            released_names,
            bbox_by_name,
            name,
        )
        z_root_min = drop_min[:, 2] - bbox_min[2]
        z_root_max = drop_max[:, 2] - bbox_min[2]
        z = torch.rand(num_envs, 1, device=env.device) * (z_root_max - z_root_min) + z_root_min
        positions = torch.cat([xy, z], dim=-1) + env.scene.env_origins[env_ids]
        yaw = torch.rand(num_envs, device=env.device) * (2.0 * yaw_half_rad) - yaw_half_rad
        zeros = torch.zeros_like(yaw)
        quat_xyzw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        pose = torch.cat([positions, quat_xyzw], dim=-1)
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
        for _ in range(max(0, collision_sync_steps)):
            obj.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt)
        _set_object_gravity_enabled(env, name, env_ids, True)
        for _ in range(drop_steps):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt)
        released_names.append(name)
        _save_drop_debug_camera_images(
            env,
            env_ids,
            matched_names,
            pick_box_min_xyz,
            pick_box_max_xyz,
            drop_min_xyz,
            drop_max_xyz,
            debug_run_dir,
            f"{release_idx:03d}_after_release_{name}",
        )

    for name in matched_names:
        _set_object_gravity_enabled(env, name, env_ids, True)

    wait_for_objects_to_settle(
        env,
        env_ids,
        matched_names,
        velocity_threshold=final_settle_velocity_threshold,
        max_steps=final_settle_max_steps,
    )
    _save_drop_debug_camera_images(
        env,
        env_ids,
        matched_names,
        pick_box_min_xyz,
        pick_box_max_xyz,
        drop_min_xyz,
        drop_max_xyz,
        debug_run_dir,
        "999_final_settled",
    )


def _save_drop_debug_camera_images(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    object_names: list[str],
    pick_box_min_xyz: tuple[float, float, float],
    pick_box_max_xyz: tuple[float, float, float],
    drop_min_xyz: tuple[float, float, float],
    drop_max_xyz: tuple[float, float, float],
    debug_run_dir: str | None,
    frame_name: str,
) -> None:
    if debug_run_dir is None:
        return
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("[drop] PIL/numpy is not installed; skipping camera debug image output", flush=True)
        return

    try:
        env.sim.render()
        camera_obs = env.observation_manager.compute_group("camera_obs", update_history=False)
    except Exception as exc:  # noqa: BLE001 - debug capture must not break reset
        print(f"[drop] camera debug capture failed ({exc}); writing top-down fallback", flush=True)
        _save_drop_debug_topdown(
            env,
            env_ids,
            object_names,
            pick_box_min_xyz,
            pick_box_max_xyz,
            drop_min_xyz,
            drop_max_xyz,
            debug_run_dir,
            frame_name,
        )
        return

    if not isinstance(camera_obs, dict) or not camera_obs:
        print("[drop] camera_obs is empty; writing top-down fallback", flush=True)
        _save_drop_debug_topdown(
            env,
            env_ids,
            object_names,
            pick_box_min_xyz,
            pick_box_max_xyz,
            drop_min_xyz,
            drop_max_xyz,
            debug_run_dir,
            frame_name,
        )
        return

    env_index = int(env_ids[0].item())
    top_camera_names = [
        name
        for name in camera_obs
        if str(name) == "first_person_camera_rgb" or "top" in str(name).lower()
    ]
    if not top_camera_names:
        print(
            f"[drop] no top camera found in camera_obs keys={list(camera_obs.keys())}; skipping camera debug image",
            flush=True,
        )
        return

    for camera_name in top_camera_names:
        tensor = camera_obs[camera_name]
        img = tensor[env_index]
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = img.transpose(1, 2, 0)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            scale = 255.0 if img.dtype.kind == "f" and float(img.max()) <= 1.0 else 1.0
            img = np.clip(img * scale, 0, 255).astype(np.uint8)
        safe_camera_name = str(camera_name).replace("/", "_").replace(os.sep, "_")
        Image.fromarray(img).save(os.path.join(debug_run_dir, f"camera_{frame_name}_{safe_camera_name}.png"))


def _save_drop_debug_topdown(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    object_names: list[str],
    pick_box_min_xyz: tuple[float, float, float],
    pick_box_max_xyz: tuple[float, float, float],
    drop_min_xyz: tuple[float, float, float],
    drop_max_xyz: tuple[float, float, float],
    debug_run_dir: str | None,
    frame_name: str,
) -> None:
    if debug_run_dir is None:
        return
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[drop] PIL is not installed; skipping debug image output", flush=True)
        return

    env_id = int(env_ids[0].item())
    origin = env.scene.env_origins[env_id].detach().cpu()
    pick_min = torch.tensor(pick_box_min_xyz) + origin
    pick_max = torch.tensor(pick_box_max_xyz) + origin
    drop_min = torch.tensor(drop_min_xyz) + origin
    drop_max = torch.tensor(drop_max_xyz) + origin

    object_positions = []
    for name in object_names:
        if name not in env.scene.rigid_objects:
            continue
        pos = wp.to_torch(env.scene.rigid_objects[name].data.root_state_w)[env_id, 0:3].detach().cpu()
        object_positions.append((name, pos))

    points = [pick_min, pick_max, drop_min, drop_max] + [pos for _, pos in object_positions]
    min_x = min(float(p[0]) for p in points) - 0.10
    max_x = max(float(p[0]) for p in points) + 0.10
    min_y = min(float(p[1]) for p in points) - 0.10
    max_y = max(float(p[1]) for p in points) + 0.10

    width, height = 900, 700
    pad = 60

    def project(p: torch.Tensor | tuple[float, float]) -> tuple[int, int]:
        x = float(p[0])
        y = float(p[1])
        px = pad + (x - min_x) / max(max_x - min_x, 1e-6) * (width - 2 * pad)
        py = height - pad - (y - min_y) / max(max_y - min_y, 1e-6) * (height - 2 * pad)
        return int(px), int(py)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def rect(min_pt: torch.Tensor, max_pt: torch.Tensor, color: tuple[int, int, int], label: str) -> None:
        x0, y0 = project((float(min_pt[0]), float(min_pt[1])))
        x1, y1 = project((float(max_pt[0]), float(max_pt[1])))
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left + 4, top + 4), label, fill=color)

    rect(pick_min, pick_max, (20, 90, 220), "pick box bbox")
    rect(drop_min, drop_max, (220, 90, 20), "drop sampling area")

    for idx, (name, pos) in enumerate(object_positions):
        x, y = project(pos)
        color = (20, 140, 60) if _point_inside_xy(pos, pick_min, pick_max) else (220, 30, 30)
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=color, outline=(0, 0, 0))
        draw.text((x + 10, y - 8), f"{idx}:{name} z={float(pos[2]):.3f}", fill=color)

    draw.text(
        (pad, height - pad + 20),
        "green = inside pick-box XY, red = outside pick-box XY",
        fill=(30, 30, 30),
    )
    image.save(os.path.join(debug_run_dir, f"topdown_{frame_name}.png"))


def _point_inside_xy(point: torch.Tensor, min_point: torch.Tensor, max_point: torch.Tensor) -> bool:
    return (
        float(min_point[0]) <= float(point[0]) <= float(max_point[0])
        and float(min_point[1]) <= float(point[1]) <= float(max_point[1])
    )


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
        art.set_joint_position_target_index(target=cur_joint_pos)
        try:
            zero_vel = torch.zeros_like(cur_joint_pos)
            art.set_joint_velocity_target_index(target=zero_vel)
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
            torch.zeros(len(env_ids), 6, device=env.device),
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
