# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bin clearing environment for double Piper: pick N objects from a source bin and place into destination."""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class BinClearingEnvironmentPiper(ExampleEnvironmentBase):

    name: str = "bin_clearing_piper"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On, RandomAroundSolution
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.bin_clearing_task import BinClearingTask
        from isaaclab_arena.utils.pose import Pose

        use_sequential_drop = args_cli.object_initialization == "sequential_drop"

        # Step 1: Background + ground plane (catches objects that fall off the table)
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")(
            initial_pose=Pose(position_xyz=(0.0, 0.0, -0.3)),
            spawner_cfg=sim_utils.GroundPlaneCfg(visible=False),
        )

        # Step 2: Table reference as anchor
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        if not use_sequential_drop:
            table_reference.add_relation(IsAnchor())

        # Step 3: Pick box (source) and place box (destination)
        _bin_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        )
        _box_collision_props = {
            "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
        }

        pick_box = self.asset_registry.get_asset_by_name(args_cli.pick_box)(
            instance_name="pick_box", spawn_cfg_addon=_box_collision_props
        )
        if not use_sequential_drop:
            pick_box.add_relation(IsAnchor())
        pick_box.set_initial_pose(
            Pose(
                position_xyz=(0.328, 0.0, 0.08 + args_cli.pick_box_z_offset),
                rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068),
            )
        )

        place_box = self.asset_registry.get_asset_by_name(args_cli.place_box)(
            instance_name="place_box", spawn_cfg_addon=_box_collision_props
        )
        if not use_sequential_drop:
            place_box.add_relation(IsAnchor())
        place_box.set_initial_pose(
            Pose(position_xyz=(0.75, 0.0, 0.08), rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068))
        )

        # Step 4: Randomly select N objects from pool and place in pick box
        # Scale randomization range down when many objects to help solver find valid layouts.
        # z_randomization should be kept small (<=0.02) to avoid objects spawning too high
        # and falling out of the pick box.
        num_objects = args_cli.num_objects
        x_half = min(0.05, 0.15 / max(num_objects, 1))
        y_half = min(0.1, 0.30 / max(num_objects, 1))

        # Enable per-body higher solver iterations to reduce penetration through bin walls.
        _bin_object_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=use_sequential_drop,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=8,
            max_depenetration_velocity=1.0,
            max_linear_velocity=2.0,
            max_angular_velocity=50.0,
            linear_damping=0.05,
            angular_damping=0.05,
            enable_gyroscopic_forces=True,
        )

        selected_names = random.choices(args_cli.object_pool, k=num_objects)
        pick_up_objects = []
        for i, obj_name in enumerate(selected_names):
            obj = self.asset_registry.get_asset_by_name(obj_name)(
                instance_name=f"pick_object_{i}",
                spawn_cfg_addon={
                    "rigid_props": _bin_object_rigid_props,
                    "collision_props": sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                        contact_offset=0.003,
                        rest_offset=0.0,
                    ),
                },
            )
            if not use_sequential_drop:
                obj.add_relation(On(pick_box))
                obj.add_relation(
                    RandomAroundSolution(
                        x_half_m=x_half,
                        y_half_m=y_half,
                        z_half_m=args_cli.z_randomization / 2,
                        z_offset_m=args_cli.z_randomization / 2,
                        yaw_half_rad=1.0,
                    )
                )
            pick_up_objects.append(obj)

        # Step 5: Lighting
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=args_cli.light_intensity),
        )
        if args_cli.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())

        # Step 6: Embodiment
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Step 7: Scene
        scene = Scene(
            assets=[
                background,
                ground_plane,
                light,
                pick_box,
                place_box,
                table_reference,
                *pick_up_objects,
            ]
        )

        # Step 8: Task
        episode_length_s = args_cli.num_objects * args_cli.time_per_object
        task = BinClearingTask(
            pick_up_object_list=pick_up_objects,
            destination_location=place_box,
            background_scene=background,
            episode_length_s=episode_length_s,
            force_threshold=args_cli.force_threshold,
            velocity_threshold=args_cli.velocity_threshold,
        )

        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            env_cfg.sim.dt = 1 / 300
            env_cfg.sim.physics_material = _bin_material
            from isaaclab_physx.physics import PhysxCfg
            if env_cfg.sim.physics is None:
                env_cfg.sim.physics = PhysxCfg()
            # Tunneling is mitigated via contact_offset on each object and higher solver iterations.
            env_cfg.sim.physics.solve_articulation_contact_last = True
            env_cfg.sim.physics.bounce_threshold_velocity = 0.2
            env_cfg.sim.physics.friction_offset_threshold = 0.01
            env_cfg.sim.physics.friction_correlation_distance = 0.00625
            env_cfg.sim.physics.gpu_max_num_partitions = 1

            # Appended after events_cfg is fully assembled (including placement_reset), so
            # this runs last within mode="reset" and always sees objects already placed.
            from isaaclab.managers import EventTermCfg

            if use_sequential_drop:
                from isaaclab_arena.tasks.events import drop_objects_into_pick_bin

                pick_box_bbox = pick_box.get_world_bounding_box()
                pick_box_min = pick_box_bbox.min_point[0].tolist()
                pick_box_max = pick_box_bbox.max_point[0].tolist()
                x_min = pick_box_min[0] + args_cli.drop_box_margin
                x_max = pick_box_max[0] - args_cli.drop_box_margin
                y_min = pick_box_min[1] + args_cli.drop_box_margin
                y_max = pick_box_max[1] - args_cli.drop_box_margin
                if args_cli.drop_x_half is not None:
                    x_center = (pick_box_min[0] + pick_box_max[0]) * 0.5
                    x_min = x_center - args_cli.drop_x_half
                    x_max = x_center + args_cli.drop_x_half
                if args_cli.drop_y_half is not None:
                    y_center = (pick_box_min[1] + pick_box_max[1]) * 0.5
                    y_min = y_center - args_cli.drop_y_half
                    y_max = y_center + args_cli.drop_y_half
                assert x_max > x_min, (
                    "Sequential-drop X range is empty. Reduce --drop_box_margin or set --drop_x_half explicitly."
                )
                assert y_max > y_min, (
                    "Sequential-drop Y range is empty. Reduce --drop_box_margin or set --drop_y_half explicitly."
                )
                z_min = pick_box_max[2] + args_cli.drop_height_min
                z_max = pick_box_max[2] + args_cli.drop_height_max
                if args_cli.drop_z_min is not None:
                    z_min = pick_box_max[2] + args_cli.drop_z_min
                if args_cli.drop_z_max is not None:
                    z_max = pick_box_max[2] + args_cli.drop_z_max
                z_min += args_cli.drop_area_z_offset
                z_max += args_cli.drop_area_z_offset
                assert z_max > z_min, "Sequential-drop Z range must satisfy max > min."
                print(
                    "[cfg] sequential-drop area: "
                    f"x=({x_min:.3f}, {x_max:.3f}), y=({y_min:.3f}, {y_max:.3f}), "
                    f"z=({z_min:.3f}, {z_max:.3f}); pick_box_z_offset={args_cli.pick_box_z_offset:.3f}, "
                    f"drop_area_z_offset={args_cli.drop_area_z_offset:.3f}, "
                    f"pick_box_bbox_min={pick_box_min}, pick_box_bbox_max={pick_box_max}",
                    flush=True,
                )

                setattr(
                    env_cfg.events,
                    "drop_objects_into_pick_bin",
                    EventTermCfg(
                        func=drop_objects_into_pick_bin,
                        mode="reset",
                        params={
                            "object_names": [obj.name for obj in pick_up_objects],
                            "pick_box_name": pick_box.name,
                            "drop_min_xyz": (x_min, y_min, z_min),
                            "drop_max_xyz": (x_max, y_max, z_max),
                            "pick_box_min_xyz": tuple(pick_box_min),
                            "pick_box_max_xyz": tuple(pick_box_max),
                            "yaw_half_rad": args_cli.drop_yaw_half,
                            "drop_interval_s": args_cli.drop_interval_s,
                            "collision_sync_steps": args_cli.drop_collision_sync_steps,
                            "final_settle_velocity_threshold": args_cli.settle_velocity_threshold,
                            "final_settle_max_steps": args_cli.settle_max_steps,
                            "debug_dir": args_cli.drop_debug_dir,
                        },
                    ),
                )
            else:
                from isaaclab_arena.tasks.events import wait_for_objects_to_settle

                setattr(
                    env_cfg.events,
                    "wait_for_objects_to_settle",
                    EventTermCfg(
                        func=wait_for_objects_to_settle,
                        mode="reset",
                        params={
                            "object_names": [obj.name for obj in pick_up_objects],
                            "velocity_threshold": args_cli.settle_velocity_threshold,
                            "max_steps": args_cli.settle_max_steps,
                        },
                    ),
                )
            print(
                f"[cfg] events fields after append = {list(env_cfg.events.__dict__.keys())}",
                flush=True,
            )
            print(
                f"[cfg] settle object_names = {[obj.name for obj in pick_up_objects]}",
                flush=True,
            )
            return env_cfg

        # Step 9: Assemble
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_set_viewer_cfg,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="double_piper_abs_joint_pos")
        parser.add_argument("--num_objects", type=int, default=3, help="Number of objects to place in pick box")
        parser.add_argument(
            "--time_per_object",
            type=float,
            default=20.0,
            help="Seconds allowed per object (episode_length = num_objects * time_per_object)",
        )
        parser.add_argument(
            "--object_pool",
            nargs="*",
            type=str,
            default=["rubiks_cube_hot3d_robolab"],
            help="Pool of object asset names to randomly select from",
        )
        parser.add_argument(
            "--z_randomization",
            type=float,
            default=0.02,
            help="Half-extent in Z for object spawn height randomization (meters)",
        )
        parser.add_argument(
            "--object_initialization",
            type=str,
            default="relation",
            choices=["relation", "sequential_drop"],
            help=(
                "Object initialization mode. 'relation' uses relation-solver placement; "
                "'sequential_drop' drops objects one at a time into the pick box during reset."
            ),
        )
        parser.add_argument(
            "--drop_x_half",
            type=float,
            default=None,
            help="Optional override for sequential-drop half-width in world X around the pick box center (meters)",
        )
        parser.add_argument(
            "--drop_y_half",
            type=float,
            default=None,
            help="Optional override for sequential-drop half-width in world Y around the pick box center (meters)",
        )
        parser.add_argument(
            "--drop_z_min",
            type=float,
            default=None,
            help="Optional override for minimum height above the pick box top for sequential-drop staging (meters)",
        )
        parser.add_argument(
            "--drop_z_max",
            type=float,
            default=None,
            help="Optional override for maximum height above the pick box top for sequential-drop staging (meters)",
        )
        parser.add_argument(
            "--drop_box_margin",
            type=float,
            default=0.06,
            help="Inset from each pick-box X/Y bounding-box side for automatic sequential-drop staging (meters)",
        )
        parser.add_argument(
            "--drop_height_min",
            type=float,
            default=0.03,
            help="Minimum automatic sequential-drop height above the pick box top (meters)",
        )
        parser.add_argument(
            "--drop_height_max",
            type=float,
            default=0.10,
            help="Maximum automatic sequential-drop height above the pick box top (meters)",
        )
        parser.add_argument(
            "--pick_box_z_offset",
            type=float,
            default=0.0,
            help="Debug offset applied to the pick-box initial Z position (meters)",
        )
        parser.add_argument(
            "--drop_area_z_offset",
            type=float,
            default=0.0,
            help="Debug offset applied to the sequential-drop Z sampling range (meters)",
        )
        parser.add_argument(
            "--drop_yaw_half",
            type=float,
            default=3.14159,
            help="Half-range of random yaw for sequential-drop staging (radians)",
        )
        parser.add_argument(
            "--drop_interval_s",
            type=float,
            default=1.0,
            help="Physics time to wait after releasing each object in sequential-drop mode (seconds)",
        )
        parser.add_argument(
            "--drop_collision_sync_steps",
            type=int,
            default=0,
            help="Physics steps to run with collision enabled and gravity disabled before each object is released",
        )
        parser.add_argument(
            "--drop_debug_dir",
            type=str,
            default="",
            help=(
                "Directory for sequential-drop reset camera debug images. Set to an empty string to disable. "
                "Images are saved once per staging/release/final-settle frame."
            ),
        )
        parser.add_argument("--pick_box", type=str, default="material_box_003_kinematic")
        parser.add_argument("--place_box", type=str, default="material_box_003_kinematic")
        parser.add_argument("--hdr", type=str, default=None)
        parser.add_argument("--light_intensity", type=float, default=500.0)
        parser.add_argument("--force_threshold", type=float, default=1.0, help="Contact force threshold for success")
        parser.add_argument("--velocity_threshold", type=float, default=0.5, help="Velocity threshold for success")
        parser.add_argument(
            "--settle_velocity_threshold",
            type=float,
            default=0.05,
            help="Max object linear/angular speed (m/s or rad/s) to consider objects settled after reset",
        )
        parser.add_argument(
            "--settle_max_steps",
            type=int,
            default=150,
            help="Max physics steps to wait for objects to settle after reset before giving up",
        )
        parser.add_argument("--teleop_device", type=str, default=None)
